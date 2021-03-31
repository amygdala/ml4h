import os
import re
import time
import json
from typing import Dict, List
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import xml.etree.ElementTree as et
import numpy as np
import blosc
import xxhash  # Checksums
import h5py
import zstandard  # Silently required by Parquet and Blosc

SAMPLE_RATE = 500
NUM_LEADS = 3
SECONDS_PER_MINUTE = 60


def _full_disclosure(root) -> np.ndarray:
    full_disclosure = root.find("./FullDisclosure/FullDisclosureData").text

    # comma delimited with random newlines and tab characters
    flat = re.sub("\n|\t", "", full_disclosure)
    ecg_raw = np.array(flat.split(",")[:-1], dtype=np.int16)

    # data is formatted like:
    # 500 samples lead I, 500 samples lead 2, 500 samples lead 3, 500 samples lead I, ...
    ecg_raw = ecg_raw.reshape((ecg_raw.shape[0] // SAMPLE_RATE, SAMPLE_RATE))
    leads = []
    for i in range(NUM_LEADS):
        # each row of ecg_raw is a different lead
        # get all of the lead i rows then flatten the result
        leads.append(ecg_raw[i::NUM_LEADS].ravel())
    return np.stack(leads).T


def _to_float_or_nan(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return np.nan


def _process_trend(root) -> Dict[str, np.ndarray]:
    # Trend measurements
    trend_entry_fields = ['HeartRate', 'Load', 'Grade', 'Mets', 'VECount', 'PaceCount']
    phase_to_int = {'Pretest': 0, 'Exercise': 1, 'Rest': 2}
    trend_entries = root.findall("./TrendData/TrendEntry")

    trends = defaultdict(list)
    for i, trend_entry in enumerate(trend_entries):
        for field in trend_entry_fields:
            field_val = trend_entry.find(field)
            field_val = _to_float_or_nan(field_val.text)
            trends[field].append(field_val)
        trends['PhaseTime'].append(
            SECONDS_PER_MINUTE * _to_float_or_nan(trend_entry.find("PhaseTime/Minute").text)
            + _to_float_or_nan(trend_entry.find("PhaseTime/Second").text)
        )
        trends['Artifact'].append(int(
            trend_entry.find('Artifact').text.replace("%", "")
        ))  # Artifact is reported as a percentage
        trends['time'].append(
            SECONDS_PER_MINUTE * _to_float_or_nan(trend_entry.find("EntryTime/Minute").text)
            + _to_float_or_nan(trend_entry.find("EntryTime/Second").text)
        )
        trends['PhaseName'].append(phase_to_int[trend_entry.find('PhaseName').text])

    return {trend_name: np.array(trend) for (trend_name, trend) in trends.items()}


def _phase_durations(root):
    phase_durations = {}
    for protocol in root.findall("./Protocol/Phase"):
        phase_name = protocol.find("PhaseName").text
        phase_duration = (
                SECONDS_PER_MINUTE * _to_float_or_nan(protocol.find("PhaseDuration/Minute").text)
                + _to_float_or_nan(protocol.find("PhaseDuration/Second").text)
        )
        phase_durations[phase_name] = int(phase_duration)
    return phase_durations


def _process_protocol(root) -> str:
    return root.find('./Protocol/Phase').find('ProtocolName').text


def _process_xml(xml_path: str):
    root = et.parse(xml_path).getroot()
    try:
        full_disclosure = _full_disclosure(root)
    except Exception as e:
        raise ValueError(f"Failed collecting full disclosure failed with {repr(e)}")
    try:
        trends = _process_trend(root)
    except Exception as e:
        print(f"Excepted error {repr(e)} during unnecesary trend extraction")
    try:
        protocol = _process_protocol(root)
    except Exception as e:
        raise ValueError(f"Failed collecting protocol with {repr(e)}")
    try:
        phase_durations = _phase_durations(root)
    except Exception as e:
        raise ValueError(f"Failed collecting phase durations with {repr(e)}")
    return full_disclosure, trends, protocol, phase_durations


# Compression and hd5 storage
def compress_and_store(
        hd5: h5py.File,
        data: np.ndarray,
        hd5_path: str,
):
    """Support function that takes arbitrary input data in the form of a Numpy array
    and compress, store, and checksum the data in a HDF5 file.
    Args:
        hd5 (h5py.File): Target HDF5-file handle.
        data (np.ndarray): Data to be compressed and saved.
        hd5_path (str): HDF5 dataframe path for the stored data.
    """
    data = data.copy(order='C')  # Required for xxhash
    compressed_data = blosc.compress(data.tobytes(), typesize=2, cname='zstd', clevel=9)
    hash_uncompressed = xxhash.xxh128_digest(data)
    hash_compressed = xxhash.xxh128_digest(compressed_data)
    decompressed = np.frombuffer(blosc.decompress(compressed_data), dtype=np.int16).reshape(data.shape)
    assert (xxhash.xxh128_digest(decompressed) == hash_uncompressed)
    dset = hd5.create_dataset(hd5_path, data=np.void(compressed_data))
    # Store meta data:
    # 1) Shape of the original tensor
    # 2) Hash of the compressed data
    # 3) Hash of the uncompressed data
    dset.attrs['shape'] = data.shape
    dset.attrs['hash_compressed'] = np.void(hash_compressed)
    dset.attrs['hash_uncompressed'] = np.void(hash_uncompressed)


def read_compressed(data_set: h5py.Dataset):
    shape = data_set.attrs['shape']
    return np.frombuffer(blosc.decompress(data_set[()]), dtype=np.int16).reshape(shape)


def _sample_id_from_path(path: str) -> int:
    return os.path.basename(path).split("_")[0]


def xml_to_hd5(xml_path: str, output_directory: str):
    sample_id = _sample_id_from_path(xml_path)
    instance = xml_path.split("_")[-2]

    full_disclosure, trends, protocol, phase_durations = _process_xml(xml_path)
    pretest = full_disclosure[:15 * SAMPLE_RATE]

    with h5py.File(os.path.join(output_directory, f"{sample_id}.h5"), "a") as hd5:
        compress_and_store(hd5, full_disclosure, f"full_disclosure/{instance}")
        compress_and_store(hd5, pretest, f"pretest/{instance}")

        hd5[f"protocol/{instance}"] = protocol
        for phase_name, duration in phase_durations.items():
            hd5[f"{phase_name}_duration/{instance}"] = duration


def _process_files(files: List[str], destination: str) -> Dict[str, str]:
    errors = {}
    name = _sample_id_from_path(files[0])

    print(f'Starting process {name} with {len(files)} files')
    for i, path in enumerate(files):
        try:
            xml_to_hd5(path, destination)
        except Exception as e:
            errors[path] = repr(e)
        if len(files) % max(i // 10, 1) == 0:
            print(f'{name}: {(i + 1) / len(files):.2%} done')

    return errors


def _partition_files(files: List[str], num_partitions: int) -> List[List[str]]:
    """
    Split files into num_partitions partitions of close to equal size.
    Partitioned by sample id, so no race conditions
    """
    id_to_file = defaultdict(list)
    for f in files:
        id_to_file[_sample_id_from_path(f)].append(f)
    sample_ids = np.array(list(id_to_file))
    np.random.shuffle(sample_ids)
    split_ids = np.array_split(sample_ids, num_partitions)
    splits = [
        sum((id_to_file[sample_id] for sample_id in split), [])
        for split in split_ids
    ]
    return [split for split in splits if split]  # lose empty splits


def multiprocess_ingest(
        files: List[str],
        destination: str,
):
    """Embarassingly parallel ingestion wrapper.

    Args:
        files (List[str]): Input list of files.
        destination (str): Output destination on disk.

    Returns:
        [dict]: Returns a dictionary of encountered errors.
    """
    print(f'Beginning ingestion of {len(files)} xmls.')
    os.makedirs(destination, exist_ok=True)
    start = time.time()
    # partition files by sample id so no race conditions across workers due to multiple instances
    split_files = _partition_files(files, cpu_count())
    errors = {}
    with Pool(cpu_count()) as pool:
        results = [pool.apply_async(_process_files, (split, destination)) for split in split_files]
        for result in results:
            errors.update(result.get())
    delta = time.time() - start
    print(f'Ingestion took {delta:.1f} seconds at {delta / len(files):.1f} s/file')
    with open(os.path.join(destination, 'errors.json'), 'w') as f:
        json.dump(errors, f)
    return errors
