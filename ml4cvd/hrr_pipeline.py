import os
import time
import h5py
import biosppy
import seaborn as sns
import logging
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, Any, Callable
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.keras import Model
from collections import namedtuple, defaultdict
import datetime
import gc

from ml4cvd.defines import TENSOR_EXT, MODEL_EXT
from ml4cvd.logger import load_config
from ml4cvd.TensorMap import TensorMap, Interpretation, no_nans
from ml4cvd.tensor_writer_ukbb import tensor_path, first_dataset_at_path
from ml4cvd.normalizer import Standardize, Normalizer, ZeroMeanStd1
from ml4cvd.tensor_from_file import _get_tensor_at_first_date
from ml4cvd.tensor_generators import test_train_valid_tensor_generators
from ml4cvd.models import train_model_from_generators, make_multimodal_multitask_model, BottleneckType
from ml4cvd.recipes import _infer_models
from ml4cvd.metrics import coefficient_of_determination


PRETEST_DUR = 15  # DURs are measured in seconds
EXERCISE_DUR = 360
RECOVERY_DUR = 60
SAMPLING_RATE = 500
HRR_TIME = 50
HR_MEASUREMENT_TIMES = 0, HRR_TIME  # relative to recovery start
HR_SEGMENT_DUR = 10  # HR measurements in recovery coalesced across a segment of this length
TREND_TRACE_DUR_DIFF = 2  # Sum of phase durations from UKBB is 2s longer than the raw traces
LEAD_NAMES = 'lead_I', 'lead_2', 'lead_3'

TENSOR_FOLDER = '/mnt/disks/ecg-bike-tensors/2019-10-10/'
REST_TENSOR_FOLDER = '/mnt/disks/ecg-rest-38k-tensors/2020-03-14/'
USER = 'ndiamant'
OUTPUT_FOLDER = f'/home/{USER}/ml/hrr_results_warp_dropout'
TRAIN_CSV_NAME = 'train_ids.csv'
VALID_CSV_NAME = 'valid_ids.csv'
TEST_CSV_NAME = 'test_ids.csv'

BIOSPPY_MEASUREMENTS_FILE = os.path.join(OUTPUT_FOLDER, 'biosppy_hr_recovery_measurements.csv')
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, 'figures')
BIOSPPY_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'biosppy')
AUGMENTATION_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'augmentations')

PRETEST_ECG_SUMMARY_STATS_CSV = os.path.join(OUTPUT_FOLDER, 'pretest_ecg_summary_stats.csv')
REST_ECG_SUMMARY_STATS_CSV = os.path.join(OUTPUT_FOLDER, 'rest_ecg_summary_stats.csv')

PRETEST_LABEL_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'pretest_labels')
PRETEST_QUANTILE_CUTOFF = .99
PRETEST_LABEL_FILE = os.path.join(OUTPUT_FOLDER, f'hr_pretest_training_data.csv')
PRETEST_TRAINING_DUR = 10  # number of seconds of pretest ECG used for prediction
VALIDATION_SPLIT = .1

DROPOUT = True
BATCH_NORM = True
AUG_RATE = .5
OVERWRITE_MODELS = False

PRETEST_MODEL_LEADS = [0]
SEED = 217
PRETEST_INFERENCE_NAME = 'pretest_model_inference.tsv'
REST_INFERENCE_FILE = os.path.join(OUTPUT_FOLDER, 'rest_model_inference.tsv')
K_SPLIT = 5
RESTING_HR_DF = os.path.join(OUTPUT_FOLDER, 'resting_hr.tsv')
RESTING_HR_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'resting_hr_stratification')


# Tensor from file helpers
def _check_phase_full_len(hd5: h5py.File, phase: str):
    phase_len = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', f'{phase}_duration')
    valid = True
    if phase == 'pretest':
        valid &= phase_len == PRETEST_DUR
    elif phase == 'exercise':
        valid &= phase_len == EXERCISE_DUR
    elif phase == 'rest':
        valid &= phase_len == RECOVERY_DUR
    else:
        raise ValueError(f'Phase {phase} is not a valid phase.')
    if not valid:
        raise ValueError(f'{phase} phase is not full length.')


def _get_bike_ecg(hd5: h5py.File, start: int, stop: int, leads: Union[List[int], slice]):
    path_prefix, name = 'ecg_bike/float_array', 'full'
    ecg_dataset = first_dataset_at_path(hd5, tensor_path(path_prefix, name))
    tensor = np.array(ecg_dataset[start: stop, leads], dtype=np.float32)
    return tensor


def _get_downsampled_bike_ecg(length: float, hd5: h5py.File, start: int, rate: float, leads: Union[List[int], slice]):
    length = int(length * rate)
    ecg = _get_bike_ecg(hd5, start, start + length, leads)
    ecg = _downsample_ecg(ecg, rate)
    return ecg


def _make_pretest_ecg_tff(downsample_rate: float, leads: Union[List[int], slice], random_start=True):
    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        _check_phase_full_len(hd5, 'pretest')
        start = np.random.randint(0, SAMPLING_RATE * PRETEST_DUR - tm.shape[0] * downsample_rate) if random_start else 0
        return _get_downsampled_bike_ecg(tm.shape[0], hd5, start, downsample_rate, leads)
    return tff


def _get_trace_recovery_start(hd5: h5py.File) -> int:
    _check_phase_full_len(hd5, 'rest')
    _check_phase_full_len(hd5, 'pretest')
    pretest_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'pretest_duration')
    exercise_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
    return int(SAMPLING_RATE * (pretest_dur + exercise_dur - HR_SEGMENT_DUR / 2 - TREND_TRACE_DUR_DIFF))


# Rest ECG
REST_CHANNEL_MAP = {'strip_I': 0}
REST_PREFIX = 'ukb_ecg_rest'


def _rest_ecg(
        hd5: h5py.File, shape: Tuple[int, ...], path_prefix: str, channel_map: Dict[str, int],
        downsample_rate: float,
) -> np.ndarray:
    tensor = np.zeros(shape, dtype=np.float32)
    for k, idx in channel_map.items():
        data = np.array(TensorMap.hd5_first_dataset_in_group(hd5, f'{path_prefix}/{k}/'))[:, np.newaxis]
        tensor[:, channel_map[k]] = _downsample_ecg(data, downsample_rate)[:, 0]
    return tensor


def _make_downsampled_rest_tff(downsample_rate: float):
    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        return _rest_ecg(hd5, tm.shape, tm.path_prefix, tm.channel_map, downsample_rate)
    return tff


# ECG summary stats
ECG_MEAN_COL = 'ecg_mean'
ECG_STD_COL = 'ecg_std'


def _pretest_mean_std(sample_id: int) -> Dict[str, float]:
    if str(sample_id).endswith('000'):
        logging.info(f'Processing sample_id {sample_id}.')
    with h5py.File(_path_from_sample_id(str(sample_id)), 'r') as hd5:
        pretest = _get_bike_ecg(hd5, 0, PRETEST_DUR * SAMPLING_RATE, PRETEST_MODEL_LEADS)
        return {'sample_id': sample_id, ECG_MEAN_COL: pretest.mean(), ECG_STD_COL: pretest.std()}


def _rest_mean_std(sample_id: int) -> Dict[str, float]:
    if str(sample_id).endswith('000'):
        logging.info(f'Processing sample_id {sample_id}.')
    with h5py.File(os.path.join(REST_TENSOR_FOLDER, f'{sample_id}{TENSOR_EXT}'), 'r') as hd5:
        ecg = _rest_ecg(
            hd5, (SAMPLING_RATE * PRETEST_TRAINING_DUR, len(REST_CHANNEL_MAP)), REST_PREFIX,
            REST_CHANNEL_MAP, 1,
        )
        return {'sample_id': sample_id, ECG_MEAN_COL: ecg.mean(), ECG_STD_COL: ecg.std()}


# ECG transformations
def _warp_ecg(ecg):
    warp_strength = .02
    i = np.linspace(0, 1, len(ecg))
    envelope = warp_strength * (.5 - np.abs(.5 - i))
    warped = i + envelope * (
        np.sin(np.random.rand() * 5 + np.random.randn() * 5)
        + np.cos(np.random.rand() * 5 + np.random.randn() * 5)
    )
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


def _random_crop_ecg(ecg):
    cropped_ecg = ecg.copy()
    for j in range(ecg.shape[1]):
        crop_len = np.random.randint(len(ecg)) // 3
        crop_start = max(0, np.random.randint(-crop_len, len(ecg)))
        cropped_ecg[:, j][crop_start: crop_start + crop_len] = np.random.randn()
    return cropped_ecg


def _downsample_ecg(ecg, rate: float):
    """
    rate=2 halves the sampling rate. Uses linear interpolation. Requires ECG to be divisible by rate.
    """
    new_len = ecg.shape[0] // rate
    i = np.linspace(0, 1, new_len)
    x = np.linspace(0, 1, ecg.shape[0])
    downsampled = np.zeros((ecg.shape[0] // rate, ecg.shape[1]))
    for j in range(ecg.shape[1]):
        downsampled[:, j] = np.interp(i, x, ecg[:, j])
    return downsampled


def _rand_add_noise(ecg):
    noise_frac = np.random.rand() * .2
    return ecg + noise_frac * ecg.std(axis=0) * np.random.randn(*ecg.shape)


def _apply_aug_rate(augmentation: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: augmentation(a) if np.random.rand() < AUG_RATE else a


# HR measurements from biosppy
BIOSPPY_DOWNSAMPLE_RATE = 4


def _get_segment_for_biosppy(ecg, mid_time: int):
    center = mid_time * SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE
    offset = (SAMPLING_RATE * HR_SEGMENT_DUR // BIOSPPY_DOWNSAMPLE_RATE) // 2
    return ecg[center - offset: center + offset]


def _get_biosppy_hr(segment: np.ndarray) -> float:
    return float(
        np.median(
            biosppy.signals.ecg.ecg(segment, sampling_rate=SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE, show=False)[-1],
        ),
    )


def _get_segments_for_biosppy(hd5: h5py.File):
    recovery_start_idx = _get_trace_recovery_start(hd5)
    length = (HR_MEASUREMENT_TIMES[-1] - HR_MEASUREMENT_TIMES[0] + HR_SEGMENT_DUR) * SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE
    ecg = _get_downsampled_bike_ecg(length, hd5, recovery_start_idx, BIOSPPY_DOWNSAMPLE_RATE, [0, 1, 2])
    for mid_time in HR_MEASUREMENT_TIMES:
        yield _get_segment_for_biosppy(ecg, mid_time + HR_SEGMENT_DUR // 2)


def _hr_and_diffs_from_segment(segment: np.ndarray) -> Tuple[float, float]:
    hr_per_lead = [_get_biosppy_hr(segment[:, i]) for i in range(segment.shape[-1])]
    max_diff = max(map(lambda pair: abs(pair[0] - pair[1]), combinations(hr_per_lead, 2)))
    return float(np.median(hr_per_lead)), max_diff


def _plot_segment(segment: np.ndarray):
    hr, max_diff = _hr_and_diffs_from_segment(segment)
    t = np.linspace(0, HR_SEGMENT_DUR, len(segment))
    for i, lead_name in enumerate(LEAD_NAMES):
        plt.plot(t, segment[:, i], label=lead_name)
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title(f'hr: {hr:.2f}, max hr difference between leads: {max_diff:.2f}')


def plot_segment_prediction(sample_id: str, t: int, pred: float, actual: float, diff: float):
    t_idx = HR_MEASUREMENT_TIMES.index(t)
    with h5py.File(_path_from_sample_id(sample_id), 'r') as hd5:
        segment = list(_get_segments_for_biosppy(hd5))[t_idx]
        x = np.linspace(0, HR_SEGMENT_DUR, len(segment))
        for i, lead_name in enumerate(LEAD_NAMES):
            plt.title(
                '\n'.join([
                    f'{sample_id} at time {t} after recovery',
                    f'biosppy hr {actual:.2f}',
                    f'model hr {pred:.2f}',
                    f'biosppy lead difference {diff:.2f}',
                ]),
            )
            plt.plot(x, segment[:, i], label=lead_name)


def _recovery_hrs_biosppy(hd5: h5py.File) -> List[Tuple[float, float]]:
    return list(map(_hr_and_diffs_from_segment, _get_segments_for_biosppy(hd5)))


def _path_from_sample_id(sample_id: str) -> str:
    return os.path.join(TENSOR_FOLDER, sample_id + TENSOR_EXT)


def _sample_id_from_hd5(hd5: h5py.File) -> int:
    return int(os.path.basename(hd5.filename).replace(TENSOR_EXT, ''))


def _sample_id_from_path(path: str) -> int:
    return int(os.path.basename(path).replace(TENSOR_EXT, ''))


def _plot_recovery_hrs(path: str):
    num_plots = len(HR_MEASUREMENT_TIMES)
    plt.figure(figsize=(10, 3 * num_plots))
    try:
        with h5py.File(path, 'r') as hd5:
            for i, segment in enumerate(_get_segments_for_biosppy(hd5)):
                plt.subplot(num_plots, 1, i + 1)
                _plot_segment(segment)
            plt.tight_layout()
            plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, f'biosppy_hr_recovery_measurements_{_sample_id_from_hd5(hd5)}.png'))
    except (ValueError, KeyError, OSError) as e:
        logging.debug(f'Plotting failed for {path} with error {e}.')


def df_hr_col(t):
    return f'{t}_hr'


def df_hrr_col(t):
    return f'{t}_hrr'


def df_diff_col(t):
    return f'{t}_diff'


DF_HR_COLS = [df_hr_col(t) for t in HR_MEASUREMENT_TIMES]
DF_DIFF_COLS = [df_diff_col(t) for t in HR_MEASUREMENT_TIMES]


def _recovery_hrs_from_path(path: str):
    sample_id = os.path.basename(path).replace(TENSOR_EXT, '')
    if sample_id.endswith('000'):
        logging.info(f'Processing sample_id {sample_id}.')
    hr_diff = np.full((len(HR_MEASUREMENT_TIMES), 2), np.nan)
    error = None
    try:
        with h5py.File(path, 'r') as hd5:
            hr_diff = np.array(_recovery_hrs_biosppy(hd5))
    except (ValueError, KeyError, OSError) as e:
        error = e
    measures = {'sample_id': sample_id, 'error': error}
    for i, (hr_col, diff_col) in enumerate(zip(DF_HR_COLS, DF_DIFF_COLS)):
        measures[hr_col] = hr_diff[i, 0]
        measures[diff_col] = hr_diff[i, 1]
    return measures


def plot_hr_from_biosppy_summary_stats():
    df = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE)

    # HR summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_HR_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna()
        sns.distplot(x, label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, 'biosppy_hr_recovery_measurements_summary_stats.png'))

    # HR lead diff summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_DIFF_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna().copy()
        sns.distplot(x[x < 5], label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, 'biosppy_hr_diff_recovery_measurements_summary_stats.png'))

    # Random sample of hr trends
    plt.figure(figsize=(15, 7))
    trend_samples = df[DF_HR_COLS].sample(1000).values
    plt.plot(HR_MEASUREMENT_TIMES, (trend_samples - trend_samples[:, :1]).T, alpha=.2, linewidth=1, c='k')
    plt.axhline(0, c='k', linestyle='--')
    plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, 'biosppy_hr_trend_samples.png'))

    # correlation heat map
    plt.figure(figsize=(7, 7))
    sns.heatmap(df[DF_HR_COLS + DF_DIFF_COLS].corr(), annot=True, cbar=False)
    plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, 'biosppy_correlations.png'))
    plt.close()


def plot_pretest_label_summary_stats():
    df = pd.read_csv(PRETEST_LABEL_FILE)

    # HR summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_HR_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna()
        sns.distplot(x, label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'pretest_training_labels_summary_stats.png'))

    # Random sample of hr trends
    plt.figure(figsize=(15, 7))
    trend_samples = df[DF_HR_COLS].sample(1000).values
    plt.plot(HR_MEASUREMENT_TIMES, (trend_samples - trend_samples[:, :1]).T, alpha=.2, linewidth=1, c='k')
    plt.axhline(0, c='k', linestyle='--')
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'pretest_training_labels_hr_trend_samples.png'))

    # correlation heat map
    plt.figure(figsize=(7, 7))
    sns.heatmap(df[DF_HR_COLS].corr(), annot=True, cbar=False)
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'biosppy_correlations.png'))
    plt.close()


def build_hr_biosppy_measurements_csv():
    paths = [os.path.join(TENSOR_FOLDER, p) for p in sorted(os.listdir(TENSOR_FOLDER)) if p.endswith(TENSOR_EXT)]
    logging.info('Plotting 10 random hr measurements from biosppy.')
    for path in np.random.choice(paths, 10):
        _plot_recovery_hrs(path)
    pool = Pool()
    logging.info('Beginning to get hr measurements from biosppy.')
    now = time.time()
    measures = pool.map(_recovery_hrs_from_path, paths)
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting hr measurements from biosppy took {delta_t // 60} minutes at {delta_t / len(paths):.2f}s per path.')
    df.to_csv(BIOSPPY_MEASUREMENTS_FILE, index=False)


def build_pretest_summary_stats_df(sample_ids: List[int]) -> pd.DataFrame:
    pool = Pool()
    logging.info('Beginning to get pretest ecg means and stds.')
    now = time.time()
    measures = pool.map(_pretest_mean_std, sample_ids)
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting pretest ecg means and stds took {delta_t // 60} minutes at {delta_t / len(sample_ids):.2f}s per path.')
    return df


def build_rest_summary_stats_df(sample_ids: List[int]) -> pd.DataFrame:
    pool = Pool()
    logging.info('Beginning to get rest ecg means and stds.')
    now = time.time()
    measures = pool.map(_rest_mean_std, sample_ids)
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting rest ecg means and stds took {delta_t // 60} minutes at {delta_t / len(sample_ids):.2f}s per path.')
    return df


def make_pretest_labels(make_ecg_summary_stats: bool):
    biosppy_labels = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE)
    new_df = pd.DataFrame()
    hr_0 = biosppy_labels[df_hr_col(HR_MEASUREMENT_TIMES[0])]
    logging.info(f'Label error counts: {biosppy_labels["error"].value_counts()}')
    drop_idx = {'ECG missing or incomplete': biosppy_labels['error'].notnull()}
    new_df['sample_id'] = biosppy_labels['sample_id']
    double_sided_quantile = (1 - PRETEST_QUANTILE_CUTOFF) / 2
    for t in HR_MEASUREMENT_TIMES:
        hr_name = df_hr_col(t)
        hr = biosppy_labels[hr_name]
        new_df[hr_name] = hr
        diff = biosppy_labels[df_diff_col(t)]
        drop_idx[f'diff {t} too high'] = diff > diff.quantile(PRETEST_QUANTILE_CUTOFF)
        drop_idx[f'hr {t} outside center {PRETEST_QUANTILE_CUTOFF:.2%}'] = (hr > hr.quantile(1 - double_sided_quantile)) | (hr < hr.quantile(double_sided_quantile))
        new_df[hr_name] = hr
        if t != 0:
            hrr = hr_0 - hr
            hrr_name = df_hrr_col(t)
            new_df[hrr_name] = hrr
            drop_idx[f'hrr {t} outside center {PRETEST_QUANTILE_CUTOFF:.2%}'] = (hrr > hrr.quantile(1 - double_sided_quantile)) | (hrr < hrr.quantile(double_sided_quantile))
            new_df[hrr_name] = hrr

    logging.info(f'Pretest labels starting at length {len(new_df)}.')
    all_drop = False
    for name, idx in drop_idx.items():
        logging.info(f'Due to filter {name}, dropping {(idx & ~all_drop).sum()} values')
        all_drop |= idx
    new_df = new_df[~all_drop]
    unknown_errors = new_df.isna().any(axis=1)
    logging.info(f'Dropping {unknown_errors.sum()} due to unknown biosppy errors.')
    new_df = new_df[~unknown_errors]  # TODO: why needed?
    assert new_df.notna().all().all()
    logging.info(f'There are {len(new_df)} pretest labels after filtering hr measures.')

    if make_ecg_summary_stats:
        pretest_df = build_pretest_summary_stats_df(new_df['sample_id'])
        pretest_df.to_csv(PRETEST_ECG_SUMMARY_STATS_CSV, index=False)
    else:
        pretest_df = pd.read_csv(PRETEST_ECG_SUMMARY_STATS_CSV)
    new_df = new_df.merge(pretest_df, on='sample_id')

    mean_low, mean_high = np.quantile(new_df[ECG_MEAN_COL], [double_sided_quantile, 1 - double_sided_quantile])
    mean_drop = (new_df[ECG_MEAN_COL] < mean_high) & (new_df[ECG_MEAN_COL] > mean_low)
    logging.info(f'Due to pretest mean outside center {PRETEST_QUANTILE_CUTOFF:.2%}, dropping {(~mean_drop).sum()}.')
    new_df = new_df[mean_drop]
    std_low, std_high = np.quantile(new_df[ECG_STD_COL], [double_sided_quantile, 1 - double_sided_quantile])
    std_drop = (new_df[ECG_STD_COL] < std_high) & (new_df[ECG_STD_COL] > std_low)
    logging.info(f'Due to pretest std outside center {PRETEST_QUANTILE_CUTOFF:.2%}, dropping {(~std_drop).sum()}.')
    new_df = new_df[std_drop]
    logging.info(f'There are {len(new_df)} pretest labels after filtering ecg ranges.')

    new_df.to_csv(PRETEST_LABEL_FILE, index=False)


# hr tmaps
def _hr_file(file_name: str, t: int, hrr=False):
    error = None
    try:
        df = pd.read_csv(file_name, dtype={'sample_id': int})
        df = df.set_index('sample_id')
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        sample_id = _sample_id_from_hd5(hd5)
        try:
            row = df.loc[sample_id]
            hr = row[df_hr_col(t)]
            if hrr:
                peak = row[df_hr_col(0)]
                out = peak - hr
            else:
                out = hr
            return np.array([out])
        except KeyError:
            raise KeyError(f'Sample id not in {file_name} for TensorMap {tm.name}.')
    return tensor_from_file


def split_folder_name(split_idx: int) -> str:
    return os.path.join(OUTPUT_FOLDER, f'split_{split_idx}')


def _split_train_name(split_idx: int) -> str:
    return os.path.join(split_folder_name(split_idx), TRAIN_CSV_NAME)


def _split_valid_name(split_idx: int) -> str:
    return os.path.join(split_folder_name(split_idx), VALID_CSV_NAME)


def _split_test_name(split_idx: int) -> str:
    return os.path.join(split_folder_name(split_idx), TEST_CSV_NAME)


# build cohort
def build_csvs():
    sample_ids = pd.read_csv(PRETEST_LABEL_FILE)['sample_id']
    split_ids = _split_ids(ids=sample_ids, n_split=K_SPLIT, validation_frac=.1)
    for i, (train_ids, valid_ids, test_ids) in enumerate(split_ids):
        pd.DataFrame({'sample_id': train_ids}).to_csv(_split_train_name(i), index=False)
        pd.DataFrame({'sample_id': valid_ids}).to_csv(_split_valid_name(i), index=False)
        pd.DataFrame({'sample_id': test_ids}).to_csv(_split_test_name(i), index=False)


def _get_hrr_summary_stats(id_csv: str) -> Tuple[float, float]:
    df = pd.read_csv(PRETEST_LABEL_FILE)
    ids = pd.read_csv(id_csv)
    hrr = df[df_hrr_col(HRR_TIME)][df['sample_id'].isin(ids['sample_id'])]
    return hrr.mean(), hrr.std()


def _get_pretest_summary_stats(id_csv: str) -> Tuple[float, float]:
    df = pd.read_csv(PRETEST_LABEL_FILE)
    ids = pd.read_csv(id_csv)
    mean = df[ECG_MEAN_COL][df['sample_id'].isin(ids['sample_id'])].mean()
    std = df[ECG_STD_COL][df['sample_id'].isin(ids['sample_id'])].mean()
    return mean, std


ModelSetting = namedtuple('ModelSetting', ['model_id', 'downsample_rate', 'augmentations', 'shift'])


AUGMENTATIONS = [_warp_ecg, _random_crop_ecg, _rand_add_noise]
MODEL_SETTINGS = [
    ModelSetting(**{'model_id': 'baseline_model', 'downsample_rate': 1, 'augmentations': [], 'shift': False}),
    ModelSetting(**{'model_id': 'shift', 'downsample_rate': 1, 'augmentations': [], 'shift': True}),
    ModelSetting(**{'model_id': 'shift_augment', 'downsample_rate': 1, 'augmentations': AUGMENTATIONS, 'shift': True}),
    ModelSetting(**{'model_id': 'downsample_model', 'downsample_rate': BIOSPPY_DOWNSAMPLE_RATE, 'augmentations': [], 'shift': True}),
    ModelSetting(**{'model_id': 'downsample_augment', 'downsample_rate': BIOSPPY_DOWNSAMPLE_RATE, 'augmentations': AUGMENTATIONS, 'shift': True}),
]


# Augmentation demonstrations
Augmentation = Callable[[np.ndarray], np.ndarray]


def _demo_augmentations(hd5_path: str, setting: ModelSetting):
    tmap = _make_ecg_tmap(setting, 0)
    num_samples = 5
    ax_size = 10
    t = np.linspace(0, PRETEST_TRAINING_DUR, tmap.shape[0])
    with h5py.File(hd5_path, 'r') as hd5:
        ecg = tmap.tensor_from_file(tmap, hd5)
        fig, axes = plt.subplots(
            nrows=num_samples, ncols=1, figsize=(ax_size * 2, num_samples * ax_size), sharex='all',
        )
        orig = tmap.postprocess_tensor(ecg, augment=False, hd5=hd5)[:, 0]
        axes[0].set_title(f'Augmentation Samples for model {setting.model_id}')
        for ax in axes:
            ax.plot(t, orig, c='k', label='Original ECG')
            ax.plot(
                t, tmap.postprocess_tensor(ecg, augment=True, hd5=hd5)[:, 0],
                c='r', alpha=.5, label='Augmented ECG',
            )
            ax.legend()
    plt.savefig(os.path.join(AUGMENTATION_FIGURE_FOLDER, f'{setting.model_id}_{_sample_id_from_path(hd5_path)}.png'))


RESTING_HR = TensorMap('resting_hr', path_prefix='ecg_bike', shape=(1,))


def _resting_hr(sample_id: int) -> Dict[str, float]:
    if str(sample_id).endswith('000'):
        logging.info(f'Processing sample_id {sample_id}.')
    with h5py.File(_path_from_sample_id(str(sample_id)), 'r') as hd5:
        try:
            hr = RESTING_HR.tensor_from_file(RESTING_HR, hd5)
        except KeyError:
            hr = np.nan
    return {'sample_id': sample_id, 'resting_hr': hr}


def _resting_hr_df(paths: List[str]) -> pd.DataFrame:
    if os.path.exists(RESTING_HR_DF):
        return pd.read_csv(RESTING_HR_DF, sep='\t')
    pool = Pool()
    logging.info('Beginning to get resting HRs.')
    now = time.time()
    measures = pool.map(_recovery_hrs_from_path, paths)
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting resting HRs took {delta_t // 60} minutes at {delta_t / len(paths):.2f}s per path.')
    df.to_csv(RESTING_HR_DF, index=False, sep='\t')
    return df


def resting_hr_explore(setting: ModelSetting):
    paths = [os.path.join(TENSOR_FOLDER, p) for p in sorted(os.listdir(TENSOR_FOLDER)) if p.endswith(TENSOR_EXT)]
    resting_hr = _resting_hr_df(paths)
    infer_df = pd.read_csv(PRETEST_INFERENCE_NAME)
    target_cols = [time_to_pred_hrr_col(HRR_TIME, setting.model_id), time_to_actual_hrr_col(HRR_TIME)]
    pretest_hr = infer_df[['sample_id'] + target_cols]
    df = pretest_hr.merge(resting_hr, on='sample_id')
    ax_size = 10
    for low, high in (0, .01), (.495, .505), (.99, 1):
        low_cut, high_cut = np.quantile(df['resting_hr'], [low, high])
        df_slice = df[(df['resting_hr'] < high_cut) & (df['resting_hr'] > low_cut)]
        for col in target_cols:
            low_row = df_slice[col].argmin()
            high_row = df_slice[col].argmax()
            fig, axes = plt.subplots(
                nrows=3, ncols=1, figsize=(ax_size * 3, ax_size), sharex='all',
            )
            sns.distplot(df_slice[col], ax=axes[0], label=col)
            axes[0].axvline(low_row[col], label=f'Low {col}', color='k', linestyle='--')
            axes[0].axvline(high_row[col], label=f'High {col}', color='k', linestyle='--')
            axes[0].set_title(
                f'Differences in {col} for {df_slice["resting_hr"].min():.2f} < resting hr < {df_slice["resting_hr"].max()}'
            )

            for ax, row in zip(axes[1:], [low_row, high_row]):
                sample_id = row['sample_id']
                with h5py.File(_path_from_sample_id(str(sample_id)), 'r') as hd5:
                    pretest = _get_bike_ecg(hd5, 0, PRETEST_DUR * SAMPLING_RATE, [0])
                ax.plot(np.linspace(0, PRETEST_TRAINING_DUR, len(pretest)), pretest, c='k', label=f'ECG for {col} = {row[col]:.2f}')
                ax.legend()
            plt.savefig(os.path.join(RESTING_HR_FIGURE_FOLDER, f'{setting.model_id}_{col}_{low}_{high}.png'))


# Model training
def _make_ecg_tmap(setting: ModelSetting, split_idx: int) -> TensorMap:
    normalizer = Standardize(*_get_pretest_summary_stats(_split_train_name(split_idx)))
    augmentations = [_apply_aug_rate(aug) for aug in setting.augmentations]
    return TensorMap(
        f'pretest_ecg_downsample_{setting.downsample_rate}',
        shape=(int(PRETEST_TRAINING_DUR * SAMPLING_RATE // setting.downsample_rate), len(PRETEST_MODEL_LEADS)),
        interpretation=Interpretation.CONTINUOUS,
        validator=no_nans, normalization=normalizer,
        tensor_from_file=_make_pretest_ecg_tff(setting.downsample_rate, PRETEST_MODEL_LEADS, random_start=setting.shift),
        cacheable=False, augmentations=augmentations,
    )


def _make_rest_tmap(setting: ModelSetting) -> TensorMap:
    if not os.path.exists(REST_ECG_SUMMARY_STATS_CSV):
        sample_ids = [_sample_id_from_path(path) for path in os.listdir(REST_TENSOR_FOLDER) if path.endswith(TENSOR_EXT)]
        df = build_rest_summary_stats_df(sample_ids)
        df.to_csv(REST_ECG_SUMMARY_STATS_CSV, index=False)
    else:
        df = pd.read_csv(REST_ECG_SUMMARY_STATS_CSV)
    # normalizer = Standardize(df[ECG_MEAN_COL].mean(), df[ECG_STD_COL].mean())
    normalizer = ZeroMeanStd1()
    return TensorMap(
        f'pretest_ecg_downsample_{setting.downsample_rate}',
        shape=(int(PRETEST_TRAINING_DUR * SAMPLING_RATE // setting.downsample_rate), len(PRETEST_MODEL_LEADS)),
        interpretation=Interpretation.CONTINUOUS,
        validator=no_nans, normalization=normalizer,
        tensor_from_file=_make_downsampled_rest_tff(setting.downsample_rate),
        cacheable=False, channel_map=REST_CHANNEL_MAP,
        path_prefix=REST_PREFIX,
    )


def _make_hrr_tmap(split_idx: int) -> TensorMap:
    normalizer = Standardize(*_get_hrr_summary_stats(_split_train_name(split_idx)))
    return TensorMap(
        df_hrr_col(HRR_TIME), shape=(1,), metrics=[],
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=_hr_file(PRETEST_LABEL_FILE, HRR_TIME, hrr=True),
        normalization=normalizer,
    )


def make_pretest_model(setting: ModelSetting, split_idx: int, load_model: bool):
    pretest_tmap = _make_ecg_tmap(setting, split_idx)
    hrr_tmap = _make_hrr_tmap(split_idx)
    model_path = pretest_model_file(split_idx, setting.model_id)
    return make_multimodal_multitask_model(
        tensor_maps_in=[pretest_tmap],
        tensor_maps_out=[hrr_tmap],
        activation='swish',
        learning_rate=1e-3,
        bottleneck_type=BottleneckType.GlobalAveragePoolStructured,
        optimizer='radam',
        dense_layers=[64],
        conv_layers=[32],
        dense_blocks=[16, 24, 32],
        conv_type='conv',
        conv_normalize='batch_norm' if BATCH_NORM else None,
        conv_x=[64],
        conv_y=[1],
        conv_z=[1],
        pool_type='max',
        pool_x=2,
        block_size=3,
        model_file=model_path if load_model else None,
        conv_regularize='spatial_dropout' if DROPOUT else None,
        conv_regularize_rate=.1 if DROPOUT else 0,
    )


def _split_ids(ids: np.ndarray, n_split: int, validation_frac: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """output is [(train_ids, valid_ids, test_ids)]"""
    validation_frac = validation_frac * n_split / (n_split - 1)
    kf = KFold(n_splits=n_split, random_state=SEED, shuffle=True)
    out = []
    for train_idx, test_idx in kf.split(ids):
        train, valid = train_test_split(ids[train_idx], test_size=validation_frac)
        out.append((train, valid, ids[test_idx]))
    return out


def pretest_model_file(split_idx: int, model_id: str) -> str:
    return os.path.join(split_folder_name(split_idx), model_id, model_id + MODEL_EXT)


def history_tsv(split_idx: int, model_id: str) -> str:
    return os.path.join(split_folder_name(split_idx), model_id, 'history.tsv')


def _train_pretest_model(
        setting: ModelSetting, split_idx: int,
) -> Tuple[Any, Dict]:
    workers = cpu_count() * 2
    patience = 5
    epochs = 200
    batch_size = 256

    train_csv = _split_train_name(split_idx)
    valid_csv = _split_valid_name(split_idx)
    test_csv = _split_test_name(split_idx)

    pretest_tmap = _make_ecg_tmap(setting, split_idx)
    hrr_tmap = _make_hrr_tmap(split_idx)

    train_len = len(pd.read_csv(train_csv))
    valid_len = len(pd.read_csv(valid_csv))
    training_steps = train_len // batch_size
    validation_steps = valid_len // batch_size * (2 if setting.shift else 1)

    generate_train, generate_valid, _ = test_train_valid_tensor_generators(
        tensor_maps_in=[pretest_tmap],
        tensor_maps_out=[hrr_tmap],
        tensors=TENSOR_FOLDER,
        batch_size=batch_size,
        num_workers=workers,
        cache_size=1e7,
        balance_csvs=[],
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
        training_steps=training_steps,
        validation_steps=validation_steps,
    )

    model = make_pretest_model(setting, split_idx, False)
    logging.info(f'Beginning training with {training_steps} training steps and {validation_steps} validation steps.')
    try:
        model, history = train_model_from_generators(
            model, generate_train, generate_valid, training_steps, validation_steps, batch_size,
            epochs, patience, split_folder_name(split_idx), setting.model_id, True, True, return_history=True,
        )
        history_df = pd.DataFrame(history.history)
        history_df['model_id'] = setting.model_id
        history_df['split_idx'] = split_idx
        history_df.to_csv(history_tsv(split_idx, setting.model_id), sep='\t', index=False)
    finally:
        generate_train.kill_workers()
        generate_valid.kill_workers()
        gc.collect()
    return model, history


# Inference
ACTUAL_POSTFIX = '_actual'
PRED_POSTFIX = '_prediction'


def _inference_file(split_idx: int) -> str:
    return os.path.join(split_folder_name(split_idx), PRETEST_INFERENCE_NAME)


def tmap_to_actual_col(tmap: TensorMap):
    return f'{tmap.name}{ACTUAL_POSTFIX}'


def tmap_to_pred_col(tmap: TensorMap, model_id: str):
    return f'{tmap.name}_{model_id}{PRED_POSTFIX}'


def time_to_pred_hr_col(t: int, model_id: str):
    return f'{df_hr_col(t)}_{model_id}{PRED_POSTFIX}'


def time_to_pred_hrr_col(t: int, model_id: str):
    return f'{df_hrr_col(t)}_{model_id}{PRED_POSTFIX}'


def time_to_actual_hr_col(t: int):
    return f'{df_hr_col(t)}{ACTUAL_POSTFIX}'


def time_to_actual_hrr_col(t: int):
    return f'{df_hrr_col(t)}{ACTUAL_POSTFIX}'


def _infer_models_split_idx(split_idx: int):
    tensor_paths = [
        _path_from_sample_id(str(sample_id)) for
        sample_id in pd.read_csv(_split_test_name(split_idx))['sample_id']
    ]
    models = [make_pretest_model(setting, split_idx, True) for setting in MODEL_SETTINGS]
    model_ids = [setting.model_id for setting in MODEL_SETTINGS]
    tmaps_in = [_make_ecg_tmap(setting, split_idx) for setting in MODEL_SETTINGS]
    tmaps_out = [_make_hrr_tmap(split_idx)]
    _infer_models(
        models=models,
        model_ids=model_ids,
        tensor_maps_in=tmaps_in,
        tensor_maps_out=tmaps_out,
        inference_tsv=_inference_file(split_idx), num_workers=8, batch_size=128, tensor_paths=tensor_paths,
    )


def _dummy_tff(_, __, ___):
    return np.zeros(1)


def _rest_model_name(setting: ModelSetting, split_idx: int) -> str:
    return f'{setting.model_id}_split_{split_idx}'


def _infer_rest_models():
    logging.info('Beginning inference on rest ECGs')
    tensor_paths = [
        os.path.join(REST_TENSOR_FOLDER, path)
        for path in os.listdir(REST_TENSOR_FOLDER) if path.endswith(TENSOR_EXT)
    ]
    setting = MODEL_SETTINGS[-1]
    models = [make_pretest_model(setting, split_idx, True) for split_idx in range(K_SPLIT)]
    model_ids = [_rest_model_name(setting, split_idx) for split_idx in range(K_SPLIT)]
    tmaps_in = [_make_rest_tmap(setting)]
    normalize = Standardize(*_get_hrr_summary_stats(PRETEST_LABEL_FILE))
    tmaps_out = [
        TensorMap(
            df_hrr_col(HRR_TIME), shape=(1,), tensor_from_file=_dummy_tff,
            normalization=normalize,
        ),
    ]
    _infer_models(
        models=models,
        model_ids=model_ids,
        tensor_maps_in=tmaps_in,
        tensor_maps_out=tmaps_out,
        inference_tsv=REST_INFERENCE_FILE, num_workers=8, batch_size=128, tensor_paths=tensor_paths,
    )


# result plotting
def _scatter_plot(ax, truth, prediction, title):
    ax.plot([np.min(truth), np.max(truth)], [np.min(truth), np.max(truth)], linewidth=2)
    ax.plot([np.min(prediction), np.max(prediction)], [np.min(prediction), np.max(prediction)], linewidth=4)
    pearson = np.corrcoef(prediction, truth)[1, 0]  # corrcoef returns full covariance matrix
    big_r_squared = coefficient_of_determination(truth, prediction)
    logging.info(f'{title} - pearson:{pearson:0.3f} r^2:{pearson*pearson:0.3f} R^2:{big_r_squared:0.3f}')
    ax.scatter(prediction, truth, label=f'Pearson:{pearson:0.3f} r^2:{pearson * pearson:0.3f} R^2:{big_r_squared:0.3f}', marker='.', s=1)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Actual')
    ax.set_title(title + '\n')
    ax.legend(loc="lower right")


def _dist_plot(ax, truth, prediction, title):
    ax.set_title(title)
    ax.legend(loc="lower right")
    sns.distplot(prediction, label='Predicted', color='r', ax=ax)
    sns.distplot(truth, label='Truth', color='b', ax=ax)
    ax.legend(loc="upper left")


def bootstrap_compare_models(
        model_ids: List[str], inference_result: pd.DataFrame,
        num_bootstraps: int = 100, bootstrap_frac: float = .5,
) -> pd.DataFrame:
    performance = {'model': [], 'R2': []}
    actual_col = time_to_actual_hrr_col(HRR_TIME)
    pred_cols = {m_id: time_to_pred_hrr_col(HRR_TIME, m_id) for m_id in model_ids}
    for _ in range(num_bootstraps):
        df = inference_result.sample(frac=bootstrap_frac, replace=True)
        for m_id, pred_col in pred_cols.items():
            pred = df[pred_col]
            R2 = coefficient_of_determination(df[actual_col], pred)
            performance['model'].append(m_id)
            performance['R2'].append(R2)
    return pd.DataFrame(performance)


def _evaluate_models():
    inference_dfs = []
    for i in range(K_SPLIT):
        inference_df = pd.read_csv(_inference_file(i), sep='\t')
        inference_df['split_idx'] = i
        inference_dfs.append(inference_df)
    inference_df = pd.concat(inference_dfs)
    inference_df.to_csv(os.path.join(OUTPUT_FOLDER, PRETEST_INFERENCE_NAME), sep='\t', index=False)

    R2_dfs = []
    ax_size = 10
    figure_folder = os.path.join(FIGURE_FOLDER, f'model_results')
    os.makedirs(figure_folder, exist_ok=True)
    for setting in MODEL_SETTINGS:
        m_id = setting.model_id
        _, ax = plt.subplots(figsize=(ax_size, ax_size))
        pred = inference_df[time_to_pred_hrr_col(HRR_TIME, m_id)]
        actual = inference_df[time_to_actual_hrr_col(HRR_TIME)]
        _scatter_plot(ax, actual, pred, f'HRR at recovery time {HRR_TIME}')
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, f'{m_id}_model_correlations.png'))

        # distributions of predicted and actual measurements
        _, ax = plt.subplots(figsize=(ax_size, ax_size))
        _dist_plot(ax, actual, pred, f'HRR at recovery time {HRR_TIME}')
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, f'{m_id}_distributions.png'))

        R2s = [
            coefficient_of_determination(
                actual[inference_df['split_idx'] == i], pred[inference_df['split_idx'] == i],
            ) for i in range(K_SPLIT)
        ]
        R2_df = pd.DataFrame({'R2': R2s})
        R2_df['model'] = m_id
        R2_dfs.append(R2_df)
        plt.close('all')

    R2_df = pd.concat(R2_dfs)
    plt.figure(figsize=(ax_size, ax_size))
    sns.boxplot(x='model', y='R2', data=R2_df)
    plt.savefig(os.path.join(figure_folder, f'model_performance_comparison_{K_SPLIT}_fold.png'))
    plt.close('all')

    model_ids = list(R2_df['model'].unique())
    logging.info('Beginning bootstrap performance evaluation.')
    R2_df = bootstrap_compare_models(model_ids, inference_df, num_bootstraps=5000, bootstrap_frac=1)

    plt.figure(figsize=(ax_size, ax_size))
    sns.violinplot(x='model', y='R2', data=R2_df)
    plt.savefig(os.path.join(figure_folder, f'bootstrap_violin.png'))

    plt.figure(figsize=(ax_size, ax_size))
    sns.boxplot(x='model', y='R2', data=R2_df)
    plt.savefig(os.path.join(figure_folder, f'bootstrap_box.png'))

    plt.figure(figsize=(ax_size, ax_size))
    cmap = cm.get_cmap('rainbow')
    final_model = MODEL_SETTINGS[-1].model_id
    final_R2 = R2_df['R2'][R2_df['model'] == final_model].values
    for i, m_id in enumerate(model_ids):
        R2 = R2_df['R2'][R2_df['model'] == m_id].values
        color = cmap(i / K_SPLIT)
        sns.distplot(R2, color=color)
        plt.axvline(R2.mean(), color=color, label=f'{m_id} mean ({R2.mean():.3f})', linestyle='--')
        logging.info(f'Probability {final_model} is not better than {m_id} (null hypothesis) is {(R2 >= final_R2).mean():.3%}.')
    plt.xlabel('R2')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(figure_folder, f'bootstrap_distributions.png'))

    plt.figure(figsize=(ax_size, ax_size))
    cmap = cm.get_cmap('rainbow')
    for i, m_id in enumerate(model_ids[:-1]):
        R2 = R2_df['R2'][R2_df['model'] == m_id].values
        diff = R2 - final_R2
        color = cmap(i / K_SPLIT)
        sns.distplot(diff, color=color)
        plt.axvline(diff.mean(), color=color, label=f'({m_id} R2 - {final_model} R2) mean ({diff.mean():.3f})', linestyle='--')
    plt.axvline(0, c='k', linestyle='--')
    plt.xlabel('R2')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(figure_folder, f'bootstrap_diff_distributions.png'))
    plt.close('all')


def plot_training_curves():
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)
    for setting in MODEL_SETTINGS:
        model_id = setting.model_id
        model_losses = []
        model_val_losses = []
        for split_idx in range(K_SPLIT):
            history = pd.read_csv(history_tsv(split_idx, model_id), sep='\t')
            model_losses.append(history['loss'])
            model_val_losses.append(history['val_loss'])
        max_len = max(map(len, model_losses))
        loss_array = np.full((K_SPLIT, max_len), np.nan)
        val_loss_array = np.full((K_SPLIT, max_len), np.nan)
        for loss, val_loss, split_idx in zip(model_losses, model_val_losses, range(K_SPLIT)):
            loss_array[split_idx, :len(loss)] = loss
            val_loss_array[split_idx, :len(loss)] = val_loss

        epoch = list(range(max_len))
        ax1.plot(epoch, loss_array.mean(axis=0), label=f'{setting.model_id} mean loss')
        ax1.fill_between(
            epoch, loss_array.min(axis=0), loss_array.max(axis=0),
            label=f'{setting.model_id} min and max loss', alpha=.2,
        )
        ax2.plot(epoch, val_loss_array.mean(axis=0), label=f'{setting.model_id} mean validation loss')
        ax2.fill_between(
            epoch, val_loss_array.min(axis=0), val_loss_array.max(axis=0),
            label=f'{setting.model_id} min and max validation loss', alpha=.2,
        )
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    plt.tight_layout()
    figure_folder = os.path.join(FIGURE_FOLDER, f'model_results')
    plt.savefig(os.path.join(figure_folder, f'training_curves.png'))


def plot_rest_inference():
    figure_folder = os.path.join(FIGURE_FOLDER, 'rest_results')
    os.makedirs(figure_folder, exist_ok=True)

    df = pd.read_csv(REST_INFERENCE_FILE, sep='\t')
    pred_cols = [
        time_to_pred_hrr_col(HRR_TIME, _rest_model_name(MODEL_SETTINGS[-1], split_idx))
        for split_idx in range(K_SPLIT)
    ]
    mean = df[pred_cols].mean(axis=1)
    ax_size = 10
    plt.figure(figsize=(ax_size, ax_size))
    sns.distplot(mean)
    plt.title('Rest ECG inference mean')
    plt.xlabel('Predicted HRR')
    plt.savefig(os.path.join(figure_folder, 'rest_mean.png'))

    std = df[pred_cols].std(axis=1)
    ax_size = 10
    plt.figure(figsize=(ax_size, ax_size))
    sns.distplot(std)
    plt.title('Rest ECG inference std')
    plt.xlabel('Predicted HRR std')
    plt.savefig(os.path.join(figure_folder, 'rest_std.png'))

    quantiles = .01, .99
    cutoffs = np.quantile(std, quantiles)
    num_samples = 2
    ecg_tmap = _make_rest_tmap(MODEL_SETTINGS[-1])
    t = np.linspace(0, PRETEST_TRAINING_DUR, ecg_tmap.shape[0])
    for quantile, cutoff in zip(quantiles, cutoffs):
        if quantile < .5:
            rows = df[std < cutoff].sample(num_samples)
        else:
            rows = df[std > cutoff].sample(num_samples)
        for i in range(num_samples):
            sample_id = rows['sample_id'].iloc[i]
            sample_mean = rows[pred_cols].iloc[i].mean()
            sample_std = rows[pred_cols].iloc[i].std()
            plt.figure(figsize=(ax_size, ax_size / 2))
            with h5py.File(os.path.join(REST_TENSOR_FOLDER, f'{sample_id}{TENSOR_EXT}'), 'r') as hd5:
                ecg = ecg_tmap.normalize(ecg_tmap.tensor_from_file(ecg_tmap, hd5))
            plt.plot(t, ecg, c='k')
            logging.info(f'Plotting rest ecg sample id {sample_id}')
            plt.title(f'{quantile} std quantile - std {sample_std:.2f} mean {sample_mean:.2f}')
            plt.savefig(os.path.join(figure_folder, f'{sample_id}_std_quantile_{quantile}.png'))


if __name__ == '__main__':
    """Always remakes figures"""
    np.random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    os.makedirs(BIOSPPY_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(PRETEST_LABEL_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(AUGMENTATION_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(RESTING_HR_FIGURE_FOLDER, exist_ok=True)
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    load_config('INFO', OUTPUT_FOLDER, 'log_' + now_string, USER)

    MAKE_LABELS = False or not os.path.exists(BIOSPPY_MEASUREMENTS_FILE)
    for i in range(K_SPLIT):
        os.makedirs(split_folder_name(i), exist_ok=True)
    MAKE_ECG_SUMMARY_STATS = False or not os.path.exists(PRETEST_ECG_SUMMARY_STATS_CSV)
    MAKE_SPLIT_CSVS = False or not all(
        os.path.exists(_split_train_name(i)) for i in range(K_SPLIT)
    )
    TRAIN_PRETEST_MODELS = False or not all(
        os.path.exists(pretest_model_file(i, MODEL_SETTINGS[0].model_id))
        for i in range(K_SPLIT)
    )
    INFER_PRETEST_MODELS = (
            False or TRAIN_PRETEST_MODELS
            or not all(os.path.exists(_inference_file(split_idx)) for split_idx in range(K_SPLIT))
    )
    INFER_REST_MODELS = False or not os.path.exists(REST_INFERENCE_FILE)

    if MAKE_LABELS:
        logging.info('Making biosppy labels.')
        build_hr_biosppy_measurements_csv()
    plot_hr_from_biosppy_summary_stats()
    plt.close('all')
    make_pretest_labels(MAKE_ECG_SUMMARY_STATS)
    plot_pretest_label_summary_stats()
    if MAKE_SPLIT_CSVS:
        build_csvs()
    aug_demo_paths = np.random.choice(sorted(os.listdir(TENSOR_FOLDER)), 1)
    for setting in MODEL_SETTINGS:
        for path in aug_demo_paths:
            path = os.path.join(TENSOR_FOLDER, path)
            _demo_augmentations(path, setting)
    if TRAIN_PRETEST_MODELS:
        for i in range(K_SPLIT):
            for setting in MODEL_SETTINGS:
                if os.path.exists(history_tsv(split_idx=i, model_id=setting.model_id)) and not OVERWRITE_MODELS:
                    logging.info(f'Skipping {setting.model_id} in split {i} since it already exists.')
                    continue
                _train_pretest_model(setting, i)
                plt.close('all')
    if INFER_PRETEST_MODELS:
        for i in range(K_SPLIT):
            logging.info(f'Running inference on split {i}.')
            _infer_models_split_idx(i)
    _evaluate_models()
    plot_training_curves()
    if INFER_REST_MODELS:
        _infer_rest_models()
    plot_rest_inference()
    resting_hr_explore(MODEL_SETTINGS[-1])
