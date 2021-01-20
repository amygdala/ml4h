import os
import csv
import logging

import h5py
import numpy as np

from ml4h import TensorMap


def build_vector_tensor_from_file(
    file_name: str,
    normalization: bool = False,
    delimiter: str = '\t',
):
    """
    Build a tensor_from_file function from a column in a file.
    Only works for continuous values.
    When normalization is True values will be normalized according to the mean and std of all of the values in the column.
    """
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        table = {row[0]: np.array([float(row_value) for row_value in row[1:]]) for row in reader}
        if normalization:
            value_array = np.array([sub_array for sub_array in table.values()])
            mean = value_array.mean()
            std = value_array.std()
            logging.info(
                f'Normalizing TensorMap from file {file_name}, with mean: '
                f'{mean:.2f}, std: {std:.2f}', )

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if normalization:
            tm.normalization = {'mean': mean, 'std': std}
        try:
            return table[os.path.basename(hd5.filename).replace(TENSOR_EXT, '')]
        except KeyError:
            raise KeyError(f'User id not in file {file_name}.')

    return tensor_from_file


latent_file = '/home/sam/trained_models/cine_segmented_lax_4ch_diastole_autoencoder_64d/hidden_inference_cine_segmented_lax_4ch_diastole_autoencoder_64d.tsv'
embed_cine_segmented_lax_4ch_diastole = TensorMap('embed_cine_segmented_lax_4ch_diastole', shape=(64,), channel_map={f'latent_{i}': i for i in range(64)},
                                                  tensor_from_file=build_vector_tensor_from_file(latent_file),
                                                  )
