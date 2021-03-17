import os
import re
import logging
from typing import Dict, Tuple, Callable

import h5py
import numpy as np

from ml4h.defines import TENSOR_EXT
from ml4h.tensormap.general import get_tensor_at_first_date


def token_dictionary_and_text_from_file(
        text_file: str,
        remove_special_chars: bool = True,
) -> Tuple[str, Dict[str, int]]:
    texts = []
    characters = set()
    with open(text_file) as file:
        for i, line in enumerate(file.readlines()):
            cur_line = _preprocess_sentence(line, remove_special_chars)
            [characters.add(char) for char in cur_line]
            texts.append(cur_line)
            if i % 50000 == 0:
                logging.info(f'Read {i} lines from {text_file}')
    logging.info(f'Total characters: {len(characters)}')
    char2index = dict((c, i) for i, c in enumerate(sorted(list(characters))))
    index2char = dict((i, c) for i, c in enumerate(sorted(list(characters))))
    logging.info(f'char2index:\n\n {char2index}  \n\n\n\n index2char: \n\n {index2char} \n\n\n')
    return ''.join(texts), char2index


def token_dictionary_from_hd5_key(
        tensors: str,
        path_prefix: str,
        name: str,
) -> Dict[str, int]:
    characters = set()
    for tp in os.listdir(tensors):
        if os.path.splitext(tp)[-1].lower() != TENSOR_EXT:
            continue
        with h5py.File(tensors + tp, 'r') as hd5:
            if name in hd5[path_prefix]:
                characters.update(np.unique(get_tensor_at_first_date(hd5, path_prefix, name)))
                break
    logging.info(f'Total characters from HD5 Tensor {path_prefix} and name {name}: {len(characters)}')
    char2index = dict((str(c), i) for i, c in enumerate(sorted(list(characters))))
    logging.info(f'char2index:\n {char2index} \n')
    return char2index


def random_text_window_tensor(
    text: str,
    window_size: int,
) -> Callable:
    def text_from_file(tm, _, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        random_index = np.random.randint(window_size, len(text)-window_size)
        for i, c in enumerate(text[random_index:random_index+window_size]):
            tensor[i] = tm.channel_map[c]
        if tm.dependent_map is not None:
            for i, dm in enumerate(tm.dependent_map):
                start_next_window = random_index+1+i
                dependents[dm] = np.zeros(dm.shape, dtype=np.float32)
                if dm.axes() == 1:
                    for j, c in enumerate(text[start_next_window:start_next_window+dm.shape[0]]):
                        dependents[dm][j] = dm.channel_map[c]
                else:
                    raise ValueError(f'No method to process dependent map:{dm.name} of shape {dm.shape}.')
                logging.debug(f'\nInput text: {text[random_index:random_index+window_size]}\n Dependent: {text[start_next_window:start_next_window+dm.shape[0]]}')
        return tensor
    return text_from_file


def _preprocess_sentence(sentence, remove_special_chars):
    sentence = sentence.strip()
    if remove_special_chars:
        #replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
    return sentence


def random_array_window_tensors(
    window_shape: Tuple[int],
    shift_axis: int = 0,
) -> Callable:
    def window_as_text_from_file(tm, hd5, dependents={}):
        full_tensor = get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
        indexes = [np.random.randint(window_shape[i], edge-window_shape[i]) for i, edge in enumerate(full_tensor.shape)]
        random_window = tuple(slice(index-window_shape[i], index) for i, index in enumerate(indexes))
        next_window1 = tuple(slice((index + 1 if i == shift_axis else index)-window_shape[i], index + 1 if i == shift_axis else index) for i, index in enumerate(indexes))
        next_window2 = tuple(slice((index + 2 if i == shift_axis else index)-window_shape[i], index + 2 if i == shift_axis else index) for i, index in enumerate(indexes))
        tensor = full_tensor[random_window].flatten().astype(str)
        if tm.dependent_map is not None:
            for dm, window in zip(tm.dependent_map, [next_window1, next_window2]):
                dependents[dm] = np.zeros(dm.shape, dtype=str)
                flat = full_tensor[window].flatten()
                for j, c in enumerate(flat):
                    dependents[dm][j] = dm.channel_map[str(c)]
        logging.debug(f'Full shape:{full_tensor.shape} window_shape:{window_shape} random idx:{indexes} tensor shape:{tensor.shape}')
        return tensor
    return window_as_text_from_file
