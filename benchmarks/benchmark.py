import os
import time
import datetime
import pandas as pd
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Generator
from abc import ABC, abstractmethod
from ml4h.defines import StorageType
from contextlib import contextmanager
from multiprocessing import cpu_count

from benchmarks.data import build_tensor_maps, build_hd5s_ukbb, get_hd5_paths, DataDescription


DELTA_COL = 'step delta'
WORKER_COL = 'num workers'
BATCH_SIZE_COL = 'batch size'
NAME_COL = 'name'


class GeneratorFactory(ABC):
    is_setup = False

    @abstractmethod
    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        pass

    @abstractmethod
    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class TensorGeneratorFactory(GeneratorFactory):

    def __init__(self, compression: str):
        super().__init__()
        self.compression = compression
        self.tmaps = None
        self.paths = None

    def get_name(self) -> str:
        return f'TensorGenerator_{self.compression}'

    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        build_hd5s_ukbb(data_descriptions, num_samples, overwrite=True, compression=self.compression)
        self.paths = get_hd5_paths(True, num_samples)
        self.tmaps = build_tensor_maps(data_descriptions)

    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        from ml4h.tensor_generators import TensorGenerator
        gen = TensorGenerator(
            batch_size=batch_size, num_workers=num_workers,
            input_maps=self.tmaps, output_maps=[],
            cache_size=0, paths=self.paths,
        )
        yield gen
        gen.kill_workers()
        del gen


FACTORIES = [
    TensorGeneratorFactory('gzip'),
    TensorGeneratorFactory('lzf'),
]


def benchmark_generator(num_steps: int, gen: Generator) -> List[float]:
    times = []
    for i in range(num_steps):
        start = time.time()
        next(gen)
        times.append(time.time() - start)
        print(f'{(i + 1) / num_steps:.1%} done', end='\r')
    print()
    return times


def benchmark_generator_factory(
        generator_factory: GeneratorFactory,
        batch_sizes: List[int], workers: List[int],
        num_steps: int,
) -> pd.DataFrame:
    result_dfs = []
    for batch_size, num_workers in product(batch_sizes, workers):
        with generator_factory(batch_size, num_workers) as gen:
            start = time.time()
            print(f'Beginning test at batch size {batch_size}, workers {num_workers}')
            deltas = benchmark_generator(num_steps // batch_size, gen)
            print(f'Test at batch size {batch_size}, workers {num_workers} took {time.time() - start:.1f}s')
        result_df = pd.DataFrame({DELTA_COL: deltas})
        result_df[BATCH_SIZE_COL] = batch_size
        result_df[WORKER_COL] = num_workers
        result_dfs.append(result_df)
    return pd.concat(result_dfs)


class Benchmark:

    def __init__(
            self, data_descriptions: List[DataDescription], num_samples: int,
            batch_sizes: List[int], num_workers: List[int],
    ):
        self.data_descriptions = data_descriptions
        self.num_samples = num_samples
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers

    def run(self, factories: List[GeneratorFactory]) -> pd.DataFrame:
        performance_dfs = []
        for factory in factories:
            name = factory.get_name()
            print(f'------------ {name} ------------')
            factory.setup(self.num_samples, self.data_descriptions)
            performance_df = benchmark_generator_factory(
                factory, self.batch_sizes, self.num_workers, self.num_samples,
            )
            performance_df[NAME_COL] = name
            performance_dfs.append(performance_df)
        return pd.concat(performance_dfs)


ECG_BENCHMARK = Benchmark(
    [
        ('ecg', (5000, 12), StorageType.CONTINUOUS),
        ('bmi', (1,), StorageType.CONTINUOUS),
    ],
    4096, [64, 128, 256], [1, 2, 4, 8]
)
MRI_BENCHMARK = Benchmark(
    [
        ('mri', (256, 256, 16), StorageType.CONTINUOUS),
        ('segmentation', (256, 256, 16), StorageType.CONTINUOUS),
    ],
    256, [4, 8, 16], [1, 2, 4, 8]
)
ECG_MULTITASK_BENCHMARK = Benchmark(
    (
        [('ecg', (5000, 12), StorageType.CONTINUOUS)]
        + [(f'interval_{i}', (1,), StorageType.CONTINUOUS) for i in range(20)]
    ),
    4096, [4, 8, 16], [1, 2, 4, 8]
)
TEST_BENCHMARK = Benchmark(
    (
        [('ecg', (5000, 12), StorageType.CONTINUOUS)]
    ),
    16, [1, 2], [1, 2, 4]
)
BENCHMARKS = {
    'test': TEST_BENCHMARK,
    'ecg_single_task': ECG_BENCHMARK,
    'mri_single_task': MRI_BENCHMARK,
    'ecg_multi_task': ECG_MULTITASK_BENCHMARK,
}


def plot_benchmark(performance_df: pd.DataFrame, save_path: str):
    performance_df['samples / sec'] = 1 / performance_df[DELTA_COL] * performance_df[BATCH_SIZE_COL]
    plt.figure(figsize=(performance_df[BATCH_SIZE_COL].nunique() * 6, 6))
    sns.catplot(
        data=performance_df, kind='point',
        hue=NAME_COL, y='samples / sec', x=WORKER_COL, col=BATCH_SIZE_COL,
    )
    plt.savefig(save_path, dpi=200)


def run_benchmark(benchmark_name: str):
    performance_df = BENCHMARKS[benchmark_name].run(FACTORIES)
    output_folder = os.path.join(os.path.dirname(__file__), 'benchmark_results', benchmark_name)
    date = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    description = f'{date}_cpus-{cpu_count()}'
    os.makedirs(output_folder, exist_ok=True)
    performance_df.to_csv(
        os.path.join(output_folder, f'{description}_results.tsv'),
        sep='\t', index=False,
    )
    plot_benchmark(performance_df, os.path.join(output_folder, f'{description}_plot.png'))


if __name__ == '__main__':
    # TODO: add memory and line profiling
    parser = ArgumentParser()
    parser.add_argument(
        '--benchmarks', required=False, nargs='*',
        help='Benchmarks to run. If no argument is provided, all will be run.',
    )
    args = parser.parse_args()
    benchmarks = args.benchmarks or list(BENCHMARKS)
    print(f'Will run benchmarks: {", ".join(benchmarks)}')
    for benchmark in benchmarks:
        print('======================================')
        print(f'Running benchmark {benchmark}')
        print('======================================')
        run_benchmark(benchmark)
