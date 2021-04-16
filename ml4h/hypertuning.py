# hyperparameters.py

# Imports
import gc
import os
import logging
import numpy as np
from timeit import default_timer as timer

import kerastuner as kt
from kerastuner.tuners import RandomSearch
from tensorflow import keras

import matplotlib
matplotlib.use('Agg') # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt # First import matplotlib, then use Agg, then import plt

from skimage.filters import threshold_otsu


from ml4h.arguments import parse_args
from ml4h.plots import plot_metric_history
from ml4h.defines import IMAGE_EXT, MODEL_EXT
from ml4h.models.train import train_model_from_generators
from ml4h.models.legacy_models import make_multimodal_multitask_model
from ml4h.models.model_factory import block_make_multimodal_multitask_model
from ml4h.tensor_generators import test_train_valid_tensor_generators, big_batch_from_minibatch_generator

MAX_LOSS = 9e9


def run(args):
    # Keep track of elapsed execution time
    start_time = timer()
    try:
        if 'conv' == args.mode:
            model_builder = make_model_builder(args)
        else:
            raise ValueError('Unknown hyper-parameter optimization mode:', args.mode)
        tuner = RandomSearch(
            model_builder,
            objective='val_loss',
            max_trials=args.max_models,
            executions_per_trial=3,
            directory=args.output_folder,
            project_name=args.id)
        generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
        tuner.search(generate_train,
                     epochs=args.epochs, steps_per_epoch=args.training_steps,
                     validation_data=generate_valid, validation_steps=args.validation_steps)
    except Exception as e:
        logging.exception(e)

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info(f"Executed the '{args.mode}' operation in {elapsed_time / 60.0:.1f} minutes")


def make_model_builder(args):
    def model_builder(hp):
        conv_layers_sets = [64, 48, 32, 24]
        args.__dict__['conv_layers'] = hp.Choice('conv_layers', values=conv_layers_sets),
        model, _, _, _ = block_make_multimodal_multitask_model(**args.__dict__)
        return model
    return model_builder


if __name__ == '__main__':
    args = parse_args()
    run(args)  # back to the top
