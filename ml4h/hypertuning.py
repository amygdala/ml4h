# hypertuning.py

# Imports
import logging
from timeit import default_timer as timer

import kerastuner as kt
from kerastuner.tuners import RandomSearch
from tensorflow import keras

from ml4h.arguments import parse_args
from ml4h.models.model_factory import block_make_multimodal_multitask_model
from ml4h.tensor_generators import test_train_valid_tensor_generators


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
            executions_per_trial=args.min_samples,
            directory=args.output_folder,
            project_name=args.id)
        generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
        tuner.search(generate_train,
                     epochs=args.epochs, steps_per_epoch=args.training_steps,
                     validation_data=generate_valid, validation_steps=args.validation_steps)
        [m.summary() for m in tuner.get_best_models(num_models=2)]
        logging.info(f"Tuning done best models above !")
    except Exception as e:
        logging.exception(e)

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info(f"Executed the '{args.mode}' operation in {elapsed_time / 60.0:.1f} minutes")


def make_model_builder(args):
    def model_builder(hp):
        num_conv_layers = hp.Int('num_conv_layers', 0, 4)
        conv_layer_size = hp.Int('conv_layer_size', 16, 128, sampling='log')
        args.__dict__['conv_layers'] = [conv_layer_size] * num_conv_layers
        num_dense_blocks = hp.Int('num_dense_blocks', 1, 6)
        dense_block_size = hp.Int('dense_block_size', 16, 128, sampling='log')
        args.__dict__['dense_blocks'] = [dense_block_size] * num_dense_blocks
        args.__dict__['block_size'] = hp.Int('block_size', 1, 7)
        num_dense_layers = hp.Int('num_dense_layers', 1, 4)
        dense_layer_size = hp.Int('dense_layer_size', 16, 128, sampling='log')
        args.__dict__['dense_layers'] = [dense_layer_size] * num_dense_layers
        model, _, _, _ = block_make_multimodal_multitask_model(**args.__dict__)
        return model
    return model_builder


if __name__ == '__main__':
    args = parse_args()
    run(args)  # back to the top
