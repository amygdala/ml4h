# recipes.py

# Imports
import os
import csv
import copy
import glob
import logging
import numpy as np
from functools import reduce
from timeit import default_timer as timer
from collections import Counter, defaultdict

from ml4h.arguments import parse_args
from ml4h.models.inspect import saliency_map
from ml4h.optimizers import find_learning_rate
from ml4h.defines import TENSOR_EXT, MODEL_EXT
from ml4h.models.train import train_model_from_generators
from ml4h.tensormap.tensor_map_maker import write_tensor_maps
from ml4h.tensorize.tensor_writer_mgb import write_tensors_mgb
from ml4h.models.model_factory import block_make_multimodal_multitask_model
from ml4h.tensor_generators import BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX
from ml4h.explorations import mri_dates, ecg_dates, predictions_to_pngs, sample_from_language_model
from ml4h.explorations import plot_while_learning, plot_histograms_of_tensors_in_pdf, cross_reference
from ml4h.explorations import test_labels_to_label_map, infer_with_pixels, explore, latent_space_dataframe
from ml4h.tensor_generators import TensorGenerator, test_train_valid_tensor_generators, big_batch_from_minibatch_generator
from ml4h.metrics import get_roc_aucs, get_precision_recall_aucs, get_pearson_coefficients, log_aucs, log_pearson_coefficients
from ml4h.plots import evaluate_predictions, plot_scatters, plot_rocs, plot_precision_recalls, subplot_roc_per_class, plot_tsne
from ml4h.plots import plot_reconstruction, plot_hit_to_miss_transforms, plot_saliency_maps, plot_partners_ecgs, plot_ecg_rest_mp
from ml4h.plots import subplot_rocs, subplot_comparison_rocs, subplot_scatters, subplot_comparison_scatters, plot_prediction_calibrations
from ml4h.models.legacy_models import make_character_model_plus, embed_model_predict, make_siamese_model, make_multimodal_multitask_model
from ml4h.models.legacy_models import get_model_inputs_outputs, make_shallow_model, make_hidden_layer_model, make_paired_autoencoder_model
from ml4h.tensorize.tensor_writer_ukbb import write_tensors, append_fields_from_csv, append_gene_csv, write_tensors_from_dicom_pngs, write_tensors_from_ecg_pngs


def run(args):
    start_time = timer()  # Keep track of elapsed execution time
    try:
        if 'tensorize' == args.mode:
            write_tensors(
                args.id, args.xml_folder, args.zip_folder, args.output_folder, args.tensors, args.dicoms, args.mri_field_ids, args.xml_field_ids,
                args.write_pngs, args.min_sample_id, args.max_sample_id, args.min_values,
            )
        elif 'tensorize_pngs' == args.mode:
            write_tensors_from_dicom_pngs(args.tensors, args.dicoms, args.app_csv, args.dicom_series, args.min_sample_id, args.max_sample_id, args.x, args.y)
        elif 'tensorize_ecg_pngs' == args.mode:
            write_tensors_from_ecg_pngs(args.tensors, args.xml_folder, args.min_sample_id, args.max_sample_id)
        elif 'tensorize_partners' == args.mode:
            write_tensors_mgb(args.xml_folder, args.tensors, args.num_workers)
        elif 'explore' == args.mode:
            explore(args)
        elif 'cross_reference' == args.mode:
            cross_reference(args)
        elif 'train' == args.mode:
            train_multimodal_multitask(args)
        elif 'test' == args.mode:
            test_multimodal_multitask(args)
        elif 'compare' == args.mode:
            compare_multimodal_multitask_models(args)
        elif 'infer' == args.mode:
            infer_multimodal_multitask(args)
        elif 'infer_hidden' == args.mode:
            infer_hidden_layer_multimodal_multitask(args)
        elif 'infer_pixels' == args.mode:
            infer_with_pixels(args)
        elif 'infer_encoders' == args.mode:
            infer_encoders_block_multimodal_multitask(args)
        elif 'test_scalar' == args.mode:
            test_multimodal_scalar_tasks(args)
        elif 'train_block' == args.mode:
            train_block(args)
        elif 'compare_scalar' == args.mode:
            compare_multimodal_scalar_task_models(args)
        elif 'plot_predictions' == args.mode:
            plot_predictions(args)
        elif 'plot_while_training' == args.mode:
            plot_while_training(args)
        elif 'plot_saliency' == args.mode:
            saliency_maps(args)
        elif 'plot_mri_dates' == args.mode:
            mri_dates(args.tensors, args.output_folder, args.id)
        elif 'plot_ecg_dates' == args.mode:
            ecg_dates(args.tensors, args.output_folder, args.id)
        elif 'plot_histograms' == args.mode:
            plot_histograms_of_tensors_in_pdf(args.id, args.tensors, args.output_folder, args.max_samples)
        elif 'plot_resting_ecgs' == args.mode:
            plot_ecg_rest_mp(args.tensors, args.min_sample_id, args.max_sample_id, args.output_folder, args.num_workers)
        elif 'plot_partners_ecgs' == args.mode:
            plot_partners_ecgs(args)
        elif 'train_shallow' == args.mode:
            train_shallow_model(args)
        elif 'train_char' == args.mode:
            train_char_model(args)
        elif 'train_siamese' == args.mode:
            train_siamese_model(args)
        elif 'train_paired' == args.mode:
            train_paired_model(args)
        elif 'inspect_paired' == args.mode:
            inspect_paired_model(args)
        elif 'write_tensor_maps' == args.mode:
            write_tensor_maps(args)
        elif 'append_continuous_csv' == args.mode:
            append_fields_from_csv(args.tensors, args.app_csv, 'continuous', ',')
        elif 'append_categorical_csv' == args.mode:
            append_fields_from_csv(args.tensors, args.app_csv, 'categorical', ',')
        elif 'append_continuous_tsv' == args.mode:
            append_fields_from_csv(args.tensors, args.app_csv, 'continuous', '\t')
        elif 'append_categorical_tsv' == args.mode:
            append_fields_from_csv(args.tensors, args.app_csv, 'categorical', '\t')
        elif 'append_gene_csv' == args.mode:
            append_gene_csv(args.tensors, args.app_csv, ',')
        elif 'pca_on_hidden_inference' == args.mode:
            pca_on_hidden_inference(args)
        elif 'find_learning_rate' == args.mode:
            _find_learning_rate(args)
        elif 'find_learning_rate_and_train' == args.mode:
            args.learning_rate = _find_learning_rate(args)
            if not args.learning_rate:
                raise ValueError('Could not find learning rate.')
            train_multimodal_multitask(args)
        else:
            raise ValueError('Unknown mode:', args.mode)

    except Exception as e:
        logging.exception(e)

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info("Executed the '{}' operation in {:.2f} seconds".format(args.mode, elapsed_time))


def _find_learning_rate(args) -> float:
    schedule = args.learning_rate_schedule
    args.learning_rate_schedule = None  # learning rate schedule interferes with setting lr done by find_learning_rate
    generate_train, _, _ = test_train_valid_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    lr = find_learning_rate(model, generate_train, args.training_steps, os.path.join(args.output_folder, args.id))
    args.learning_rate_schedule = schedule
    return lr


def train_multimodal_multitask(args):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size, args.epochs,
        args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels, args.tensor_maps_out,
        save_last_model=args.save_last_model
    )

    out_path = os.path.join(args.output_folder, args.id + '/')
    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    return _predict_and_evaluate(
        model, test_data, test_labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, out_path, test_paths, args.embed_visualization, args.alpha,
    )


def train_block(args):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model, encoders, decoders, merger = block_make_multimodal_multitask_model(**args.__dict__)
    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size, args.epochs,
        args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels, args.tensor_maps_out,
        save_last_model=args.save_last_model
    )
    for tm in encoders:
        encoders[tm].save(f'{args.output_folder}{args.id}/encoder_{tm.name}.h5')
    for tm in decoders:
        decoders[tm].save(f'{args.output_folder}{args.id}/decoder_{tm.name}.h5')
    if merger:
        merger.save(f'{args.output_folder}{args.id}/merger.h5')

    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    performance_metrics = _predict_and_evaluate(
        model, test_data, test_labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, os.path.join(args.output_folder, args.id + '/'), test_paths, args.embed_visualization, args.alpha,
    )

    predictions_list = model.predict(test_data)
    samples = min(args.test_steps * args.batch_size, 12)
    out_path = os.path.join(args.output_folder, args.id, 'reconstructions/')
    if len(args.tensor_maps_out) == 1:
        predictions_list = [predictions_list]
    predictions_dict = {name: pred for name, pred in zip(model.output_names, predictions_list)}
    logging.info(f'Predictions and shapes are: {[(p, predictions_dict[p].shape) for p in predictions_dict]}')

    for i, etm in enumerate(encoders):
        embed = encoders[etm].predict(test_data[etm.input_name()])
        if etm.output_name() in predictions_dict:
            plot_reconstruction(etm, test_data[etm.input_name()], predictions_dict[etm.output_name()], out_path, test_paths, samples)
        for dtm in decoders:
            reconstruction = decoders[dtm].predict(embed)
            logging.info(f'{dtm.name} has prediction shape: {reconstruction.shape} from embed shape: {embed.shape}')
            my_out_path = os.path.join(out_path, f'decoding_{dtm.name}_from_{etm.name}/')
            os.makedirs(os.path.dirname(my_out_path), exist_ok=True)
            if dtm.axes() > 1:
                plot_reconstruction(dtm, test_labels[dtm.output_name()], reconstruction, my_out_path, test_paths, samples)
            else:
                evaluate_predictions(dtm, reconstruction, test_labels[dtm.output_name()], {}, dtm.name, my_out_path, test_paths)
    return performance_metrics




def test_multimodal_multitask(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    out_path = os.path.join(args.output_folder, args.id + '/')
    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    return _predict_and_evaluate(
        model, data, labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, out_path, paths, args.embed_visualization, args.alpha,
    )


def test_multimodal_scalar_tasks(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    p = os.path.join(args.output_folder, args.id + '/')
    return _predict_scalars_and_evaluate_from_generator(
        model, generate_test, args.tensor_maps_in, args.tensor_maps_out,
        args.tensor_maps_protected, args.test_steps, args.hidden_layer, p, args.alpha,
    )


def compare_multimodal_multitask_models(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    models_inputs_outputs = get_model_inputs_outputs(args.model_files, args.tensor_maps_in, args.tensor_maps_out)
    input_data, output_data, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    common_outputs = _get_common_outputs(models_inputs_outputs, 'output')
    predictions = _get_predictions(args, models_inputs_outputs, input_data, common_outputs, 'input', 'output')
    _calculate_and_plot_prediction_stats(args, predictions, output_data, paths)


def compare_multimodal_scalar_task_models(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    models_io = get_model_inputs_outputs(args.model_files, args.tensor_maps_in, args.tensor_maps_out)
    outs = _get_common_outputs(models_io, "output")
    predictions, labels, paths = _scalar_predictions_from_generator(args, models_io, generate_test, args.test_steps, outs, "input", "output")
    _calculate_and_plot_prediction_stats(args, predictions, labels, paths)


def _make_tmap_nan_on_fail(tmap):
    """
    Builds a copy TensorMap with a tensor_from_file that returns nans on errors instead of raising an error
    """
    new_tmap = copy.deepcopy(tmap)
    new_tmap.validator = lambda _, x, hd5: x  # prevent failure caused by validator

    def _tff(tm, hd5, dependents=None):
        try:
            return tmap.tensor_from_file(tm, hd5, dependents)
        except (IndexError, KeyError, ValueError, OSError, RuntimeError):
            return np.full(shape=tm.shape, fill_value=np.nan)

    new_tmap.tensor_from_file = _tff
    return new_tmap


def inference_file_name(output_folder: str, id_: str) -> str:
    return os.path.join(output_folder, id_, 'inference_' + id_ + '.tsv')


def infer_multimodal_multitask(args):
    stats = Counter()
    tensor_paths_inferred = set()
    inference_tsv = inference_file_name(args.output_folder, args.id)
    tsv_style_is_genetics = 'genetics' in args.tsv_style

    model = make_multimodal_multitask_model(**args.__dict__)
    no_fail_tmaps_out = [_make_tmap_nan_on_fail(tmap) for tmap in args.tensor_maps_out]
    tensor_paths = _tensor_paths_from_sample_csv(args.tensors, args.sample_csv)
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    generate_test = TensorGenerator(
        1, args.tensor_maps_in, no_fail_tmaps_out, tensor_paths, num_workers=0,
        cache_size=0, keep_paths=True, mixup=args.mixup_alpha,
    )
    logging.info(f"Found {len(tensor_paths)} tensor paths.")
    generate_test.set_worker_paths(tensor_paths)
    with open(inference_tsv, mode='w') as inference_file:
        # TODO: csv.DictWriter is much nicer for this
        inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['sample_id']
        if tsv_style_is_genetics:
            header = ['FID', 'IID']
        for ot, otm in zip(args.output_tensors, args.tensor_maps_out):
            logging.info(f"Got ot  {ot} and otm {otm}  ot and otm {otm.name} ot  and otm {otm.channel_map} channel_map and otm {otm.interpretation}.")
            if len(otm.shape) == 1 and otm.is_continuous():
                header.extend([ot+'_prediction', ot+'_actual'])
            elif len(otm.shape) == 1 and otm.is_categorical():
                channel_columns = []
                for k in otm.channel_map:
                    channel_columns.append(ot + '_' + k + '_prediction')
                    channel_columns.append(ot + '_' + k + '_actual')
                header.extend(channel_columns)
        inference_writer.writerow(header)

        while True:
            batch = next(generate_test)
            input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
            if tensor_paths[0] in tensor_paths_inferred:
                next(generate_test)  # this prints end of epoch info
                logging.info(f"Inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                break
            prediction = model.predict(input_data)
            if len(no_fail_tmaps_out) == 1:
                prediction = [prediction]

            csv_row = [os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')]  # extract sample id
            if tsv_style_is_genetics:
                csv_row *= 2
            for y, tm in zip(prediction, no_fail_tmaps_out):
                if len(tm.shape) == 1 and tm.is_continuous():
                    csv_row.append(str(tm.rescale(y)[0][0]))  # first index into batch then index into the 1x1 structure
                    if ((tm.sentinel is not None and tm.sentinel == output_data[tm.output_name()][0][0])
                            or np.isnan(output_data[tm.output_name()][0][0])):
                        csv_row.append("NA")
                    else:
                        csv_row.append(str(tm.rescale(output_data[tm.output_name()])[0][0]))
                elif len(tm.shape) == 1 and tm.is_categorical():
                    for k, i in tm.channel_map.items():
                        try:
                            csv_row.append(str(y[0][tm.channel_map[k]]))
                            actual = output_data[tm.output_name()][0][i]
                            csv_row.append("NA" if np.isnan(actual) else str(actual))
                        except IndexError:
                            logging.debug(f'index error at {tm.name} item {i} key {k} with cm: {tm.channel_map} y is {y.shape} y is {y}')

            inference_writer.writerow(csv_row)
            tensor_paths_inferred.add(tensor_paths[0])
            stats['count'] += 1
            if stats['count'] % 250 == 0:
                logging.info(f"Wrote:{stats['count']} rows of inference.  Last tensor:{tensor_paths[0]}")


def _tensor_paths_from_sample_csv(tensors, sample_csv):
    sample_set = None
    if sample_csv is not None:
        with open(sample_csv, 'r') as csv_file:
            sample_ids = [row[0] for row in csv.reader(csv_file)]
            sample_set = set(sample_ids[1:])
    tensor_paths = [
        file for file in glob.glob(os.path.join(tensors, f"*{TENSOR_EXT}"))
        if sample_set is None or os.path.splitext(os.path.basename(file))[0] in sample_set
    ]
    return tensor_paths


def _hidden_file_name(output_folder: str, prefix_: str, id_: str, extension_: str) -> str:
    return os.path.join(output_folder, id_, prefix_ + id_ + extension_)


def infer_hidden_layer_multimodal_multitask(args):
    stats = Counter()
    args.num_workers = 0
    inference_tsv = _hidden_file_name(args.output_folder, 'hidden_inference_', args.id, '.tsv')
    tsv_style_is_genetics = 'genetics' in args.tsv_style
    tensor_paths = _tensor_paths_from_sample_csv(args.tensors, args.sample_csv)
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    generate_test = TensorGenerator(
        1, args.tensor_maps_in, args.tensor_maps_out, tensor_paths, num_workers=0,
        cache_size=args.cache_size, keep_paths=True, mixup=args.mixup_alpha,
    )
    generate_test.set_worker_paths(tensor_paths)
    full_model = make_multimodal_multitask_model(**args.__dict__)
    embed_model = make_hidden_layer_model(full_model, args.tensor_maps_in, args.hidden_layer)
    embed_model.save(_hidden_file_name(args.output_folder, f'{args.hidden_layer}_encoder_', args.id, '.h5'))
    dummy_input = {tm.input_name(): np.zeros((1,) + full_model.get_layer(tm.input_name()).input_shape[0][1:]) for tm in args.tensor_maps_in}
    dummy_out = embed_model.predict(dummy_input)
    latent_dimensions = int(np.prod(dummy_out.shape[1:]))
    logging.info(f'Dummy output shape is: {dummy_out.shape} latent dimensions: {latent_dimensions} Will write inferences to: {inference_tsv}')
    with open(inference_tsv, mode='w') as inference_file:
        inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['FID', 'IID'] if tsv_style_is_genetics else ['sample_id']
        header += [f'latent_{i}' for i in range(latent_dimensions)]
        inference_writer.writerow(header)

        while True:
            batch = next(generate_test)
            input_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_PATHS_INDEX]
            if tensor_paths[0] in stats:
                next(generate_test)  # this prints end of epoch info
                logging.info(f"Latent space inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                break

            sample_id = os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')
            prediction = embed_model.predict(input_data)
            prediction = np.reshape(prediction, (latent_dimensions,))
            csv_row = [sample_id, sample_id] if tsv_style_is_genetics else [sample_id]
            csv_row += [f'{prediction[i]}' for i in range(latent_dimensions)]
            inference_writer.writerow(csv_row)
            stats[tensor_paths[0]] += 1
            stats['count'] += 1
            if stats['count'] % 500 == 0:
                logging.info(f"Wrote:{stats['count']} rows of latent space inference.  Last tensor:{tensor_paths[0]}")


def train_shallow_model(args):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = make_shallow_model(args.tensor_maps_in, args.tensor_maps_out, args.learning_rate, args.model_file, args.model_layers)
    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size,
        args.epochs, args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels,
    )

    p = os.path.join(args.output_folder, args.id + '/')
    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    return _predict_and_evaluate(
        model, test_data, test_labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, p, test_paths, args.embed_visualization, args.alpha,
    )


def train_char_model(args):
    args.num_workers = 0
    logging.info(f'Number of workers forced to 0 for character emitting LSTM model.')
    base_model = make_multimodal_multitask_model(**args.__dict__)
    model, char_model = make_character_model_plus(
        args.tensor_maps_in, args.tensor_maps_out, args.learning_rate, base_model, args.language_layer,
        args.language_prefix, args.model_layers,
    )
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)

    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size,
        args.epochs, args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels,
    )
    batch = next(generate_test)
    input_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_PATHS_INDEX]
    sample_from_char_embed_model(args.tensor_maps_in, char_model, input_data, tensor_paths)

    out_path = os.path.join(args.output_folder, args.id + '/')
    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    return _predict_and_evaluate(
        model, data, labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, out_path, paths, args.embed_visualization, args.alpha,
    )


def train_siamese_model(args):
    base_model = make_multimodal_multitask_model(**args.__dict__)
    siamese_model = make_siamese_model(base_model, **args.__dict__)
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__, siamese=True)
    siamese_model = train_model_from_generators(
        siamese_model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size,
        args.epochs, args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels,
    )

    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    prediction = siamese_model.predict(data)
    return subplot_roc_per_class(
        prediction, labels['output_siamese'], {'random_siamese_verification_task': 0},
        args.protected_maps, args.id, os.path.join(args.output_folder, args.id + '/'),
    )


def train_paired_model(args):
    full_model, encoders, decoders = make_paired_autoencoder_model(**args.__dict__)
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    full_model = train_model_from_generators(
        full_model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size, args.epochs,
        args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels, args.tensor_maps_out,
        save_last_model=args.save_last_model
    )
    for tm in encoders:
        encoders[tm].save(f'{args.output_folder}{args.id}/encoder_{tm.name}.h5')
    for tm in decoders:
        decoders[tm].save(f'{args.output_folder}{args.id}/decoder_{tm.name}.h5')
    out_path = os.path.join(args.output_folder, args.id, 'reconstructions/')
    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    samples = min(args.test_steps * args.batch_size, 12)
    predictions_list = full_model.predict(test_data)
    predictions_dict = {name: pred for name, pred in zip(full_model.output_names, predictions_list)}
    logging.info(f'Predictions and shapes are: {[(p, predictions_dict[p].shape) for p in predictions_dict]}')
    performance_metrics = {}
    for tm in args.tensor_maps_out:
        if tm.axes() == 1:
            y = predictions_dict[tm.output_name()]
            y_truth = np.array(test_labels[tm.output_name()])
            metrics = evaluate_predictions(tm, y, y_truth, {}, tm.name, os.path.join(args.output_folder, args.id), test_paths)
            performance_metrics.update(metrics)
    for i, etm in enumerate(encoders):
        embed = encoders[etm].predict(test_data[etm.input_name()])
        plot_reconstruction(etm, test_data[etm.input_name()], predictions_dict[etm.output_name()], out_path, test_paths, samples)
        # fixed_point_predictions = plot_autoencoder_towards_attractor(full_model, test_data, etm, rows=samples, folder=out_path,
        #                                                              frames=min(5, args.attractor_iterations), steps=args.attractor_iterations)

        #plot_reconstruction(etm, test_data[etm.input_name()], fixed_point_predictions[etm.output_name()], out_path, test_paths, samples)
        # test_data[etm.input_name()] = np.random.random(test_data[etm.input_name()].shape)
        # plot_autoencoder_towards_attractor(full_model, test_data, etm, rows=samples, folder=out_path+'random/',
        #                                    frames=min(5, args.attractor_iterations), steps=args.attractor_iterations)
        for dtm in decoders:
            reconstruction = decoders[dtm].predict(embed)
            logging.info(f'{dtm.name} has prediction shape: {reconstruction.shape} from embed shape: {embed.shape}')
            my_out_path = os.path.join(out_path, f'decoding_{dtm.name}_from_{etm.name}/')
            os.makedirs(os.path.dirname(my_out_path), exist_ok=True)

            # fixed_point_predictions = plot_autoencoder_towards_attractor(full_model, test_data, dtm,  reconstruction=reconstruction, rows=samples, folder=my_out_path,
            #                                                              frames=min(5, args.attractor_iterations), steps=args.attractor_iterations)
            if dtm.axes() > 1:
                plot_reconstruction(dtm, test_data[dtm.input_name()], reconstruction, my_out_path, test_paths, samples)
            else:
                evaluate_predictions(dtm, reconstruction, test_labels[dtm.output_name()], {}, dtm.name, my_out_path, test_paths)
    return performance_metrics


def infer_encoders_block_multimodal_multitask(args):
    args.num_workers = 0
    tsv_style_is_genetics = 'genetics' in args.tsv_style
    _, encoders, _, _ = block_make_multimodal_multitask_model(**args.__dict__)
    latent_dimensions = args.dense_layers[-1]
    for encoder in encoders:
        stats = Counter()
        inference_tsv = _hidden_file_name(args.output_folder, encoder.name, args.id, '.tsv')
        logging.info(f'Will write encodings from {encoder.name} to: {inference_tsv}')
        tensor_paths = _tensor_paths_from_sample_csv(args.tensors, args.sample_csv)
        # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
        generate_test = TensorGenerator(
            1, [encoder], [], tensor_paths, num_workers=0,
            cache_size=args.cache_size, keep_paths=True, mixup=args.mixup_alpha,
        )
        generate_test.set_worker_paths(tensor_paths)
        with open(inference_tsv, mode='w') as inference_file:
            inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = ['FID', 'IID'] if tsv_style_is_genetics else ['sample_id']
            header += [f'latent_{i}' for i in range(latent_dimensions)]
            inference_writer.writerow(header)

            while True:
                batch = next(generate_test)
                input_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_PATHS_INDEX]
                if tensor_paths[0] in stats:
                    next(generate_test)  # this prints end of epoch info
                    logging.info(f"Latent space inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                    del stats
                    break

                sample_id = os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')
                prediction = encoders[encoder].predict(input_data)
                prediction = np.reshape(prediction, (latent_dimensions,))
                csv_row = [sample_id, sample_id] if tsv_style_is_genetics else [sample_id]
                csv_row += [f'{prediction[i]}' for i in range(latent_dimensions)]
                inference_writer.writerow(csv_row)
                stats[tensor_paths[0]] += 1
                stats['count'] += 1
                if stats['count'] % 500 == 0:
                    logging.info(f"Wrote:{stats['count']} rows of latent space inference.  Last tensor:{tensor_paths[0]}")


def inspect_paired_model(args):
    full_model, encoders, decoders, merger = block_make_multimodal_multitask_model(**args.__dict__)
    infer_hidden_tsv = _hidden_file_name(args.output_folder, 'hidden_inference_', args.id, '.tsv')
    latent_df = latent_space_dataframe(infer_hidden_tsv, args.app_csv)
    out_folder = os.path.join(args.output_folder, args.id, 'latent_transformations/')
    for tm in args.tensor_maps_protected:
        index2channel = {v: k for k, v in tm.channel_map.items()}
        thresh = 1 if tm.is_categorical() else tm.normalization.mean
        plot_hit_to_miss_transforms(latent_df, decoders,
                                    feature=index2channel[0],
                                    thresh=thresh,
                                    scalar=args.alpha,
                                    latent_dimension=args.dense_layers[0],
                                    prefix=out_folder,
                                    test_csv=args.test_csv)


def pca_on_hidden_inference(args):
    latent_cols = [f'latent_{i}' for i in range(args.dense_layers[0])]
    pca_on_tsv(args.app_csv, latent_cols, 'sample_id', args.dense_layers[1])


def plot_predictions(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    predictions = model.predict(data, batch_size=args.batch_size)
    if len(args.tensor_maps_out) == 1:
        predictions = [predictions]
    folder = os.path.join(args.output_folder, args.id, 'prediction_pngs/')
    predictions_to_pngs(predictions, args.tensor_maps_in, args.tensor_maps_out, data, labels, paths, folder)


def plot_while_training(args):
    generate_train, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    model = make_multimodal_multitask_model(**args.__dict__)

    plot_folder = os.path.join(args.output_folder, args.id, 'training_frames/')
    plot_while_learning(
        model, args.tensor_maps_in, args.tensor_maps_out, generate_train, test_data, test_labels, test_paths, args.epochs,
        args.batch_size, args.training_steps, plot_folder, args.write_pngs,
    )


def saliency_maps(args):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    in_tensor = data[args.tensor_maps_in[0].input_name()]
    for tm in args.tensor_maps_out:
        if len(tm.shape) > 1:
            continue
        for channel in tm.channel_map:
            gradients = saliency_map(in_tensor, model, tm.output_name(), tm.channel_map[channel])
            plot_saliency_maps(in_tensor, gradients, paths, os.path.join(args.output_folder, f'{args.id}/saliency_maps/{tm.name}_{channel}'))


def _predict_and_evaluate(
    model, test_data, test_labels, tensor_maps_in, tensor_maps_out, tensor_maps_protected,
    batch_size, hidden_layer, plot_path, test_paths, embed_visualization, alpha,
):
    layer_names = [layer.name for layer in model.layers]
    performance_metrics = {}
    scatters = []
    rocs = []

    y_predictions = model.predict(test_data, batch_size=batch_size)
    protected_data = {tm: test_labels[tm.output_name()] for tm in tensor_maps_protected}
    for y, tm in zip(y_predictions, tensor_maps_out):
        if tm.output_name() not in layer_names:
            continue
        if not isinstance(y_predictions, list):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
            y = y_predictions
        y_truth = np.array(test_labels[tm.output_name()])
        performance_metrics.update(evaluate_predictions(tm, y, y_truth, protected_data, tm.name, plot_path, test_paths, rocs=rocs, scatters=scatters))
        if tm.is_language():
            sample_from_language_model(tensor_maps_in[0], tm, model, test_data, max_samples=16)

    if len(rocs) > 1:
        subplot_rocs(rocs, plot_path)
    if len(scatters) > 1:
        subplot_scatters(scatters, plot_path)

    test_labels_1d = {tm: np.array(test_labels[tm.output_name()]) for tm in tensor_maps_out if tm.output_name() in test_labels}
    if embed_visualization == "tsne":
        _tsne_wrapper(model, hidden_layer, alpha, plot_path, test_paths, test_labels_1d, test_data=test_data, tensor_maps_in=tensor_maps_in, batch_size=batch_size)

    return performance_metrics


def _predict_scalars_and_evaluate_from_generator(
    model, generate_test, tensor_maps_in, tensor_maps_out, tensor_maps_protected,
    steps, hidden_layer, plot_path, alpha,
):
    layer_names = [layer.name for layer in model.layers]
    model_predictions = [tm.output_name() for tm in tensor_maps_out if tm.output_name() in layer_names]
    scalar_predictions = {tm.output_name(): [] for tm in tensor_maps_out if len(tm.shape) == 1 and tm.output_name() in layer_names}
    test_labels = {tm.output_name(): [] for tm in tensor_maps_out if len(tm.shape) == 1}
    protected_data = {tm: [] for tm in tensor_maps_protected}

    logging.info(f"Scalar predictions {model_predictions} names: {scalar_predictions.keys()} test labels: {test_labels.keys()}")
    embeddings = []
    test_paths = []
    for i in range(steps):
        batch = next(generate_test)
        input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
        y_predictions = model.predict(input_data)
        test_paths.extend(tensor_paths)
        if hidden_layer in layer_names:
            x_embed = embed_model_predict(model, tensor_maps_in, hidden_layer, input_data, 2)
            embeddings.extend(np.copy(np.reshape(x_embed, (x_embed.shape[0], np.prod(x_embed.shape[1:])))))

        for tm_output_name in test_labels:
            test_labels[tm_output_name].extend(np.copy(output_data[tm_output_name]))
        for tm in tensor_maps_protected:
            protected_data[tm].extend(np.copy(output_data[tm.output_name()]))

        for y, tm_output_name in zip(y_predictions, model_predictions):
            if not isinstance(y_predictions, list):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
                y = y_predictions
            if tm_output_name in scalar_predictions:
                scalar_predictions[tm_output_name].extend(np.copy(y))

    performance_metrics = {}
    scatters = []
    rocs = []
    for tm in tensor_maps_protected:
        protected_data[tm] = np.array(protected_data[tm])

    for tm in tensor_maps_out:
        if tm.output_name() in scalar_predictions:
            y_predict = np.array(scalar_predictions[tm.output_name()])
            y_truth = np.array(test_labels[tm.output_name()])
            metrics = evaluate_predictions(tm, y_predict, y_truth, protected_data, tm.name, plot_path, test_paths, rocs=rocs, scatters=scatters)
            performance_metrics.update(metrics)

    if len(rocs) > 1:
        subplot_rocs(rocs, plot_path)
    if len(scatters) > 1:
        subplot_scatters(scatters, plot_path)
    if len(embeddings) > 0:
        test_labels_1d = {tm: np.array(test_labels[tm.output_name()]) for tm in tensor_maps_out if tm.output_name() in test_labels}
        _tsne_wrapper(model, hidden_layer, alpha, plot_path, test_paths, test_labels_1d, embeddings=embeddings)

    return performance_metrics


def _get_common_outputs(models_inputs_outputs, output_prefix):
    """Returns a set of outputs common to all the models so we can compare the models according to those outputs only"""
    all_outputs = []
    for (_, ios) in models_inputs_outputs.items():
        outputs = {k: v for (k, v) in ios.items() if k == output_prefix}
        for (_, output) in outputs.items():
            all_outputs.append(set(output))
    return reduce(set.intersection, all_outputs)


def _get_predictions(args, models_inputs_outputs, input_data, outputs, input_prefix, output_prefix):
    """Makes multi-modal predictions for a given number of models.

    Returns:
        dict: The nested dictionary of predicted values.

            {
                'tensor_map_1':
                    {
                        'model_1': [[value1, value2], [...]],
                        'model_2': [[value3, value4], [...]]
                    },
                'tensor_map_2':
                    {
                        'model_2': [[value5, value6], [...]],
                        'model_3': [[value7, value8], [...]]
                    }
            }
    """
    predictions = defaultdict(dict)
    for model_file in models_inputs_outputs.keys():
        args.tensor_maps_out = models_inputs_outputs[model_file][output_prefix]
        args.tensor_maps_in = models_inputs_outputs[model_file][input_prefix]
        args.model_file = model_file
        model = make_multimodal_multitask_model(**args.__dict__)
        model_name = os.path.basename(model_file).replace(MODEL_EXT, '_')

        # We can feed 'model.predict()' the entire input data because it knows what subset to use
        y_pred = model.predict(input_data, batch_size=args.batch_size)

        for i, tm in enumerate(args.tensor_maps_out):
            if tm in outputs:
                if len(args.tensor_maps_out) == 1:
                    predictions[tm][model_name] = y_pred
                else:
                    predictions[tm][model_name] = y_pred[i]

    return predictions


def _scalar_predictions_from_generator(args, models_inputs_outputs, generator, steps, outputs, input_prefix, output_prefix):
    """Makes multi-modal scalar predictions for a given number of models.

    Returns:
        dict: The nested dictionary of predicted values.

            {
                'tensor_map_1':
                    {
                        'model_1': [[value1, value2], [...]],
                        'model_2': [[value3, value4], [...]]
                    },
                'tensor_map_2':
                    {
                        'model_2': [[value5, value6], [...]],
                        'model_3': [[value7, value8], [...]]
                    }
            }
    """
    models = {}
    test_paths = []
    scalar_predictions = {}
    test_labels = {tm.output_name(): [] for tm in args.tensor_maps_out if len(tm.shape) == 1}

    for model_file in models_inputs_outputs:
        args.model_file = model_file
        args.tensor_maps_in = models_inputs_outputs[model_file][input_prefix]
        args.tensor_maps_out = models_inputs_outputs[model_file][output_prefix]
        model = make_multimodal_multitask_model(**args.__dict__)
        model_name = os.path.basename(model_file).replace(MODEL_EXT, '')
        models[model_name] = model
        scalar_predictions[model_name] = [tm for tm in models_inputs_outputs[model_file][output_prefix] if len(tm.shape) == 1]

    predictions = defaultdict(dict)
    for j in range(steps):
        batch = next(generator)
        input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
        test_paths.extend(tensor_paths)
        for tl in test_labels:
            test_labels[tl].extend(np.copy(output_data[tl]))

        for model_name, model_file in zip(models, models_inputs_outputs):
            # We can feed 'model.predict()' the entire input data because it knows what subset to use
            y_predictions = models[model_name].predict(input_data)

            for y, tm in zip(y_predictions, models_inputs_outputs[model_file][output_prefix]):
                if not isinstance(y_predictions, list):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
                    y = y_predictions
                if j == 0 and tm in scalar_predictions[model_name]:
                    predictions[tm][model_name] = []
                if tm in scalar_predictions[model_name]:
                    predictions[tm][model_name].extend(np.copy(y))

    for tm in predictions:
        logging.info(f"{tm.output_name()} labels: {len(test_labels[tm.output_name()])}")
        test_labels[tm.output_name()] = np.array(test_labels[tm.output_name()])
        for m in predictions[tm]:
            logging.info(f"{tm.output_name()} model: {m} prediction length:{len(predictions[tm][m])}")
            predictions[tm][m] = np.array(predictions[tm][m])

    return predictions, test_labels, test_paths


def _calculate_and_plot_prediction_stats(args, predictions, outputs, paths):
    rocs = []
    scatters = []
    for tm in args.tensor_maps_out:
        if tm not in predictions:
            continue
        plot_title = tm.name+'_'+args.id
        plot_folder = os.path.join(args.output_folder, args.id)
        if tm.is_categorical() and tm.axes() == 1:
            for m in predictions[tm]:
                logging.info(f"{tm.name} channel map {tm.channel_map}\nsum truth = {np.sum(outputs[tm.output_name()], axis=0)}\nsum preds = {np.sum(predictions[tm][m], axis=0)}")
            plot_rocs(predictions[tm], outputs[tm.output_name()], tm.channel_map, plot_title, plot_folder)
            rocs.append((predictions[tm], outputs[tm.output_name()], tm.channel_map))
        elif tm.is_categorical() and tm.axes() == 4:
            for p in predictions[tm]:
                y = predictions[tm][p]
                melt_shape = (y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3], y.shape[4])
                predictions[tm][p] = y.reshape(melt_shape)

            y_truth = outputs[tm.output_name()].reshape(melt_shape)
            plot_rocs(predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder)
            plot_precision_recalls(predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder)
            roc_aucs = get_roc_aucs(predictions[tm], y_truth, tm.channel_map)
            precision_recall_aucs = get_precision_recall_aucs(predictions[tm], y_truth, tm.channel_map)
            aucs = {"ROC": roc_aucs, "Precision-Recall": precision_recall_aucs}
            log_aucs(**aucs)
        elif tm.is_categorical() and tm.axes() == 3:
            for p in predictions[tm]:
                y = predictions[tm][p]
                melt_shape = (y.shape[0]*y.shape[1]*y.shape[2], y.shape[3])
                predictions[tm][p] = y.reshape(melt_shape)

            y_truth = outputs[tm.output_name()].reshape(melt_shape)
            plot_rocs(predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder)
            plot_precision_recalls(predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder)
            roc_aucs = get_roc_aucs(predictions[tm], y_truth, tm.channel_map)
            precision_recall_aucs = get_precision_recall_aucs(predictions[tm], y_truth, tm.channel_map)
            aucs = {"ROC": roc_aucs, "Precision-Recall": precision_recall_aucs}
            log_aucs(**aucs)
        elif tm.is_continuous() and tm.axes() == 1:
            scaled_predictions = {k: tm.rescale(predictions[tm][k]) for k in predictions[tm]}
            plot_scatters(scaled_predictions, tm.rescale(outputs[tm.output_name()]), plot_title, plot_folder) #, paths)
            scatters.append((scaled_predictions, tm.rescale(outputs[tm.output_name()]), plot_title, None))
            coefs = get_pearson_coefficients(scaled_predictions, tm.rescale(outputs[tm.output_name()]))
            log_pearson_coefficients(coefs, tm.name)
        elif tm.is_time_to_event():
            new_predictions = {}
            for m in predictions[tm]:
                c_index = concordance_index_censored(outputs[tm.output_name()][:, 0] == 1.0, outputs[tm.output_name()][:, 1], predictions[tm][m][:, 0])
                concordance_return_values = ['C-Index', 'Concordant Pairs', 'Discordant Pairs', 'Tied Predicted Risk', 'Tied Event Time']
                logging.info(f"Model: {m} {[f'{label}: {value}' for label, value in zip(concordance_return_values, c_index)]}")
                new_predictions[f'{m}_C_Index_{c_index[0]:0.3f}'] = predictions[tm][m]
            plot_rocs(new_predictions, outputs[tm.output_name()][:, 0, np.newaxis], {f'_vs_ROC': 0}, plot_title, plot_folder)
            rocs.append((new_predictions, outputs[tm.output_name()][:, 0, np.newaxis], {f'_vs_ROC': 0}))
            plot_prediction_calibrations(new_predictions, outputs[tm.output_name()][:, 0, np.newaxis], {f'_vs_ROC': 0}, plot_title, plot_folder)
        else:
            scaled_predictions = {k: tm.rescale(predictions[tm][k]) for k in predictions[tm]}
            plot_scatters(scaled_predictions, tm.rescale(outputs[tm.output_name()]), plot_title, plot_folder)
            coefs = get_pearson_coefficients(scaled_predictions, tm.rescale(outputs[tm.output_name()]))
            log_pearson_coefficients(coefs, tm.name)

    if len(rocs) > 1:
        subplot_comparison_rocs(rocs, plot_folder)
    if len(scatters) > 1:
        subplot_comparison_scatters(scatters, plot_folder)


def _tsne_wrapper(model, hidden_layer_name, alpha, plot_path, test_paths, test_labels, test_data=None, tensor_maps_in=None, batch_size=16, embeddings=None):
    """Plot 2D t-SNE of a model's hidden layer colored by many different co-variates.

    Callers must provide either model's embeddings or test_data on which embeddings will be inferred

    :param model: Keras model
    :param hidden_layer_name: String name of the hidden layer whose embeddings will be visualized
    :param alpha: Transparency of each data point
    :param plot_path: Image file name and path for the t_SNE plot
    :param test_paths: Paths for hd5 file containing each sample
    :param test_labels: Dictionary mapping TensorMaps to numpy arrays of labels (co-variates) to color code the t-SNE plots with
    :param test_data: Input data for the model necessary if embeddings is None
    :param tensor_maps_in: Input TensorMaps of the model necessary if embeddings is None
    :param batch_size: Batch size necessary if embeddings is None
    :param embeddings: (optional) Model's embeddings
    :return: None
    """
    if hidden_layer_name not in [layer.name for layer in model.layers]:
        logging.warning(f"Can't compute t-SNE, layer:{hidden_layer_name} not in provided model.")
        return

    if embeddings is None:
        embeddings = embed_model_predict(model, tensor_maps_in, hidden_layer_name, test_data, batch_size)

    gene_labels = []
    label_dict, categorical_labels, continuous_labels = test_labels_to_label_map(test_labels, len(test_paths))
    if len(categorical_labels) > 0 or len(continuous_labels) > 0 or len(gene_labels) > 0:
        plot_tsne(embeddings, categorical_labels, continuous_labels, gene_labels, label_dict, plot_path, alpha)


if __name__ == '__main__':
    arguments = parse_args()
    run(arguments)  # back to the top
