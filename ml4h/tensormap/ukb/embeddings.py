from ml4h import TensorMap
from ml4h.tensormap.general import build_vector_tensor_from_file


latent_file = '/home/sam/trained_models/cine_segmented_lax_4ch_diastole_autoencoder_64d/hidden_inference_cine_segmented_lax_4ch_diastole_autoencoder_64d.tsv'
embed_cine_segmented_lax_4ch_diastole = TensorMap('embed_cine_segmented_lax_4ch_diastole', shape=(64,), channel_map={f'latent_{i}': i for i in range(64)},
                                                  tensor_from_file=build_vector_tensor_from_file(latent_file),
                                                  )
