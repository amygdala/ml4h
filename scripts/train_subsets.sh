ECHO=
MODEL_FILES=
TENSORS="/mnt/disks/annotated-cardiac-tensors-45k-2021-03-25/2020-09-21/"
TENSOR_MAPS="ecg.ecg_rest_median_raw_10 mri.lax_4ch_heart_center "
array=( "multimodal_split_64.csv" "multimodal_split_128.csv" "multimodal_split_256.csv" "multimodal_split_512.csv" "multimodal_split_1024.csv"  "multimodal_split_2048.csv")
for i in "${array[@]}"
do
    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode train_block \
    --tensors "$TENSORS" --input_tensors "$TENSOR_MAPS" --output_tensors "$TENSOR_MAPS" \
    --encoder_blocks /home/sam/trained_models/hypertuned_48m_16e_ecg_median_raw_10_autoencoder_256d/encoder_ecg_rest_median_raw_10.h5 \
                     /home/sam/trained_models/hypertuned_32m_8e_lax_4ch_heart_center_autoencoder_256d/encoder_lax_4ch_heart_center.h5 \
    --merge_blocks \
    --decoder_blocks /home/sam/trained_models/hypertuned_48m_16e_ecg_median_raw_10_autoencoder_256d/decoder_ecg_rest_median_raw_10.h5 \
                     /home/sam/trained_models/hypertuned_32m_8e_lax_4ch_heart_center_autoencoder_256d/decoder_lax_4ch_heart_center.h5 \
    --pairs "$TENSOR_MAPS" --pair_loss contrastive --pair_loss_weight 0.1 --pair_merge dropout \
    --batch_size 4 --epochs 316 --training_steps 128 --validation_steps 32 --test_steps 1 \
    --num_workers 4 --patience 108 --tensormap_prefix ml4h.tensormap.ukb \
    --id "drop_fuse_${i%.*}" --output_folder /home/sam/trained_models/ \
    --inspect_model --save_last_model \
    --train_csv "/home/sam/csvs/${i}" \
    --valid_csv /home/sam/csvs/multimodal_validation.csv \
    --test_csv /home/sam/csvs/multimodal_test.csv \
    --learning_rate 0.00005

    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode infer_encoders \
    --tensors "$TENSORS" --input_tensors "$TENSOR_MAPS" --output_tensors "$TENSOR_MAPS" \
    --model_file "/home/sam/trained_models/drop_fuse_${i%.*}/drop_fuse_${i%.*}.h5" \
    --id "drop_fuse_${i%.*}" --output_folder /home/sam/trained_models/ \
    --sample_csv /home/sam/csvs/multimodal_test.csv \
    --tensormap_prefix ml4h.tensormap.ukb \
    --dense_layers 256
done
