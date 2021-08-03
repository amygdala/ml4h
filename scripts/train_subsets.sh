ECHO=
MODEL_FILES=
TENSORS="/mnt/disks/annotated-cardiac-tensors-45k-2021-03-25/2020-09-21/"
TENSOR_MAPS="ecg.ecg_rest_median_raw_10 mri.lax_4ch_heart_center "
array=( "multimodal_split_64.csv" "multimodal_split_128.csv" "multimodal_split_256.csv" "multimodal_split_512.csv" "multimodal_split_1024.csv"  "multimodal_split_2048.csv")
for i in "${array[@]}"
do
#    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode train_block \
#    --tensors "$TENSORS" --input_tensors "$TENSOR_MAPS" --output_tensors "$TENSOR_MAPS" \
#    --encoder_blocks conv_encode \
#    --merge_blocks \
#    --decoder_blocks conv_decode \
#    --pairs "$TENSOR_MAPS" --pair_loss contrastive --pair_loss_weight 0.1 --pair_merge dropout \
#    --batch_size 4 --epochs 316 --training_steps 128 --validation_steps 32 --test_steps 1 \
#    --num_workers 4 --patience 4 --tensormap_prefix ml4h.tensormap.ukb \
#    --id "drop_fuse_early_stop_v2_${i%.*}" --output_folder /home/sam/trained_models/ \
#    --inspect_model --activation mish --dense_layers 256 \
#    --train_csv "/home/sam/csvs/${i}" \
#    --valid_csv /home/sam/csvs/multimodal_validation.csv \
#    --test_csv /home/sam/csvs/multimodal_test.csv


    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode infer_encoders \
    --tensors "$TENSORS" --input_tensors "$TENSOR_MAPS" --output_tensors "$TENSOR_MAPS" \
    --model_file "/home/sam/trained_models/drop_fuse_early_stop_v2_${i%.*}/drop_fuse_early_stop_v2_${i%.*}.h5" \
    --id "drop_fuse_early_stop_v2_all_infer_${i%.*}" --output_folder /home/sam/trained_models/ \
    --sample_csv /home/sam/csvs/sample_id_returned_lv_mass.csv \
    --tensormap_prefix ml4h.tensormap.ukb \
    --dense_layers 256
done
