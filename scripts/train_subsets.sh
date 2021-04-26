ECHO=echo
MODEL_FILES=
array=( "multimodal_train_set_1024.csv"  "multimodal_train_set_16.csv"  "multimodal_train_set_2048.csv"  "multimodal_train_set_256.csv"  "multimodal_train_set_3350.csv"  "multimodal_train_set_3488.csv"  "multimodal_train_set_64.csv" )
for i in "${array[@]}"
do
	$ECHO $i
    $ECHO   ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode train_block --tensors /mnt/disks/annotated-cardiac-tensors-45k-2021-03-25/2020-09-21/  \
    --input_tensors ecg.ecg_rest_median_raw_10  --output_tensors mri.LVEDV mri.LVEF mri.LVESV mri.LVM mri.LVSV mri.RVEDV mri.RVEF mri.RVESV mri.RVSV \
    --encoder_blocks conv_encode --merge_blocks --decoder_blocks dense_decode --activation mish --conv_layers 32 --dense_blocks 32 32 32 --dense_layers 32 --block_size 3 \
    --batch_size 1 --epochs 1 --training_steps 1 --validation_steps 1 --test_steps 40 \
    --num_workers 4 --patience 12 --tensormap_prefix ml4h.tensormap.ukb \
    --id ${i}_to_lv_rv --output_folder /home/sam/trained_models/ \
     --inspect_model --sample_csv /home/sam/csvs/$i \
     --learning_rate 0.00005
     MODEL_FILES="${MODEL_FILES} /home/sam/trained_models/${i}_to_lv_rv/${i}_to_lv_rv.h5"
done
$ECHO $MODEL_FILES