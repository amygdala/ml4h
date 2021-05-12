ECHO=echo
MODEL_FILES=
ID=liver_fat_student
INPUT_TENSORS=lms_ideal_optimised_low_flip_6dyn
OUTPUT_TENSORS=proton_fat
FOLDS=10
TEST_CSV=/home/sam/testing_sample_ids_ideal_protocol.txt
TENSORS=/mnt/disks/liver-tensors-40k/2019-11-13/
for i in $(seq 1 $FOLDS)
do
    $ECHO "Cross validation fold: ${i}"
    $ECHO  ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode explore --tensors ${TENSORS}  \
        --input_tensors $INPUT_TENSORS $OUTPUT_TENSORS  --output_tensors --tensormap_prefix ml4h.tensormap.ukb.mri \
        --batch_size 4 \
        --random_seed $i \
        --id ${ID}_fold_${i} --output_folder /home/sam/trained_models/ \
        --test_csv $TEST_CSV
    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode train --tensors ${TENSORS}  \
         --input_tensors $INPUT_TENSORS --output_tensors $OUTPUT_TENSORS --tensormap_prefix ml4h.tensormap.ukb.mri \
         --dense_blocks 32 24 16 --dense_layers 16 64 \
         --training_steps 96 --validation_steps 32 --test_steps 32 --batch_size 8 --epochs 64 --patience 12 \
         --output_folder /home/sam/trained_models/ --test_csv $TEST_CSV \
         --id ${ID}_fold_${i} --random_seed $i
    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode infer --tensors ${TENSORS}  \
         --input_tensors $INPUT_TENSORS --output_tensors  $OUTPUT_TENSORS --tensormap_prefix ml4h.tensormap.ukb.mri \
         --output_folder /home/sam/ml/trained_models/ --sample_csv $TEST_CSV \
         --model_file /home/sam/trained_models/${ID}_fold_${i}/${ID}_fold_${i}.h5 \
         --random_seed $i --id ${ID}_fold_${i}
done
# $ECHO   ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode compare_scalar --tensors /mnt/disks/annotated-cardiac-tensors-45k-2021-03-25/2020-09-21/  \
#     --input_tensors ecg.ecg_rest_median_raw_10  --output_tensors mri.LVEDV mri.LVEF mri.LVESV mri.LVM mri.LVSV mri.RVEDV mri.RVEF mri.RVESV mri.RVSV \
#     --batch_size 4 --epochs 1 --training_steps 1 --validation_steps 1 --test_steps 120 \
#     --num_workers 1 --patience 12 --tensormap_prefix ml4h.tensormap.ukb \
#     --id ecg_subsets_to_lv_rv --output_folder /home/sam/trained_models/ \
#      --test_csv /home/sam/csvs/multimodal_test_set_488.csv \
#      --model_files $MODEL_FILES