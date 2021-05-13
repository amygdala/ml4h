ECHO=
MODEL_FILES=
ID=liver_fat_student
INPUT_TENSORS=lms_ideal_optimised_low_flip_6dyn
OUTPUT_TENSORS=liver_fat_echo_predicted
OUTPUT_FOLDER=/home/sam/trained_models/
FOLDS=10
TEST_CSV=/home/sam/testing_sample_ids_ideal_protocol.txt
TENSORS=/mnt/disks/liver-tensors-40k/2019-11-13/
for i in $(seq 1 $FOLDS)
do
    $ECHO echo "Cross validation fold: ${i}"
    $ECHO  ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode explore --tensors ${TENSORS}  \
        --input_tensors $INPUT_TENSORS $OUTPUT_TENSORS  --output_tensors --tensormap_prefix ml4h.tensormap.ukb.mri \
        --batch_size 4 \
        --random_seed $i \
        --id ${ID}_fold_${i} --output_folder $OUTPUT_FOLDER \
        --test_csv $TEST_CSV
    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode train --tensors ${TENSORS}  \
         --input_tensors $INPUT_TENSORS --output_tensors $OUTPUT_TENSORS --tensormap_prefix ml4h.tensormap.ukb.mri \
         --dense_blocks 32 24 16 --dense_layers 16 64 \
         --training_steps 96 --validation_steps 32 --test_steps 16 --batch_size 2 --epochs 84 --patience 16 \
         --output_folder $OUTPUT_FOLDER --test_csv $TEST_CSV \
         --id ${ID}_fold_${i} --random_seed $i
    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode infer --tensors ${TENSORS}  \
         --input_tensors $INPUT_TENSORS --output_tensors  $OUTPUT_TENSORS --tensormap_prefix ml4h.tensormap.ukb.mri \
         --output_folder $OUTPUT_FOLDER --sample_csv $TEST_CSV \
         --model_file /home/sam/trained_models/${ID}_fold_${i}/${ID}_fold_${i}.h5 \
         --random_seed $i --id ${ID}_fold_${i}
done
