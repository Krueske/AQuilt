domian_task=""
model_path=""
unlabeled_data_path=""
output_data_path=""
CUDA_VISIBLE_DEVICES=0 python -u ./dataGen/$domain_task\_sft_dataGen.py $model_path $unlabeled_data_path $outoput_data_path > ./logs/data_gen.log 2>&1 &
