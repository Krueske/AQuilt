judged_data_path=""
train_data_path=""
CUDA_VISIBLE_DEVICES=0 python -u process_data_judged.py $judged_data_path $train_data_path > ./logs/process_data.log 2>&1 &