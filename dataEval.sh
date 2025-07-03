model_path=""
eval_lora_path=""
data_path=""
output_datapath=""
CUDA_VISIBLE_DEVICES=0 python -u data_eval.py $model_path $eval_lora_path $data_path $output_datapath > ./logs/data_eval.log 2>&1 &
