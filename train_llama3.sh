NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train LLaMA-Factory/llama3_lora_sft1.yaml

wait

modelPath=Meta-Llama3-8B-Instruct

adapterModelPath=lora_path

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli export \
  --model_name_or_path $modelPath \
  --adapter_name_or_path $adapterModelPath \
  --template empty \
  --finetuning_type lora \
  --export_dir output_model_path \
  --export_size 2 \
  --export_legacy_format False