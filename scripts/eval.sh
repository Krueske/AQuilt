model_path=""
domain_task=""
CUDA_VISIBLE_DEVICES=0 python -u ../tests/$domain_task\_test.py $model_path > ./logs/$domain_task\_eval.log 2>&1 &
