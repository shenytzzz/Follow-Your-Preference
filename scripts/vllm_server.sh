VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve CodeGoat24/UnifiedReward-qwen-7b \
    --host 127.0.0.1 \
    --trust-remote-code \
    --served-model-name UnifiedReward \
    --gpu-memory-utilization 0.4 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt image=2 \
    --port 8080