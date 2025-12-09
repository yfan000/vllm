#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASETS=("longcot_1000_priority_original.jsonl" "longwriter_1000_priority_original.jsonl")
OUTLEN=12800
NUM_PROMPTS=1000

# Sweep spaces ↓ Modify freely
REQUEST_RATES=("inf" "5" "10" "15" "20")
MAX_NUM_SEQS=("128" "256")

# mkdir -p benchmark_results

for ds in "${DATASETS[@]}"; do
    base=$(basename "$ds" .jsonl)

    for mns in "${MAX_NUM_SEQS[@]}"; do
        for rate in "${REQUEST_RATES[@]}"; do

            OUT="bench_${base}_mns${mns}_rate${rate}.json"

            echo "==============================="
            echo "Start vLLM serve → max-num-seqs=${mns}"
            echo "Benchmark dataset=${base}, request-rate=${rate}"
            echo "==============================="

            # ---- Start server ----
            nohup vllm serve $MODEL \
                --scheduling-policy priority \
                --max-num-seqs "$mns" \
                > "server_${base}_mns${mns}_rate${rate}.log" 2>&1 &

            SERVER_PID=$!
            echo "Server running (PID=$SERVER_PID), warming up..."
            sleep 200    # Increase if model loading slow

            # ---- Run benchmark ----
            vllm bench serve \
                --backend vllm \
                --model "$MODEL" \
                --dataset-name custom \
                --dataset-path "$ds" \
                --num-prompts "$NUM_PROMPTS" \
                --request-rate "$rate" \
                --save-result \
                --result-dir benchmark_results \
                --result-filename "$OUT" \
                --save-detailed \
                --custom-output-len "$OUTLEN"

            # ---- Terminate server ----
            echo "Stopping server PID=$SERVER_PID"
            kill "$SERVER_PID"
            sleep 200   # let CUDA/GPU release before next run

        done
    done
done

echo "All benchmarks finished"
