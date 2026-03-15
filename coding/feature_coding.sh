#!/usr/bin/env bash
set -e          # 只要脚本发生错误就停止运行
set -u          # 如果遇到不存在的变量就报错并停止执行
# set -x          # 运行指令结果的时候，输出对应的指令
set -o pipefail # 确保只要一个子命令失败，整个管道命令就失败

# export CUDA_VISIBLE_DEVICES=0

# Codec configurations - both will be evaluated for each handler
declare -A CODEC_NAMES CODEC_WEIGHTS_DIRS CODEC_ARCHS
CODEC_NAMES[0]="hyperprior"
CODEC_WEIGHTS_DIRS[0]="codec_weights/hyperprior_hybrid"
CODEC_ARCHS[0]="hyperprior-featurecoding"

CODEC_NAMES[1]="elic"
CODEC_WEIGHTS_DIRS[1]="codec_weights/elic_hybrid"
CODEC_ARCHS[1]="elic-featurecoding"

CODEC_COUNT=2

# Handler to datasets mapping using associative arrays
# Format: "name1:path1|name2:path2|..."
declare -A HANDLER_DATASETS
HANDLER_DATASETS["dinov3-total"]="cls_in1kval:../datasets/FeatureCoding-DINOv3/imagenet1k-val|cls_in1ka:../datasets/FeatureCoding-DINOv3/imagenet1k-a|cls_in1kr:../datasets/FeatureCoding-DINOv3/imagenet1k-r|nyudv2_test:../datasets/FeatureCoding-DINOv3/DINOv3Dep-NYUDv2Test-4lvl-100Features|ade20k_val:../datasets/FeatureCoding-DINOv3/DINOv3Seg-ADE20KVal-4lvl-100Features"
HANDLER_DATASETS["qwen"]="fc_arc_challenge:../datasets/Qwen3-500features-L5wCache/Qwen3-500features-L5wCache/qwen/qwen3-8b/fc_arc_challenge|fc_gsm8k:../datasets/Qwen3-500features-L5wCache/Qwen3-500features-L5wCache/qwen/qwen3-8b/fc_gsm8k|fc_hellaswag:../datasets/Qwen3-500features-L5wCache/Qwen3-500features-L5wCache/qwen/qwen3-8b/fc_hellaswag|fc_truthfulqa_mc1:../datasets/Qwen3-500features-L5wCache/Qwen3-500features-L5wCache/qwen/qwen3-8b/fc_truthfulqa_mc1|fc_winogrande:../datasets/Qwen3-500features-L5wCache/Qwen3-500features-L5wCache/qwen/qwen3-8b/fc_winogrande"
HANDLER_DATASETS["falconmamba"]="fc_arc_challenge:../datasets/FalconMamba-500features-L5wCache/FalconMamba-500features-L5wCache/tiiuae/falcon-mamba-7b-instruct/fc_arc_challenge|fc_gsm8k:../datasets/FalconMamba-500features-L5wCache/FalconMamba-500features-L5wCache/tiiuae/falcon-mamba-7b-instruct/fc_gsm8k|fc_hellaswag:../datasets/FalconMamba-500features-L5wCache/FalconMamba-500features-L5wCache/tiiuae/falcon-mamba-7b-instruct/fc_hellaswag|fc_truthfulqa_mc1:../datasets/FalconMamba-500features-L5wCache/FalconMamba-500features-L5wCache/tiiuae/falcon-mamba-7b-instruct/fc_truthfulqa_mc1|fc_winogrande:../datasets/FalconMamba-500features-L5wCache/FalconMamba-500features-L5wCache/tiiuae/falcon-mamba-7b-instruct/fc_winogrande"
HANDLER_DATASETS["kimiaudio"]="librispeech-test-clean:../datasets/KimiAudio-7B-Instruct-500features-L5wCache//KimiAudio-7B-Instruct-500features-L5wCache/librispeech-test-clean|librispeech-test-other:../datasets/KimiAudio-7B-Instruct-500features-L5wCache//KimiAudio-7B-Instruct-500features-L5wCache/librispeech-test-other|advbench:../datasets/KimiAudio-7B-Instruct-500features-L5wCache//KimiAudio-7B-Instruct-500features-L5wCache/advbench|openbookqa:../datasets/KimiAudio-7B-Instruct-500features-L5wCache//KimiAudio-7B-Instruct-500features-L5wCache/openbookqa|sd-qa:../datasets/KimiAudio-7B-Instruct-500features-L5wCache//KimiAudio-7B-Instruct-500features-L5wCache/sd-qa"
HANDLER_DATASETS["sd35"]="sd35cond:../datasets/FeatureCoding-StableDiffusion3.5/sd3.5-l-controlnet-canny-cond-tti"

# Handlers to evaluate (comment out to skip)
# "qwen" "falconmamba" "kimiaudio" "sd35" "dinov3-total"
HANDLERS=("qwen" "falconmamba" "kimiaudio" "sd35" "dinov3-total")
# Lambda values to evaluate (leave empty to evaluate all)
LAMBDA_FILTER=(0.001 0.004 0.007 0.01 0.02)

# Skip combinations
# Format: "arch:handler:lambda:dataset" (use * as wildcard)
# Examples:
#   "hyperprior-featurecoding:qwen:0.001:fc_gsm8k"       # 跳过指定组合
#   "hyperprior-featurecoding:dinov3-total:*:cls_in1ka"       # 跳过 dinov3-total 所有 lambda 的 cls_in1ka
#   "hyperprior-featurecoding:falconmamba:0.02:*"       # 跳过 falconmamba lambda=0.02 的所有 dataset
SKIP_COMBINATIONS=(
    # "hyperprior-featurecoding:qwen:0.01:*"
    # "elic-featurecoding:qwen:0.01:*"
)

# Function to check if lambda is in filter list
in_lambda_filter() {
    local lambda="$1"
    if [[ ${#LAMBDA_FILTER[@]} -eq 0 ]]; then
        return 0  # No filter, accept all
    fi
    for l in "${LAMBDA_FILTER[@]}"; do
        if [[ "$lambda" == "$l" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to check if combination should be skipped
should_skip() {
    local arch="$1"
    local handler="$2"
    local lambda="$3"
    local dataset="$4"
    
    for skip in "${SKIP_COMBINATIONS[@]}"; do
        IFS=':' read -r skip_arch skip_handler skip_lambda skip_dataset <<< "$skip"
        
        local match_arch=false
        local match_handler=false
        local match_lambda=false
        local match_dataset=false
        
        [[ "$skip_arch" == "$arch" ]] && match_arch=true
        [[ "$skip_handler" == "$handler" ]] && match_handler=true
        [[ "$skip_lambda" == "*" || "$skip_lambda" == "$lambda" ]] && match_lambda=true
        [[ "$skip_dataset" == "*" || "$skip_dataset" == "$dataset" ]] && match_dataset=true
        
        if $match_arch && $match_handler && $match_lambda && $match_dataset; then
            return 0  # Should skip
        fi
    done
    return 1  # Should not skip
}

echo "Codec configurations: $CODEC_COUNT"
for ((i=0; i<CODEC_COUNT; i++)); do
    echo "  - ${CODEC_NAMES[$i]}: ${CODEC_ARCHS[$i]}"
done
echo "Handlers: ${HANDLERS[*]}"
echo "Lambda filter: ${LAMBDA_FILTER[*]}"
echo ""

for handler in "${HANDLERS[@]}"; do
    datasets_str="${HANDLER_DATASETS[$handler]}"
    
    if [[ -z "$datasets_str" ]]; then
        echo "WARNING: No datasets configured for handler '$handler', skipping..."
        continue
    fi
    
    # Count datasets
    dataset_count=$(echo "$datasets_str" | tr '|' '\n' | wc -l)
    
    echo "============================================"
    echo "HANDLER: $handler ($dataset_count datasets)"
    echo "============================================"
    
    # For each handler, complete all codec configs before moving to next handler
    for ((codec_idx=0; codec_idx<CODEC_COUNT; codec_idx++)); do
        WEIGHTS_DIR="${CODEC_WEIGHTS_DIRS[$codec_idx]}"
        ARCH="${CODEC_ARCHS[$codec_idx]}"
        CODEC_NAME="${CODEC_NAMES[$codec_idx]}"
        
        echo ""
        echo "  ----------------------------------------"
        echo "  CODEC: $CODEC_NAME ($ARCH)"
        echo "  ----------------------------------------"
        
        # Get all checkpoint files and extract lambda values, sort descending
        checkpoints=()
        lambdas=()
        
        if [[ -d "$WEIGHTS_DIR" ]]; then
            while IFS= read -r line; do
                if [[ -n "$line" ]]; then
                    ckpt_file=$(echo "$line" | cut -d':' -f1)
                    lambda_val=$(echo "$line" | cut -d':' -f2)
                    checkpoints+=("$ckpt_file")
                    lambdas+=("$lambda_val")
                fi
            done < <(
                for f in "$WEIGHTS_DIR"/*.pth.tar; do
                    if [[ -f "$f" ]]; then
                        filename=$(basename "$f")
                        if [[ "$filename" =~ lambda([0-9.]+) ]]; then
                            echo "$f:${BASH_REMATCH[1]}"
                        fi
                    fi
                done | sort -t':' -k2 -rn
            )
        fi
        
        echo "  Found ${#checkpoints[@]} checkpoints in $WEIGHTS_DIR"
        
        for ((ckpt_idx=0; ckpt_idx<${#checkpoints[@]}; ckpt_idx++)); do
            ckpt="${checkpoints[$ckpt_idx]}"
            lambda="${lambdas[$ckpt_idx]}"
            
            # Skip if lambda is not in the filter list (when filter is defined)
            if ! in_lambda_filter "$lambda"; then
                # echo "Skipping lambda $lambda (not in filter list)"
                continue
            fi
            
            echo ""
            echo "    Lambda: $lambda"
            echo "    Checkpoint: $(basename "$ckpt")"
            
            # Parse datasets
            IFS='|' read -ra ds_array <<< "$datasets_str"
            for ds_entry in "${ds_array[@]}"; do
                IFS=':' read -r ds_name ds_path <<< "$ds_entry"
                
                # Check if this combination should be skipped
                if should_skip "$ARCH" "$handler" "$lambda" "$ds_name"; then
                    echo "      SKIPPING: $ds_name with lambda=$lambda (in skip list)"
                    continue
                fi
                
                output_dir="output/$handler/lambda$lambda/$ARCH-8bit-individual/$ds_name"
                echo "      Dataset: $ds_name -> $output_dir"
                
                python feature_coding.py roundtrip \
                    "$ds_path" \
                    --output "$output_dir" \
                    --handler "$handler" \
                    --arch "$ARCH" \
                    --checkpoint "$ckpt" \
                    --strategy individual

            done
        done
    done
    
    echo ""
done

echo "All evaluations completed!"
