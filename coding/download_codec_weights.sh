#!/usr/bin/env bash
set -e          # 只要脚本发生错误就停止运行
set -u          # 如果遇到不存在的变量就报错并停止执行
set -x          # 运行指令结果的时候，输出对应的指令
set -o pipefail # 确保只要一个子命令失败，整个管道命令就失败

pip install -U huggingface_hub

export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# export HF_ENDPOINT=https://hf-mirror.com

hf download chansongoal/DT-UFC --repo-type model --include "hyperprior_hybrid/bmshj2018-hyperprior_lambda0.02_*" --local-dir ./codec_weights
hf download chansongoal/DT-UFC --repo-type model --include "hyperprior_hybrid/bmshj2018-hyperprior_lambda0.01_*" --local-dir ./codec_weights
hf download chansongoal/DT-UFC --repo-type model --include "hyperprior_hybrid/bmshj2018-hyperprior_lambda0.007_*" --local-dir ./codec_weights
hf download chansongoal/DT-UFC --repo-type model --include "hyperprior_hybrid/bmshj2018-hyperprior_lambda0.004_*" --local-dir ./codec_weights
hf download chansongoal/DT-UFC --repo-type model --include "hyperprior_hybrid/bmshj2018-hyperprior_lambda0.001_*" --local-dir ./codec_weights

hf download chansongoal/DT-UFC --repo-type model --include "elic_hybrid/elic2022-official_lambda0.02_*" --local-dir ./codec_weights
hf download chansongoal/DT-UFC --repo-type model --include "elic_hybrid/elic2022-official_lambda0.01_*" --local-dir ./codec_weights
hf download chansongoal/DT-UFC --repo-type model --include "elic_hybrid/elic2022-official_lambda0.007_*" --local-dir ./codec_weights
hf download chansongoal/DT-UFC --repo-type model --include "elic_hybrid/elic2022-official_lambda0.004_*" --local-dir ./codec_weights
hf download chansongoal/DT-UFC --repo-type model --include "elic_hybrid/elic2022-official_lambda0.001_*" --local-dir ./codec_weights
