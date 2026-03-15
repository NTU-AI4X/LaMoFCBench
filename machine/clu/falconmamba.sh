#!/usr/bin/env bash
# set -e          # 只要脚本发生错误就停止运行
# set -u          # 如果遇到不存在的变量就报错并停止执行
set -x          # 运行指令结果的时候，输出对应的指令
# set -o pipefail # 确保只要一个子命令失败，整个管道命令就失败

python falconmamba.py --output output/hyperprior-lambda0.001-8bit-individual --load_root features/lambda0.001/hyperprior-featurecoding-8bit-individual
python falconmamba.py --output output/hyperprior-lambda0.004-8bit-individual --load_root features/lambda0.004/hyperprior-featurecoding-8bit-individual
python falconmamba.py --output output/hyperprior-lambda0.007-8bit-individual --load_root features/lambda0.007/hyperprior-featurecoding-8bit-individual
python falconmamba.py --output output/hyperprior-lambda0.01-8bit-individual --load_root features/lambda0.01/hyperprior-featurecoding-8bit-individual
python falconmamba.py --output output/hyperprior-lambda0.02-8bit-individual --load_root features/lambda0.02/hyperprior-featurecoding-8bit-individual

python falconmamba.py --output output/elic-lambda0.001-8bit-individual --load_root features/lambda0.001/elic-featurecoding-8bit-individual
python falconmamba.py --output output/elic-lambda0.004-8bit-individual --load_root features/lambda0.004/elic-featurecoding-8bit-individual
python falconmamba.py --output output/elic-lambda0.007-8bit-individual --load_root features/lambda0.007/elic-featurecoding-8bit-individual
python falconmamba.py --output output/elic-lambda0.01-8bit-individual --load_root features/lambda0.01/elic-featurecoding-8bit-individual
python falconmamba.py --output output/elic-lambda0.02-8bit-individual --load_root features/lambda0.02/elic-featurecoding-8bit-individual