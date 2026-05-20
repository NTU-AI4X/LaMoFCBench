#!/usr/bin/env bash
# set -e          # 只要脚本发生错误就停止运行
# set -u          # 如果遇到不存在的变量就报错并停止执行
set -x          # 运行指令结果的时候，输出对应的指令
# set -o pipefail # 确保只要一个子命令失败，整个管道命令就失败

# hyperprior
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.001-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.001/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.004-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.004/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.007-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.007/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.01-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.01/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.02-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.02/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"

python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.001-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.001/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.004-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.004/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.007-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.007/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.01-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.01/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/hyperprior-lambda0.02-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.02/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"

python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.001-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.001/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.004-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.004/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.007-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.007/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.01-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.01/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.02-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.02/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"

python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.001-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.001/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.004-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.004/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.007-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.007/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.01-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.01/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/hyperprior-lambda0.02-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.02/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"

python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.001-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.001/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.004-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.004/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.007-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.007/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.01-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.01/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.02-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.02/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.39"

python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.001-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.001/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.004-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.004/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.007-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.007/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.01-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.01/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/hyperprior-lambda0.02-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.02/hyperprior-featurecoding-8bit-layerwise --skip_layer "blocks.9"

# elic
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.001-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.001/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.004-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.004/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.007-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.007/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.01-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.01/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.02-8bit-layerwise/cls_in_v_layer9 --load_root features/lambda0.02/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"

python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.001-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.001/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.004-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.004/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.007-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.007/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.01-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.01/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-val --data_root <imagenet-val_dinov3cls_data_root> --output output/elic-lambda0.02-8bit-layerwise/cls_in_v_layer39 --load_root features/lambda0.02/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"

python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.001-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.001/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.004-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.004/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.007-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.007/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.01-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.01/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.02-8bit-layerwise/cls_in_a_layer9 --load_root features/lambda0.02/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"

python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.001-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.001/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.004-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.004/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.007-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.007/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.01-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.01/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-adv --data_root <imagenet-a_dinov3cls_data_root> --output output/elic-lambda0.02-8bit-layerwise/cls_in_a_layer39 --load_root features/lambda0.02/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"

python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.001-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.001/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.004-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.004/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.007-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.007/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.01-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.01/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.02-8bit-layerwise/cls_in_r_layer9 --load_root features/lambda0.02/elic-featurecoding-8bit-layerwise --skip_layer "blocks.39"

python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.001-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.001/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.004-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.004/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.007-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.007/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.01-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.01/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
python dinov3cls.py --dataset imagenet1k-ren --data_root <imagenet-r_dinov3cls_data_root> --output output/elic-lambda0.02-8bit-layerwise/cls_in_r_layer39 --load_root features/lambda0.02/elic-featurecoding-8bit-layerwise --skip_layer "blocks.9"
