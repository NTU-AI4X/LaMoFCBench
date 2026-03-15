"""
DINOv3 Feature Coding Evaluation Script.

该脚本评估被codec重构后的特征对原始模型下游任务性能的影响。

支持两种模式:
- multi: 同时替换所有4层中间层特征 (blocks.9/19/29/39)，通过替换 backbone 的 _get_intermediate_layers_not_chunked 方法实现
- single: 仅替换指定单层特征，通过 forward hooks 实现，允许观察单层替换对深层特征的级联影响

使用示例:
    # Multi 模式 (默认)
    python featurecoding_for_dinov3dep_evalrec.py --mode multi --data_root ... --load_root ...

    # Single 模式 (仅替换 blocks.9)
    python featurecoding_for_dinov3dep_evalrec.py --mode single --target_layer 9 --data_root ... --load_root ...
"""

import argparse
import io
import math
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov3.eval.depth.metrics import DEPTH_METRICS, DEPTH_METRICS_NAME
from dinov3.eval.depth.utils import align_depth_least_square
from dinov3.hub.depthers import dinov3_vit7b16_dd
from PIL import Image
from tabulate import SEPARATING_LINE, tabulate
from torch import nn
from torch.utils import data
from torchvision.transforms import v2
from tqdm import tqdm

sys.path.insert(0, "../../lmfc_utils")
from handlers.utils import compute_mse, inspect_structure, load_tensor_using_ref, load_zst_tensor

# os.environ["TORCH_HOME"] = "/root/.cache/torch"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TeeLogger:
    """将输出同时写入终端和日志文件的类。"""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class NYUDepthV2(data.Dataset):
    DEPTH_SCALE = 1000.0
    MIN_DEPTH_M = 0.001  # m
    MAX_DEPTH_M = 10  # m

    def __init__(self, data_root: str):
        self.samples = []
        for scene_root in Path(data_root).iterdir():
            if not scene_root.is_dir():
                continue
            for file_path in scene_root.iterdir():
                file_name = file_path.name
                if file_name.startswith("rgb_"):
                    image_path = file_path
                    depth_path = scene_root / file_name.replace("rgb_", "sync_depth_")
                    depth_path = depth_path.with_suffix(".png")
                    self.samples.append((image_path, depth_path))
        tqdm.write(f"Total: {len(self.samples)} pairs of rgb and depth images.")
        # self.samples = self.samples[:5]

        self.transforms = v2.Compose(
            [
                T.ToTensor(),
                T.Resize(size=768, interpolation=T.InterpolationMode.BILINEAR),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, depth_path = self.samples[idx]

        image = self.load_image(image_path)  # H,W
        image = self.transforms(image)  # 3,H,W
        return {"image": image, "image_path": image_path.as_posix(), "depth_path": depth_path.as_posix()}

    def load_image(self, path):
        with open(path, mode="rb") as f:
            image_data = f.read()

        f = io.BytesIO(image_data)
        return Image.open(f).convert(mode="RGB")

    def load_depth(self, path):
        with open(path, mode="rb") as f:
            target_data = f.read()

        f = io.BytesIO(target_data)
        return Image.open(f)

    def get_depth_map(self, depth_path: Path):
        """Image.open() -> np.asarray(np.float32) -> / 1000.0"""
        depth = self.load_depth(depth_path)  # H,W,millimeter (uint16)
        depth = np.asarray(depth, dtype=np.float32) / self.DEPTH_SCALE  # millimeter -> meter
        return depth


class ImageLeftRightFlipAug:
    """
    Test time augmentation for depth estimation
    from https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/processing.py#L721

    this is just returning two versions of the same image, and the according labels
    """

    def __init__(self, flip: bool = False):
        self._flip = flip

    def __call__(self, img):
        """Call function to apply test time augment transforms on results.

        Args:
            img: Data to transform.

        Returns:
            list: A list of augmented data.
        """

        do_flips = [False, True] if self._flip else [False]
        results_images = []
        for do_flip in do_flips:
            image_aug = TF.hflip(img) if do_flip else img
            results_images.append(image_aug)
        return results_images

    def inverse(self, stacked_lr_pair: torch.Tensor) -> torch.Tensor:
        if not self._flip:
            return stacked_lr_pair

        pre_aug_batch_size = stacked_lr_pair.shape[0] // 2
        assert pre_aug_batch_size * 2 == stacked_lr_pair.shape[0]
        return (stacked_lr_pair[:pre_aug_batch_size] + TF.hflip(stacked_lr_pair[pre_aug_batch_size:])) / 2


class FeatureCodingWrapper:
    """
    DINOv3 深度估计模型的 Feature Coding 包装器。

    支持两种模式:
    - multi: 替换 backbone._get_intermediate_layers_not_chunked 方法，同时替换所有4层中间层特征
    - single: 通过 forward hooks 仅替换指定单层特征
    """

    def __init__(self, load_root=None, mode="multi", target_layer=None):
        """
        初始化 FeatureCodingWrapper。

        Args:
            load_root: 加载重构特征的路径
            mode: 模式选择, "multi" 或 "single"
            target_layer: 目标层列表, 例如 9
        """
        self.model = dinov3_vit7b16_dd(pretrained=True, autocast_dtype=torch.float32, check_hash=True)

        # 根据模式设置特征替换策略
        self.mode = mode
        self.target_layer = target_layer
        if self.mode == "multi":
            tqdm.write("[Mode: MULTI] 替换 backbone._get_intermediate_layers_not_chunked 方法...")
            self.model.encoder.backbone._get_intermediate_layers_not_chunked = types.MethodType(
                self._create_multi_hook(), self.model.encoder.backbone
            )
            self.hook_handles = []
        elif self.mode == "single":
            target_layer_names = [f"encoder.backbone.blocks.{self.target_layer}"]
            tqdm.write(f"[Mode: SINGLE] 注册 forward hooks 到层: {target_layer_names}")
            # Target layers for feature extraction
            self.hook_handles = self._register_forward_hooks(target_layer_names)
        else:
            raise ValueError(f"未知模式: {self.mode}, 请选择 'multi' 或 'single'")

        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.load_root = load_root
        self.sample_fc_stats = []

    def set_sample_info(self, sample_id: str, current_idx: int = 0, total_samples: int = 0):
        """Set current sample info for logging."""
        self.sample_id = sample_id

        # 显示进度信息
        tqdm.write(
            tabulate(
                [[f"[{current_idx}/{total_samples}]", sample_id, self.load_root]],
                headers=["Progress", "Sample", "Load Root"],
            )
        )

        load_path = os.path.join(self.load_root, f"{sample_id}.zst")
        assert os.path.exists(load_path), load_path
        reconstructed_data = load_zst_tensor(file_path=load_path)

        if "metadata" in reconstructed_data:
            self.current_target_metadata = {**reconstructed_data["metadata"]}
            meta_msg = [
                [
                    k,
                    v
                    if not isinstance(v, dict)
                    else {
                        _k: _v if _k != "enc_strings" else f"{sum(len(s) for sl in _v for s in sl)} Bytes"
                        for _k, _v in v.items()
                    },
                ]
                for k, v in self.current_target_metadata.items()
            ]
            meta_msg = tabulate(meta_msg, headers=["Key", "Value"])
            tqdm.write(f"Loaded features from {load_path} with meta information:\n{meta_msg}")

            self.current_target_features = {
                # 因为映射问题，导致生成的feature中的名称全部统一成了segmentation的版本，需要重新映射
                "encoder.backbone.blocks.9": reconstructed_data["features"]["layer.9"],
                "encoder.backbone.blocks.19": reconstructed_data["features"]["layer.19"],
                "encoder.backbone.blocks.29": reconstructed_data["features"]["layer.29"],
                "encoder.backbone.blocks.39": reconstructed_data["features"]["layer.39"],
            }
        else:
            self.current_target_metadata = {}
            self.current_target_features = {**reconstructed_data}

        tqdm.write(f"Loaded features {self.current_target_features.keys()} from {reconstructed_data.keys()}.")
        inspect_structure(self.current_target_features)

    def _create_multi_hook(self):
        """创建用于 multi 模式的 backbone 方法替换 hook。"""
        wrapper_self = self  # 闭包捕获 wrapper 实例

        def hook(backbone_self, x: torch.Tensor, n: int = 1) -> List[torch.Tensor]:
            x, (H, W) = backbone_self.prepare_tokens_with_masks(x)
            # If n is an int, take the n last blocks. If it's a list, take them
            outputs, total_block_len = [], len(backbone_self.blocks)
            blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
            for i, blk in enumerate(backbone_self.blocks):
                if backbone_self.rope_embed is not None:
                    rope_sincos = backbone_self.rope_embed(H=H, W=W)
                else:
                    rope_sincos = None
                x = blk(x, rope_sincos)
                if i in blocks_to_take:
                    outputs.append(x)
            assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} blocks found"

            tqdm.write(f"-->>\tSample {wrapper_self.sample_id} <<--")
            inspect_structure(outputs, prefix="Intermediate Layers Outputs", print_fn=tqdm.write)

            # Feature coding: encode and decode in memory
            if wrapper_self.load_root:
                assert wrapper_self.current_target_features, "No features to load!"

                for idx, output in enumerate(outputs):
                    name = f"encoder.backbone.blocks.{blocks_to_take[idx]}"
                    tqdm.write(f"[Feature Coding] Replace {name}...")

                    _loaded = wrapper_self.current_target_features[name]
                    _loaded = load_tensor_using_ref(_loaded, ref=output)
                    _mse_sum = compute_mse(_loaded, output)
                    wrapper_self.total_mse_sum += _mse_sum
                    _numel = _loaded.numel()
                    wrapper_self.total_numel += _numel
                    wrapper_self.sample_mse_msg.append([name, f"{_mse_sum / _numel:.8f}"])

                    output.zero_()
                    output.copy_(_loaded)

            return outputs

        return hook

    def _get_single_hook(self, name):
        """创建用于 single 模式的 forward hook。"""

        def hook(module, inputs, outputs):
            tqdm.write(f"-->>\t{name} for Sample {self.sample_id} <<--")
            inspect_structure(outputs, prefix="Output", print_fn=tqdm.write)

            # Feature coding: encode and decode in memory
            if self.load_root:
                assert self.current_target_features, "No features to load!"

                tqdm.write(f"[Feature Coding] Replace {name}...")
                _loaded = self.current_target_features[name]
                _loaded = load_tensor_using_ref(_loaded, ref=outputs)
                _mse_sum = compute_mse(_loaded, outputs)
                self.total_mse_sum += _mse_sum
                _numel = _loaded.numel()
                self.total_numel += _numel
                self.sample_mse_msg.append([name, f"{_mse_sum / _numel:.8f}"])

                outputs.zero_()
                outputs.copy_(_loaded)
            return outputs

        return hook

    def _register_forward_hooks(self, layer_names: list):
        """注册 forward hooks 到指定层。"""
        hooks = []
        for name in layer_names:
            module: nn.Module = self.model.get_submodule(name)
            tqdm.write(f"[Feature Coding] Register hook for {name}...")
            hooks.append(module.register_forward_hook(self._get_single_hook(name)))
        return hooks

    def remove_hooks(self):
        """移除所有注册的 forward hooks。"""
        if not self.hook_handles:
            return
        num_hooks = len(self.hook_handles)
        for hook in self.hook_handles:
            hook.remove()
        self.hook_handles.clear()
        tqdm.write(f"Removed {num_hooks} hooks!")

    @torch.no_grad()
    def predict(self, x):
        self.sample_mse_msg = []
        self.total_mse_sum = 0
        self.total_numel = 0

        x = self.model(x)

        if self.load_root:
            mse_msg = tabulate(self.sample_mse_msg, headers=["Tensor", "MSE"])
            tqdm.write(f"[Feature Coding] Done. Injected decoded features with MSE:\n{mse_msg}")
            # Append per-sample stats
            self.sample_fc_stats.append(
                {
                    "elements": self.total_numel,
                    "mse_recalc": self.total_mse_sum / self.total_numel,
                    # information from target_metadata
                    "arch": self.current_target_metadata.get("arch", None),
                    "handler": self.current_target_metadata.get("handler", None),
                    "strategy": self.current_target_metadata.get("strategy", None),
                    "transform_type": self.current_target_metadata.get("transform_type", None),
                    "bit_depth": self.current_target_metadata.get("bit_depth", None),
                    "bpfp": self.current_target_metadata.get("bpfp", -1),
                    "ebpfp": self.current_target_metadata.get("ebpfp", -1),
                    "mse": self.current_target_metadata.get("mse", -1),
                }
            )
        return x

    def print_feature_coding_stats(self):
        """Print accumulated feature coding statistics per group."""
        if not self.sample_fc_stats:
            return

        # Print stats for each group separately
        num_samples = len(self.sample_fc_stats)
        if num_samples == 0:
            tqdm.write("No feature coding stats")
            return

        # Compute averages from totals
        total_elements = sum(s["elements"] for s in self.sample_fc_stats)
        per_sample_avg_bpfp = sum(s["bpfp"] for s in self.sample_fc_stats) / num_samples
        per_sample_avg_ebpfp = sum(s["ebpfp"] for s in self.sample_fc_stats) / num_samples
        per_sample_avg_mse = sum(s["mse"] for s in self.sample_fc_stats) / num_samples
        per_sample_avg_mse_recalc = sum(s["mse_recalc"] for s in self.sample_fc_stats) / num_samples

        # Print summary table
        tqdm.write(
            tabulate(
                [
                    ["Mode", self.mode.upper()],
                    ["Target Layers", self.target_layer],
                    SEPARATING_LINE,
                    ["Feature Coding Calls", num_samples],
                    ["Total Elements", total_elements],
                    SEPARATING_LINE,
                    ["AVERAGE", "Value"],
                    ["BPFP (bits/point)", per_sample_avg_bpfp],
                    ["RBPFP (relative bits/point)", per_sample_avg_ebpfp],
                    ["Total-MSE (from pre-calculation)", per_sample_avg_mse],
                    ["Total-MSE (from re-calculation)", per_sample_avg_mse_recalc],
                ],
                headers=["FEATURE CODING STATISTICS", ""],
            )
        )


class DepthEstimationEvaluator:
    def __init__(self, data_root: str, load_root=None, mode="multi", target_layer=None):
        self.dataset = NYUDepthV2(data_root=data_root)
        tqdm.write(f"Initialized the dataset {self.dataset.__class__.__name__}...")
        self.dataloader = data.DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)
        tqdm.write(f"Evaluate {len(self.dataset)} samples with {len(self.dataloader)} batches.")

        tqdm.write("Initialize the DINOv3 model...")
        self.model = FeatureCodingWrapper(load_root=load_root, mode=mode, target_layer=target_layer)

        self.ignored_value = 0
        self.min_depth = NYUDepthV2.MIN_DEPTH_M
        self.max_depth = NYUDepthV2.MAX_DEPTH_M

        self.garg_crop: bool = False
        self.eigen_crop: bool = True
        self.results = []

    @staticmethod
    def make_valid_mask(input, eval_crop: str, ignored_value: float = 0.0):
        """Following Adabins, Do grag_crop or eigen_crop for testing
        Args:
            input: input tensor in HxW format
            eval_crop (_EvalCropType): evaluation crop used for evaluation
            ignored_value (float): value from input to be ignored during evaluation
        """
        h, w = input.shape
        eval_mask = torch.zeros(input.shape, device=input.device)
        if eval_crop == "NYU_EIGEN":
            y1, y2, x1, x2 = 45, 471, 41, 601
            orig_h, orig_w = 480, 640
            y1_new = int((y1 / orig_h) * h)
            y2_new = int((y2 / orig_h) * h)
            x1_new = int((x1 / orig_w) * w)
            x2_new = int((x2 / orig_w) * w)
            eval_mask[y1_new:y2_new, x1_new:x2_new] = 1
        elif eval_crop == "FULL":
            eval_mask.fill_(1)
        else:
            raise NotImplementedError(eval_crop)

        # make mask from ignored values
        ignored_value_mask = torch.ones((h, w), device=eval_mask.device)
        ignored_value_mask[input == ignored_value] = 0

        eval_mask = eval_mask * ignored_value_mask
        return eval_mask.bool()

    @staticmethod
    def calculate_depth_metrics(gt, pred, valid_mask, list_metrics=list(DEPTH_METRICS)) -> dict:
        assert gt.shape == pred.shape, (gt.shape, pred.shape)
        valid_mask = torch.logical_and(valid_mask, gt > 0)
        gt = gt[valid_mask]
        pred = pred[valid_mask]

        metric_names = [metric.name for metric in list_metrics]

        metrics_dict = {}
        thresh = torch.maximum((gt / pred), (pred / gt))
        metrics_dict["a1"] = (thresh < 1.25).float().mean() if "a1" in metric_names else torch.nan
        metrics_dict["a2"] = (thresh < 1.25**2).float().mean() if "a2" in metric_names else torch.nan
        metrics_dict["a3"] = (thresh < 1.25**3).float().mean() if "a3" in metric_names else torch.nan

        error = gt - pred
        sq_error = error**2
        metrics_dict["mae"] = torch.mean(torch.abs(error)) if "mae" in metric_names else torch.nan
        metrics_dict["abs_rel"] = torch.mean(torch.abs(error) / gt) if "abs_rel" in metric_names else torch.nan
        metrics_dict["sq_rel"] = torch.mean(sq_error / gt) if "sq_rel" in metric_names else torch.nan

        metrics_dict["rmse"] = torch.sqrt(sq_error.mean()) if "rmse" in metric_names else torch.nan

        error_log = torch.log(gt) - torch.log(pred)
        sq_error_log = error_log**2
        metrics_dict["rmse_log"] = torch.sqrt(sq_error_log.mean()) if "rmse_log" in metric_names else torch.nan
        if "silog" in metric_names:
            silog = torch.sqrt(torch.mean(sq_error_log) - torch.mean(error_log) ** 2) * 100
            if torch.isnan(silog):
                silog = torch.tensor(0)
            metrics_dict["silog"] = silog
        else:
            metrics_dict["silog"] = torch.nan
        metrics_dict["log_10"] = (
            (torch.abs(torch.log10(gt) - torch.log10(pred))).mean() if "log_10" in metric_names else math.inf
        )
        return metrics_dict

    @torch.no_grad()
    def evaluate(self):
        t_dataset_start = time.perf_counter()

        self.tta_transform = ImageLeftRightFlipAug(flip=True)

        total_samples = len(self.dataloader)
        for idx, sample in tqdm(enumerate(self.dataloader), total=total_samples, ncols=78):
            t_sample_start = time.perf_counter()

            images: torch.Tensor = sample["image"].to(DEVICE)
            image_paths = sample["image_path"]
            assert images.shape[0] == 1 and len(image_paths) == 1

            image_path = Path(image_paths[0])
            image_scene = image_path.parent.name
            image_stem = image_path.stem

            paired_images = self.tta_transform(img=images)
            images = torch.cat(paired_images, dim=0)  # 2B,3,H,W

            sample_id = f"{image_scene}-{image_stem}-wflip"
            if self.model.load_root:
                self.model.set_sample_info(sample_id=sample_id, current_idx=idx, total_samples=total_samples)
            preds = self.model.predict(images)  # 2B,1,H,W, 0.001m~100m
            preds = self.tta_transform.inverse(preds)  # B,1,H,W

            for idx_in_batch in range(preds.shape[0]):
                depth_path = Path(sample["depth_path"][idx_in_batch])
                depth = torch.from_numpy(self.dataset.get_depth_map(depth_path)).to(device=preds.device)
                depth = torch.where(
                    torch.logical_or(depth >= self.max_depth, depth <= self.min_depth),
                    self.ignored_value,
                    depth,
                )
                valid_mask = self.make_valid_mask(depth, eval_crop="NYU_EIGEN", ignored_value=self.ignored_value)

                pred = preds[idx_in_batch : idx_in_batch + 1]
                if pred.shape[-2:] != depth.shape[-2:]:
                    pred = F.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=False)
                pred = pred.squeeze()
                pred = align_depth_least_square(depth, pred, valid_mask)[0]

                results = self.calculate_depth_metrics(depth, pred, valid_mask)
                results = np.asarray([results[m].cpu().numpy() for m in DEPTH_METRICS_NAME]).reshape(1, -1)
                self.results.append(results)

            t_sample_end = time.perf_counter()
            tqdm.write(f"Sample {sample_id} took {(t_sample_end - t_sample_start):.2f} seconds")

        results = np.concatenate(self.results, axis=0)  # Nx10
        assert results.shape[-1] == 10, results.shape
        results = np.nanmean(results, axis=0).tolist()
        results = {k: v for k, v in zip(DEPTH_METRICS_NAME, results)}
        tqdm.write(f"Depth Estimation: {results}")

        t_dataset_end = time.perf_counter()
        tqdm.write(f"Dataset took {(t_dataset_end - t_dataset_start):.2f} seconds")

        # Print feature coding stats
        self.model.print_feature_coding_stats()

        # Remove hooks
        self.model.remove_hooks()


def main():
    # fmt: off
    parser = argparse.ArgumentParser(
        description="DINOv3 Depth Estimation Feature Coding Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
        # Multi 模式 (默认): 同时替换所有4层中间层特征
        python featurecoding_for_dinov3dep_evalrec.py --mode multi --data_root ... --load_root ...

        # Single 模式: 仅替换 blocks.9
        python featurecoding_for_dinov3dep_evalrec.py --mode single --target_layer 9 --data_root ... --load_root ...
        """,
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to grouped images")
    parser.add_argument("--output", type=str, default="new_results", help="Output directory for logs")
    parser.add_argument("--load_root", type=str, help="Path to loading features")
    parser.add_argument("--mode", type=str, choices=["multi", "single"], default="multi", help="Feature replacement mode: 'multi' (replace all 4 layers via method override) or 'single' (replace via forward hooks)")
    parser.add_argument("--target_layer", type=int, help="Target layer index to replace for the single mode, default is None")
    args = parser.parse_args()
    # fmt: on

    if args.mode == "single":
        assert args.target_layer is not None, "Target layer index must be specified for the single mode"

    os.makedirs(args.output, exist_ok=True)

    log_filename = f"dinov3dep_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = Path(args.output) / log_filename
    tee_logger = TeeLogger(str(log_filepath))
    original_stdout = sys.stdout
    sys.stdout = tee_logger
    tqdm.write(f"日志将写入: {log_filepath}")
    tqdm.write(f"模式: {args.mode.upper()}, 目标层: {args.target_layer}")

    evaluator = DepthEstimationEvaluator(
        data_root=args.data_root, load_root=args.load_root, mode=args.mode, target_layer=args.target_layer
    )
    evaluator.evaluate()

    # 恢复 stdout 并关闭日志文件
    tqdm.write(f"\n日志已保存至: {log_filepath}")
    sys.stdout = original_stdout
    tee_logger.close()


if __name__ == "__main__":
    main()
