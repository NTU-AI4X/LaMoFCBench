"""
DINOv3 Feature Coding Evaluation Script.

该脚本仅评估被codec重构后的特征自身对原始模型下游任务上的性能。
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import torch
from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.classifiers import dinov3_vit7b16_lc
from tabulate import SEPARATING_LINE, tabulate
from torch import nn
from torch.utils import data
from torchvision import datasets
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


class ImageNet1K_A_DINOv3(datasets.ImageFolder):
    def __init__(self, data_root, transform=make_classification_eval_transform(resize_size=512, crop_size=512)):
        super().__init__(root=data_root, transform=transform)

    @staticmethod
    def get_index_mapping_1k_to_subval():
        # https://github.com/hendrycks/imagenet-r/blob/3131cacee97c407bd2ecea15846a67817a64f924/eval.py#L17-L25
        with open("imagenet1k_classwnids.json", "r") as f:
            all_wnids = json.load(f)
        all_wnids = sorted(all_wnids)

        with open("imagenet1k-val_data_infos_dinov3cls.json", "r") as f:
            sub_class_infos = json.load(f)
        # (wnid, [{"file_name": xxx, "class_score": xxx}])
        sorted_sub_class_infos: list = sorted(sub_class_infos.items(), key=lambda kv: kv[0])  # 基于key wnid排序

        class_id_sub_to_full = {}
        for curr_class_id, curr_sample_info in enumerate(sorted_sub_class_infos):
            curr_class_wnid = curr_sample_info[0]
            class_id_sub_to_full[curr_class_id] = all_wnids.index(curr_class_wnid)
        assert len(class_id_sub_to_full) == len(sub_class_infos)

        return [i for i in range(1000)], class_id_sub_to_full

    @staticmethod
    def get_index_mapping_1k_to_adv200():
        # https://github.com/hendrycks/natural-adv-examples/blob/07770705658c3a1c8acce31fd9dbd68f06e297c3/eval.py#L12
        with open("imagenet1k_a_classinfo.json", "r") as f:
            full_class_infos = json.load(f)

        # 取了子集之后，类别对应的id发生变化，需要重新将其映射到原本的200类中
        with open("imagenet1k-adv_data_infos_dinov3cls.json", "r") as f:
            sub_class_infos = json.load(f)
        # (wnid, [{"file_name": xxx, "class_score": xxx}])
        sorted_sub_class_infos: list = sorted(sub_class_infos.items(), key=lambda kv: kv[0])  # 基于key wnid排序

        class_id_sub_to_full = {}
        for curr_class_id, curr_sample_info in enumerate(sorted_sub_class_infos):
            curr_class_wnid = curr_sample_info[0]
            for class_info_in_a in full_class_infos:
                if curr_class_wnid == class_info_in_a["class_wnid"]:
                    class_id_sub_to_full[curr_class_id] = class_info_in_a["class_id_a"]
                    break

        assert len(class_id_sub_to_full) == len(sub_class_infos)
        return [info["class_id_1k"] for info in full_class_infos], class_id_sub_to_full

    @staticmethod
    def get_index_mapping_1k_to_ren200():
        # https://github.com/hendrycks/imagenet-r/blob/3131cacee97c407bd2ecea15846a67817a64f924/eval.py#L17-L25
        with open("imagenet1k_r_classinfo.json", "r") as f:
            full_class_infos = json.load(f)

        # 取了子集之后，类别对应的id发生变化，需要重新将其映射到原本的200类中
        with open("imagenet1k-ren_data_infos_dinov3cls.json", "r") as f:
            sub_class_infos = json.load(f)
        # (wnid, [{"file_name": xxx, "class_score": xxx}])
        sorted_sub_class_infos: list = sorted(sub_class_infos.items(), key=lambda kv: kv[0])  # 基于key wnid排序

        class_id_sub_to_full = {}
        for curr_class_id, curr_sample_info in enumerate(sorted_sub_class_infos):
            curr_class_wnid = curr_sample_info[0]
            for class_info_in_a in full_class_infos:
                if curr_class_wnid == class_info_in_a["class_wnid"]:
                    class_id_sub_to_full[curr_class_id] = class_info_in_a["class_id_r"]
                    break

        assert len(class_id_sub_to_full) == len(sub_class_infos)
        return [info["class_id_1k"] for info in full_class_infos], class_id_sub_to_full

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # return the path of image to construct the filename of the intermediate feature map
        return sample, target, path


class FeatureCodingWrapper:
    def __init__(self, load_root: str = "", skip_layer: str = ""):
        self.model = dinov3_vit7b16_lc(pretrained=True, autocast_dtype=torch.float32, check_hash=True)
        self.model = self.model.to(DEVICE)
        self.model.eval()

        # Target layers for feature extraction
        target_layer_names = ["backbone.blocks.9", "backbone.blocks.39"]
        self.hook_handles = self.register_forward_hooks(
            [layer_name for layer_name in target_layer_names if not skip_layer or not layer_name.endswith(skip_layer)]
        )
        self.load_root = load_root
        self.sample_fc_stats = []

    def set_sample_info(self, sample_id: str = "", current_idx: int = 0, total_samples: int = 0):
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
                "backbone.blocks.9": reconstructed_data["features"]["segmentation_model.0.backbone.blocks.9"],
                "backbone.blocks.39": reconstructed_data["features"]["segmentation_model.0.backbone.blocks.39"],
            }
        else:
            self.current_target_metadata = {}
            self.current_target_features = {**reconstructed_data}

        tqdm.write(f"Loaded features {self.current_target_features.keys()}.")
        inspect_structure(self.current_target_features)

    def get_hook(self, name):
        def hook(module, inputs, outputs):
            tqdm.write(f"-->>\t{name} for Sample {self.sample_id} <<--")
            inspect_structure(outputs, prefix="Output", print_fn=tqdm.write)

            # Feature coding: encode and decode in memory
            if self.load_root:
                assert self.current_target_features, "No features to load!"

                tqdm.write(f"[Feature Coding] Replace {name}...")

                _loaded = self.current_target_features[name][0]
                _loaded = load_tensor_using_ref(_loaded, ref=outputs[0])
                _mse_sum = compute_mse(_loaded, outputs[0])
                self.total_mse_sum += _mse_sum
                _numel = _loaded.numel()
                self.total_numel += _numel
                self.sample_mse_msg.append([name, f"{_mse_sum / _numel:.8f}"])

                outputs[0].zero_()
                outputs[0].copy_(_loaded)
            return outputs

        return hook

    def register_forward_hooks(self, layer_names: dict):
        hooks = []
        for name in layer_names:
            module: nn.Module = self.model.get_submodule(name)
            tqdm.write(f"[Feature Coding] Register hook for {name}...")
            hooks.append(module.register_forward_hook(self.get_hook(name)))
        return hooks

    def remove_hooks(self):
        """Remove all registered forward hooks to avoid memory leaks."""
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

        x = self.model.backbone.forward_features(x)
        x = torch.cat([x["x_norm_clstoken"], x["x_norm_patchtokens"].mean(dim=1)], dim=1)
        x = self.model.linear_head(x)

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


class IN1K_Cls_Evaluator:
    def __init__(self, data_root: str, load_root: str = "", skip_layer: str = ""):
        self.dataset = ImageNet1K_A_DINOv3(data_root=data_root)
        tqdm.write(f"Initialized the dataset {self.dataset.__class__.__name__}...")
        self.dataloader = data.DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)
        tqdm.write(f"Evaluate {len(self.dataset)} samples with {len(self.dataloader)} batches.")

        tqdm.write("Initialize the DINOv3 model...")
        self.model = FeatureCodingWrapper(load_root=load_root, skip_layer=skip_layer)

    @torch.no_grad()
    def _base_pipeline(self, indices_in_1k: list = None, sub_to_full: dict = None):
        t_dataset_start = time.perf_counter()

        num_correct = 0
        total_samples = len(self.dataloader)
        for idx, (sample, target, path) in enumerate(tqdm(self.dataloader)):
            t_sample_start = time.perf_counter()

            path = Path(path[0])
            sample_id = f"{path.parent.name}-{path.stem}"

            sample = sample.to(DEVICE)
            if sub_to_full:
                target = torch.as_tensor([sub_to_full[t] for t in target.tolist()])

            # Set sample info for feature coding
            self.model.set_sample_info(sample_id=sample_id, current_idx=idx + 1, total_samples=total_samples)
            output = self.model.predict(sample)
            if indices_in_1k:
                output = output[:, indices_in_1k]

            score, index = output.detach().cpu().max(1)
            are_correct = index.eq(target)
            num_correct += are_correct.sum().item()

            t_sample_end = time.perf_counter()
            tqdm.write(f"Sample {sample_id} took {(t_sample_end - t_sample_start):.2f} seconds")

        acc_classification = num_correct / len(self.dataset)
        tqdm.write(f"Classification Accuracy: {acc_classification:.4f}")

        t_dataset_end = time.perf_counter()
        tqdm.write(f"Dataset took {(t_dataset_end - t_dataset_start):.2f} seconds")

        # Print feature coding stats
        self.model.print_feature_coding_stats()

        # Remove hooks
        self.model.remove_hooks()
        return acc_classification

    def evaluate_on_imagenet1k_adv(self):
        tqdm.write("Evaluate on ImageNet-A (Adversarial)...")
        indices_in_1k, sub_to_full = self.dataset.get_index_mapping_1k_to_adv200()
        return self._base_pipeline(indices_in_1k=indices_in_1k, sub_to_full=sub_to_full)

    def evaluate_on_imagenet1k_ren(self):
        tqdm.write("Evaluate on ImageNet-R (Renditions)...")
        indices_in_1k, sub_to_full = self.dataset.get_index_mapping_1k_to_ren200()
        return self._base_pipeline(indices_in_1k=indices_in_1k, sub_to_full=sub_to_full)

    def evaluate_on_imagenet1k_val(self):
        tqdm.write("Evaluate on ImageNet-1K Validation...")
        indices_in_1k, sub_to_full = self.dataset.get_index_mapping_1k_to_subval()
        return self._base_pipeline(indices_in_1k=indices_in_1k, sub_to_full=sub_to_full)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="DINOv3 Feature Coding Classification Evaluation")
    parser.add_argument("--dataset", required=True, choices=["imagenet1k-val", "imagenet1k-adv", "imagenet1k-ren"], help="Dataset to evaluate on")
    parser.add_argument("--data_root", type=str, required=True, help="Path to grouped images")
    parser.add_argument("--output", type=str, default="new_results", help="Output directory for logs")
    parser.add_argument("--load_root", type=str, default="", help="Path for saving features")
    parser.add_argument("--skip_layer", type=str, choices=["blocks.9", "blocks.39"], help="Layer name to skip")
    args = parser.parse_args()
    # fmt: on

    os.makedirs(args.output, exist_ok=True)

    log_filename = f"dinov3_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = Path(args.output) / log_filename
    tee_logger = TeeLogger(str(log_filepath))
    original_stdout = sys.stdout
    sys.stdout = tee_logger
    tqdm.write(f"日志将写入: {log_filepath}")

    if args.dataset == "imagenet1k-val":
        load_root = os.path.join(args.load_root, "cls_in1kval")
        if os.path.isdir(load_root):
            tqdm.write(f"Load features from {load_root}")
        else:
            load_root = args.load_root
            tqdm.write(f"Load features from {load_root}")
        evaluator = IN1K_Cls_Evaluator(data_root=args.data_root, load_root=load_root, skip_layer=args.skip_layer)
        evaluator.evaluate_on_imagenet1k_val()

    elif args.dataset == "imagenet1k-adv":
        load_root = os.path.join(args.load_root, "cls_in1ka")
        if os.path.isdir(load_root):
            tqdm.write(f"Load features from {load_root}")
        else:
            load_root = args.load_root
            tqdm.write(f"Load features from {load_root}")
        evaluator = IN1K_Cls_Evaluator(data_root=args.data_root, load_root=load_root, skip_layer=args.skip_layer)
        evaluator.evaluate_on_imagenet1k_adv()

    elif args.dataset == "imagenet1k-ren":
        load_root = os.path.join(args.load_root, "cls_in1kr")
        if os.path.isdir(load_root):
            tqdm.write(f"Load features from {load_root}")
        else:
            load_root = args.load_root
            tqdm.write(f"Load features from {load_root}")
        evaluator = IN1K_Cls_Evaluator(data_root=args.data_root, load_root=load_root, skip_layer=args.skip_layer)
        evaluator.evaluate_on_imagenet1k_ren()
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    # 恢复 stdout 并关闭日志文件
    tqdm.write(f"\n日志已保存至: {log_filepath}")
    sys.stdout = original_stdout
    tee_logger.close()


if __name__ == "__main__":
    main()
