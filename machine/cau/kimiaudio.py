"""
KimiAudio Feature Coding Evaluation Script.

该脚本仅评估被codec重构后的特征自身对原始模型下游任务上的性能。
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import torch
from almeval.datasets import DATASETS
from almeval.utils import dump
from tabulate import SEPARATING_LINE, tabulate
from tqdm import tqdm

sys.path.insert(0, "../../lmfc_utils")
from handlers.utils import compute_mse, inspect_structure, load_tensor_using_ref, load_zst_tensor

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, "almeval/models/kimi_audio")  # noqa
from kimia_infer.api.kimia import KimiAudio

# =============================================================================
seed = 20260123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 固定 cuDNN 行为（你已经在用）
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# 强烈建议：关闭 TF32（对 learned compression 很关键）
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# 可选：更“狠”的确定性（可能会变慢，或遇到不支持的算子直接报错）
# torch.use_deterministic_algorithms(True)
# 可选：让 cuBLAS 更确定性（有时 GEMM 相关会需要）
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# =============================================================================


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


def to_export_recursive(obj, to_numpy=False):
    # 1. 处理 Tensor：关键步骤 detach -> cpu -> numpy
    if isinstance(obj, torch.Tensor):
        # detach() 是必须的，如果 tensor 带有梯度，不 detach 无法转 numpy
        if to_numpy:
            return obj.detach().cpu().numpy()
        return obj.detach().cpu()

    # 2. 处理字典：递归处理 value，保持 key 不变
    elif isinstance(obj, dict):
        return {k: to_export_recursive(v, to_numpy=to_numpy) for k, v in obj.items()}

    # 3. 处理列表：递归处理每个元素
    elif isinstance(obj, list):
        return [to_export_recursive(v, to_numpy=to_numpy) for v in obj]

    # 4. 处理元组：递归处理后需要重新转回 tuple（因为 tuple 不可变）
    elif isinstance(obj, tuple):
        return tuple(to_export_recursive(v, to_numpy=to_numpy) for v in obj)

    # 5. 其他类型（int, float, str, None）：直接返回，保持原样
    else:
        return obj


def print_performance_table(results_dict: Dict[str, Any]):
    """
    提取并美观地打印出评估结果字典中不同数据集的性能指标。

    参数:
        results_dict: 包含评估结果的字典，键是数据集名称，值包含 'performance' 键。
    """
    if not results_dict:
        tqdm.write("评估结果字典为空。")
        return

    # 1. 准备数据行
    table_data = []

    for dataset_name, data in results_dict.items():
        task = data.get("task", "N/A")
        performance = data.get("performance", {})
        eval_method = data.get("eval_method", "N/A")

        if not performance:
            # 记录没有性能数据的任务
            table_data.append([dataset_name, task, eval_method, "N/A", "N/A"])
            continue

        # performance 键下的值通常是一个包含子数据集/子任务的字典
        for sub_task_name, metrics in performance.items():
            # 提取主要性能指标 (acc, wer, etc.)
            metric_key = None
            metric_value = None

            # 优先查找常见的性能指标，如 acc, wer, f1 等
            for key, value in metrics.items():
                if key in ["acc", "accuracy", "wer", "cer", "f1", "ppl"]:
                    metric_key = key.upper()
                    if isinstance(value, float):
                        metric_value = f"{value:.2f}"
                    elif isinstance(value, int):
                        metric_value = str(value)
                    else:
                        metric_value = str(value)
                    break

            # 如果没有找到常见指标，打印第一个非 total/valid/correct 的指标
            if metric_key is None:
                for key, value in metrics.items():
                    if key not in ["total", "valid", "correct"]:
                        metric_key = key.upper()
                        if isinstance(value, float):
                            metric_value = f"{value:.4f}"
                        else:
                            metric_value = str(value)
                        break

            # 如果性能数据中包含了 total 或 valid 数量，可以一起显示
            total_samples = metrics.get("total", metrics.get("valid", "-"))

            # 将每一行数据添加到列表中
            table_data.append([dataset_name, task, eval_method, metric_key, metric_value, total_samples])

    # 2. 打印 Markdown 格式的表格
    headers = ["Dataset", "Task", "EvalMethod", "Metric", "Result", "(Total/Valid)"]
    col_widths = [len(h) for h in headers]  # 初始宽度

    # 计算列的最大宽度以保证对齐
    for row in table_data:
        for i, item in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(item)))

    # 格式化函数，居中对齐
    def format_cell(text, width):
        return str(text).ljust(width)

    # 打印表头
    header_line = "| " + " | ".join(format_cell(h, col_widths[i]) for i, h in enumerate(headers)) + " |"
    tqdm.write(header_line)

    # 打印分隔线
    separator_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    tqdm.write(separator_line)

    # 打印数据行
    for row in table_data:
        data_line = "| " + " | ".join(format_cell(row[i], col_widths[i]) for i in range(len(headers))) + " |"
        tqdm.write(data_line)


class FeatureCodingWrapper:
    NAME = "Kimi-Audio-7B-Instruct"
    SKIPPED_FEATURES = [
        # "model.layers.0.k_cache",
        # "model.layers.0.v_cache",
        # "model.layers.1.k_cache",
        # "model.layers.1.v_cache",
        # "model.layers.2.k_cache",
        # "model.layers.2.v_cache",
        # "model.layers.3.k_cache",
        # "model.layers.3.v_cache",
        # "model.layers.4.k_cache",
        # "model.layers.4.v_cache",
        # "model.layers.4.output",
    ]

    def __init__(self, load_root=None):
        self.model = KimiAudio(model_path="moonshotai/Kimi-Audio-7B-Instruct", load_detokenizer=False)
        tqdm.write(str(self.model.alm))

        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.1,
            "text_repetition_window_size": 16,
            "max_new_tokens": 1024,  # limit the max_new_tokens to avoid the model from generating too long responses
        }

        target_layer_names = {
            "model.layers.0": True,
            "model.layers.1": True,
            "model.layers.2": True,
            "model.layers.3": True,
            "model.layers.4": False,
        }
        self.load_root = load_root
        self.sample_fc_stats = defaultdict(list)
        self.hook_handles = self.register_forward_hooks(target_layer_names)

    def get_group_name(self, sample_name: str):
        if sample_name.startswith("LibriSpeech"):
            if "clean" in sample_name:
                return "librispeech-test-clean"
            else:
                return "librispeech-test-other"
        elif sample_name.startswith("AdvBench"):
            return "advbench"
        elif sample_name.startswith("OpenBookQA"):
            return "openbookqa"
        elif sample_name.startswith("SD-QA"):
            return "sd-qa"
        else:
            raise ValueError(f"Unknown sample name: {sample_name}")

    def load_feature(self, sample_name: str):
        self.group_name = self.get_group_name(sample_name)
        self.sample_name = sample_name

        load_path = os.path.join(self.load_root, self.group_name, f"{sample_name}.zst")
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

            self.current_target_features = {**reconstructed_data["features"]}
        else:
            self.current_target_features = {**reconstructed_data}
            self.current_target_metadata = {}

        tqdm.write(f"Loaded features {self.current_target_features.keys()}.")
        inspect_structure(self.current_target_features)

    def get_hook(self, name, only_kvcache):
        def hook(module, inputs, outputs):
            skip_feature_coding = False
            if isinstance(inputs, (list, tuple)):
                shape = [i.shape for i in inputs]
                if inputs[0].shape[1] == 1:
                    skip_feature_coding = True
            else:
                shape = inputs.shape
                if inputs.shape[1] == 1:
                    skip_feature_coding = True

            if not skip_feature_coding:
                tqdm.write(f"-->>\tInspecting Layer: {name} w/ Inputs Tensor {shape} (Prefill) ---")
                inspect_structure(outputs, prefix="Output", print_fn=tqdm.write)

                # 这里可以基于实际的输入形状来判断是否处于prefill阶段
                if self.load_root:
                    assert self.current_target_features, "No features to load!"

                    if only_kvcache:
                        feature_key = f"{name}.k_cache"
                        if feature_key in self.SKIPPED_FEATURES:
                            tqdm.write(f"[Feature Coding] Skip {feature_key}...")
                        else:
                            tqdm.write(f"[Feature Coding] Replace {feature_key}...")
                            _loaded_k = self.current_target_features[name][0]
                            _loaded_k = load_tensor_using_ref(_loaded_k, ref=outputs[1][0])
                            _mse_sum = compute_mse(_loaded_k, outputs[1][0])
                            self.total_mse_sum += _mse_sum
                            _numel = _loaded_k.numel()
                            self.total_numel += _numel
                            self.sample_mse_msg.append([feature_key, f"{_mse_sum / _numel:.8f}"])

                            outputs[1][0].zero_()
                            outputs[1][0].copy_(_loaded_k)

                        feature_key = f"{name}.v_cache"
                        if feature_key in self.SKIPPED_FEATURES:
                            tqdm.write(f"[Feature Coding] Skip {feature_key}...")
                        else:
                            tqdm.write(f"[Feature Coding] Replace {feature_key}...")
                            _loaded_v = self.current_target_features[name][1]
                            _loaded_v = load_tensor_using_ref(_loaded_v, ref=outputs[1][1])
                            _mse_sum = compute_mse(_loaded_v, outputs[1][1])
                            self.total_mse_sum += _mse_sum
                            _numel = _loaded_v.numel()
                            self.total_numel += _numel
                            self.sample_mse_msg.append([feature_key, f"{_mse_sum / _numel:.8f}"])

                            outputs[1][1].zero_()
                            outputs[1][1].copy_(_loaded_v)

                        # outputs[0].zero_() # 会干扰后续计算，导致无法检查后续层的output、key和value的误差
                    else:
                        feature_key = f"{name}.k_cache"
                        if feature_key in self.SKIPPED_FEATURES:
                            tqdm.write(f"[Feature Coding] Skip {feature_key}...")
                        else:
                            tqdm.write(f"[Feature Coding] Replace {feature_key}...")
                            _loaded_k = self.current_target_features[name][1][0]
                            _loaded_k = load_tensor_using_ref(_loaded_k, ref=outputs[1][0])
                            _mse_sum = compute_mse(_loaded_k, outputs[1][0])
                            self.total_mse_sum += _mse_sum
                            _numel = _loaded_k.numel()
                            self.total_numel += _numel
                            self.sample_mse_msg.append([feature_key, f"{_mse_sum / _numel:.8f}"])

                            outputs[1][0].zero_()
                            outputs[1][0].copy_(_loaded_k)

                        feature_key = f"{name}.v_cache"
                        if feature_key in self.SKIPPED_FEATURES:
                            tqdm.write(f"[Feature Coding] Skip {feature_key}...")
                        else:
                            tqdm.write(f"[Feature Coding] Replace {feature_key}...")
                            _loaded_v = self.current_target_features[name][1][1]
                            _loaded_v = load_tensor_using_ref(_loaded_v, ref=outputs[1][1])
                            _mse_sum = compute_mse(_loaded_v, outputs[1][1])
                            self.total_mse_sum += _mse_sum
                            _numel = _loaded_v.numel()
                            self.total_numel += _numel
                            self.sample_mse_msg.append([feature_key, f"{_mse_sum / _numel:.8f}"])

                            outputs[1][1].zero_()
                            outputs[1][1].copy_(_loaded_v)

                        feature_key = f"{name}.output"
                        if feature_key in self.SKIPPED_FEATURES:
                            tqdm.write(f"[Feature Coding] Skip {feature_key}...")
                        else:
                            tqdm.write(f"[Feature Coding] Replace {feature_key}...")
                            _loaded_o = self.current_target_features[name][0]
                            _loaded_o = load_tensor_using_ref(_loaded_o, ref=outputs[0])
                            _mse_sum = compute_mse(_loaded_o, outputs[0])
                            self.total_mse_sum += _mse_sum
                            _numel = _loaded_o.numel()
                            self.total_numel += _numel
                            self.sample_mse_msg.append([feature_key, f"{_mse_sum / _numel:.8f}"])

                            outputs[0].zero_()
                            outputs[0].copy_(_loaded_o)
            return outputs

        return hook

    def register_forward_hooks(self, layer_names: dict):
        hooks = []
        for name, only_kvcache in layer_names.items():
            module = self.model.alm.get_submodule(name)
            hooks.append(module.register_forward_hook(self.get_hook(name, only_kvcache=only_kvcache)))
        return hooks

    def remove_hooks(self):
        """Remove all registered forward hooks to avoid memory leaks."""
        num_hooks = len(self.hook_handles)
        for hook in self.hook_handles:
            hook.remove()
        self.hook_handles.clear()
        tqdm.write(f"Removed {num_hooks} hooks!")

    @staticmethod
    def check_audio_legal(audio_path: str | list[str], max_duration: float = 60) -> bool:
        """by default, we discard audio longer than 60s. subclasses can override this method (depends on model requirements)"""
        if isinstance(audio_path, str):
            duration = librosa.get_duration(path=audio_path)
            if duration > max_duration or duration < 0.1:
                return False
        else:
            for path in audio_path:
                duration = librosa.get_duration(path=path)
                if duration > max_duration or duration < 0.1:
                    return False
        return True

    @torch.inference_mode()
    def __call__(self, msg: dict) -> str:
        self.sample_mse_msg = []
        self.total_mse_sum = 0
        self.total_numel = 0

        if not self.check_audio_legal(msg["audio"]):
            tqdm.write(
                f"dataset: {msg['meta']['dataset_name']}, audio: {msg['audio']}, duration exceeds 60s limit, skipping this sample"
            )
            return msg["text"], None

        real_prompt, response = self.generate_inner(msg)

        if self.load_root:
            mse_msg = tabulate(self.sample_mse_msg, headers=["Tensor", "MSE"])
            tqdm.write(f"[Feature Coding] Done. Injected decoded features with MSE:\n{mse_msg}")
            # Append per-sample stats
            self.sample_fc_stats[self.group_name].append(
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
        return real_prompt, response

    def get_prompt(self, msg: dict) -> str:
        return msg["text"]

    def generate_inner(self, msg: dict):
        audio = msg["audio"]
        if isinstance(audio, list) and len(audio) == 1:
            audio = audio[0]
        else:
            raise NotImplementedError(f"Audio {type(audio)} with length {len(audio)} not supported")

        prompt: str = self.get_prompt(msg)

        messages = []
        if prompt is not None and prompt.strip() != "":
            messages.append({"role": "user", "message_type": "text", "content": prompt})
        messages.append({"role": "user", "message_type": "audio", "content": audio})
        _, text = self.model.generate(messages, **self.sampling_params, output_type="text")
        return prompt, text

    def print_feature_coding_stats(self):
        """Print accumulated feature coding statistics per group."""
        if not self.sample_fc_stats:
            return

        # Print stats for each group separately
        for group_name, stats in sorted(self.sample_fc_stats.items()):
            num_samples = len(stats)
            if num_samples == 0:
                tqdm.write(f"No feature coding stats for group: {group_name}")
                continue

            # Compute averages from totals
            total_elements = sum(s["elements"] for s in stats)
            per_sample_avg_bpfp = sum(s["bpfp"] for s in stats) / num_samples
            per_sample_avg_ebpfp = sum(s["ebpfp"] for s in stats) / num_samples
            per_sample_avg_mse = sum(s["mse"] for s in stats) / num_samples
            per_sample_avg_mse_recalc = sum(s["mse_recalc"] for s in stats) / num_samples

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
                    headers=[f"FEATURE CODING STATISTICS: {group_name}", ""],
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_info", type=str, default="kimiaudio7binstruct_audio_500samples/data.json")
    parser.add_argument("--output", type=str, default="outputs-2026-01-28")
    parser.add_argument("--load_root", type=str, help="Path for saving features")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    log_filename = f"kimiaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = Path(args.output) / log_filename
    tee_logger = TeeLogger(str(log_filepath))
    original_stdout = sys.stdout
    sys.stdout = tee_logger
    tqdm.write(f"日志将写入: {log_filepath}")

    with open(args.data_info, mode="r", encoding="utf-8") as f:
        raw_dataset_infos = json.load(f)

    model = FeatureCodingWrapper(load_root=args.load_root)
    tqdm.write(f"Evaluating the model {model.NAME}...")

    total_results = {}
    for dataset_name, dataset_class in DATASETS.items():
        data_infos = raw_dataset_infos[dataset_name]
        if args.limit > 0:
            data_infos = data_infos[: args.limit]

        for data_info in data_infos:
            data_info["audio_path"] = os.path.abspath(data_info["new_audio_path"])
        dataset = dataset_class(data_info=data_infos)
        tqdm.write(f"Collecting information from the dataset: {dataset.DATASET_NAME}")

        results = {}
        t_dataset_start = time.perf_counter()
        for msg in tqdm(dataset, total=len(dataset), ncols=78):
            idx = int(msg["index"])
            t_sample_start = time.perf_counter()

            if model.load_root:
                model.load_feature(sample_name=f"{dataset_name}-{msg['meta']['subset']}-{msg['index']}")

            real_prompt, response = model(msg)
            if response is None:
                tqdm.write("response is None!!!")
                continue

            t_sample_end = time.perf_counter()
            tqdm.write(f"Sample {idx} took {(t_sample_end - t_sample_start):.2f} seconds")

            results[idx] = {"prompt": real_prompt, "prediction": response}

        t_dataset_end = time.perf_counter()
        tqdm.write(f"Dataset {dataset_name} took {(t_dataset_end - t_dataset_start):.2f} seconds")

        raw_data = dataset.data
        for x in raw_data:
            idx = int(x["index"])
            if idx not in results:
                tqdm.write(f"index {idx} not found in results, details: {x}")
                x["prediction"] = "null"
                x["real_prompt"] = ""
                continue
            x["prediction"] = str(results[idx]["prediction"])
            x["real_prompt"] = str(results[idx]["prompt"])

        results_tmpfile = os.path.join(args.output, f"{dataset.DATASET_NAME}.jsonl")
        dump(raw_data, results_tmpfile)

        tqdm.write(f"evaluating for {dataset.DATASET_NAME}...")
        perf = dataset.evaluate(results_tmpfile)
        total_results[dataset_name] = perf

    # Print feature coding stats
    model.print_feature_coding_stats()

    # Remove hooks to avoid memory leaks
    model.remove_hooks()

    print_performance_table(total_results)

    # 恢复 stdout 并关闭日志文件
    tqdm.write(f"\n日志已保存至: {log_filepath}")
    sys.stdout = original_stdout
    tee_logger.close()


if __name__ == "__main__":
    main()
