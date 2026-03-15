"""
FalconMamba Feature Coding Evaluation Script.

该脚本仅评估被codec重构后的特征自身对原始模型下游任务上的性能。
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["GIT_DISCOVERY_ACROSS_FILESYSTEM"] = "1"  # 消除lm-eval的git警告

import lm_eval
import numpy as np
from lm_eval.api.registry import register_model
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import simple_parse_args_string
from tabulate import SEPARATING_LINE, tabulate
from tqdm import tqdm
from transformers.models.falcon_mamba.modeling_falcon_mamba import FalconMambaCache

sys.path.insert(0, "../../lmfc_utils")
from lmfc_utils.handlers.utils import compute_mse, inspect_structure, load_tensor_using_ref, load_zst_tensor

TASK_NAMES = ["fc_arc_challenge", "fc_hellaswag", "fc_truthfulqa_mc1", "fc_winogrande", "fc_gsm8k"]
BASE_BITS = 32  # Base bits for RBPFP calculation (normalized to 32-bit floating point)


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


def check_sample_correctness(sample_metrics):
    """判断单个样本是否正确。
    lm-eval 的 metrics key 可能是 'acc', 'exact_match,flexible-extract' 等。
    这里按优先级查找。
    """
    # 优先级列表：越靠前越优先采用
    target_keys = ["exact_match,flexible-extract", "exact_match,get_answer"]

    for key in target_keys:
        if key in sample_metrics:
            val = sample_metrics[key]
            # 处理 numpy 类型或普通 float
            if isinstance(val, (np.floating, float, int)):
                return float(val) == 1.0
            return False

    return False


def print_aggregated_results(results):
    """核心统计函数。
    Args:
        results: 字典结构 results[set_name][sample_id] = {metrics...}
    """

    # 基础统计：计算每个 Task 的 (Correct, Total)
    group_stats = {}  # {task_name: {'correct': 0, 'total': 0}}
    for set_name, samples in results.items():
        correct_count = 0
        total_count = 0
        for sample_id, metrics in samples.items():
            total_count += 1
            if check_sample_correctness(metrics[set_name]):
                correct_count += 1
        group_stats[set_name] = {"correct": correct_count, "total": total_count}

    # 打印所有任务
    msg = []
    total_correct = 0
    total_samples = 0
    for task_name, stats in sorted(group_stats.items()):
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        msg.append([f"{task_name:<35}", f"{acc:6.2f}%", f"{stats['correct']}/{stats['total']}"])
        total_correct += stats["correct"]
        total_samples += stats["total"]
    assert total_samples > 0, "No samples found"
    overall_acc = (total_correct / total_samples) * 100
    msg.append(["OVERALL", f"{overall_acc:.02f}%", f"{total_correct}/{total_samples}"])
    tqdm.write(tabulate(msg, headers=["BENCHMARKS", "ACC", "SAMPLES"]))


@register_model("fc_hooked_hf")
class HookedHFLM(HFLM):
    SKIPPED_FEATURES = [
        # ".blocks.39.cls_token",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.model.config.name_or_path.lower()
        print(self.model)

        tqdm.write(f"{self.model_name} is loaded. Registering the hook function...")
        # 从layer4获取layer4输出的特征和所有0-4层的kvcache
        self.hook_handles = self.register_forward_hooks(["backbone.layers.4"])

        # Initialize feature coding statistics structure as a nested dict.
        self.sample_fc_stats = defaultdict(list)

    def set_sample_info(
        self,
        load_root: str = "",
        group_name: str = "",
        sample_id: str = "",
        current_idx: int = 0,
        total_samples: int = 0,
    ):
        """Set data info for current sample."""
        self.group_name = group_name
        self.sample_id = sample_id

        if load_root:
            load_root = os.path.join(load_root, self.group_name)
            assert os.path.isdir(load_root), load_root
        self.load_root = load_root

        # 显示进度信息
        tqdm.write(
            tabulate(
                [[f"[{current_idx}/{total_samples}]", group_name, sample_id, load_root]],
                headers=["Progress", "Group", "Sample", "Load Root"],
            )
        )

    def get_hook(self, name):
        current_layer_idx = int(name.split(".")[-1])

        def hook(module, inputs, kwargs, outputs):
            skip_feature_coding = False
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            shape = [i.shape for i in inputs]
            if inputs[0].shape[1] == 1:
                skip_feature_coding = True

            if not skip_feature_coding:
                tqdm.write(f"-->>\t{name} w/ Inputs {shape} for Sample {self.sample_id} from {self.group_name} <<--")
                cache: FalconMambaCache = kwargs["cache_params"]

                sample_name = f"sample{self.sample_id}-layer{current_layer_idx}-item1.zst"
                inspect_structure(outputs, prefix="Output", print_fn=tqdm.write)
                for layer_id in range(current_layer_idx + 1):  # ..., current_layer_idx
                    inspect_structure(cache.ssm_states[layer_id], prefix=f"SSM-{layer_id}", print_fn=tqdm.write)
                    inspect_structure(cache.conv_states[layer_id], prefix=f"Conv-{layer_id}", print_fn=tqdm.write)
                tqdm.write("-" * 30)

                # Feature coding: encode and decode in memory
                if self.load_root:
                    tqdm.write(f"[Feature Coding] Replace features for layer {name}...")

                    load_path = os.path.join(self.load_root, sample_name)
                    assert os.path.exists(load_path), load_path
                    reconstructed_data = load_zst_tensor(load_path)

                    if "metadata" in reconstructed_data:
                        target_metadata = {**reconstructed_data["metadata"]}
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
                            for k, v in target_metadata.items()
                        ]
                        meta_msg = tabulate(meta_msg, headers=["Key", "Value"])
                        tqdm.write(f"Loaded features from {load_path} with meta information:\n{meta_msg}")

                        # features from reconstructed data
                        target_features = {**reconstructed_data["features"]}
                    else:
                        target_metadata = {}
                        target_features = {**reconstructed_data}

                    total_mse_sum = 0
                    total_numel = 0

                    mse_msg = []
                    loaded_o = target_features["output"]
                    loaded_o = load_tensor_using_ref(loaded_o, ref=outputs)
                    _mse_sum = compute_mse(loaded_o, outputs)
                    total_mse_sum += _mse_sum
                    _numel = loaded_o.numel()
                    total_numel += _numel
                    mse_msg.append([f"layer{layer_id}.output", f"{_mse_sum / _numel:.8f}"])
                    outputs.copy_(loaded_o)  # update

                    for layer_id, (loaded_ssm, loaded_conv) in enumerate(
                        zip(target_features["ssm_state"], target_features["conv_state"])
                    ):
                        raw_ssm_states = cache.ssm_states[layer_id]
                        loaded_ssm = load_tensor_using_ref(loaded_ssm, ref=raw_ssm_states)
                        _mse_sum = compute_mse(loaded_ssm, raw_ssm_states)
                        total_mse_sum += _mse_sum
                        _numel = loaded_ssm.numel()
                        total_numel += _numel
                        mse_msg.append([f"layer{layer_id}.key", f"{_mse_sum / _numel:.8f}"])

                        raw_conv_states = cache.conv_states[layer_id]
                        loaded_conv = load_tensor_using_ref(loaded_conv, ref=raw_conv_states)
                        _mse_sum = compute_mse(loaded_conv, raw_conv_states)
                        total_mse_sum += _mse_sum
                        _numel = loaded_conv.numel()
                        total_numel += _numel
                        mse_msg.append([f"layer{layer_id}.value", f"{_mse_sum / _numel:.8f}"])

                        cache.ssm_states[layer_id].copy_(loaded_ssm)
                        cache.conv_states[layer_id].copy_(loaded_conv)

                    mse_msg.append(["total mse", f"{total_mse_sum / total_numel:.8f}"])
                    mse_msg = tabulate(mse_msg, headers=["Feature Name/ID", "MSE"])
                    tqdm.write(f"[Feature Coding] Done. Injected decoded features with MSE:\n{mse_msg}")

                    # Append per-sample stats
                    self.sample_fc_stats[self.group_name].append(
                        {
                            "sample_id": self.sample_id,
                            "group_name": self.group_name,
                            "elements": total_numel,
                            "mse_recalc": total_mse_sum / total_numel,
                            # information from target_metadata
                            "arch": target_metadata.get("arch", None),
                            "handler": target_metadata.get("handler", None),
                            "strategy": target_metadata.get("strategy", None),
                            "transform_type": target_metadata.get("transform_type", None),
                            "bit_depth": target_metadata.get("bit_depth", None),
                            "bpfp": target_metadata.get("bpfp", -1),
                            "ebpfp": target_metadata.get("ebpfp", -1),
                            "mse": target_metadata.get("mse", -1),
                        }
                    )
            return outputs

        return hook

    def register_forward_hooks(self, layer_names: list[str]):
        hooks = []
        for name in layer_names:
            module = self.model.get_submodule(name)
            hooks.append(module.register_forward_hook(self.get_hook(name), with_kwargs=True))
        return hooks

    def remove_hooks(self):
        """Remove all registered forward hooks to avoid memory leaks."""
        num_hooks = len(self.hook_handles)
        for hook in self.hook_handles:
            hook.remove()
        self.hook_handles.clear()
        tqdm.write(f"Removed {num_hooks} hooks!")

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
    tqdm.write("开始进行评估...")
    MODEL_CONFIGS = dict(
        falconmamba=dict(
            model="fc_hooked_hf",
            model_args="pretrained=tiiuae/falcon-mamba-7b-instruct,max_length=8192",
            gen_kwargs={"temperature": 0},
            apply_chat_template=True,
            selected_samples="falconmamba.json",
        ),
    )

    BASE_CONFIG = dict(
        num_fewshot=0,
        batch_size=1,
        log_samples=False,
        limit=None,
        #
        max_batch_size=None,
        device=None,
        use_cache=None,
        check_integrity=False,
        write_out=False,
        show_config=False,
        system_instruction=None,
        predict_only=False,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
        confirm_run_unsafe_code=False,
        metadata=None,
    )

    # fmt: off
    parser = argparse.ArgumentParser(description="Feature coding for FalconMamba")
    parser.add_argument("--models", nargs="+", choices=MODEL_CONFIGS.keys(), default=['falconmamba'])
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES, choices=TASK_NAMES)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--load_root", type=str, default="", help="Path for saving features")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()
    # fmt: on

    os.makedirs(args.output, exist_ok=True)

    # 设置日志记录
    log_filename = f"falconmamba_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = Path(args.output) / log_filename
    tee_logger = TeeLogger(str(log_filepath))
    original_stdout = sys.stdout
    sys.stdout = tee_logger
    tqdm.write(f"日志将写入: {log_filepath}")

    # 初始化 TaskManager，加载自定义任务目录
    custom_tasks_path = Path(__file__).parent / "custom_tasks"
    task_manager = TaskManager(include_path=str(custom_tasks_path))
    tqdm.write(f"Loaded custom tasks from: {custom_tasks_path}")

    for model_name in args.models:
        model_cfg = MODEL_CONFIGS[model_name]
        tqdm.write(f"Evaluating {model_name} ({model_cfg})...")

        output_path = Path(args.output) / f"{model_name}.json"
        hf_hub_log_args = f"output_path={output_path}"
        if os.environ.get("HF_TOKEN", None):
            hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
        evaluation_tracker = EvaluationTracker(**simple_parse_args_string(hf_hub_log_args))

        selected_samples = None
        if "selected_samples" in model_cfg:
            sample_file = Path(__file__).parent / model_cfg["selected_samples"]
            if sample_file.is_file():
                with open(sample_file, mode="r", encoding="utf-8") as f:
                    selected_samples = json.load(f)
            else:
                tqdm.write(f"Warning: selected_samples file not found: {sample_file}")
        if selected_samples is None:
            tqdm.write("No selected samples found. Skipping evaluation.")
            continue

        # Create model
        model: HookedHFLM = lm_eval.api.registry.get_model(model_cfg["model"]).create_from_arg_string(
            model_cfg["model_args"],
            dict(
                batch_size=BASE_CONFIG["batch_size"],
                max_batch_size=BASE_CONFIG["max_batch_size"],
                device=BASE_CONFIG["device"],
            ),
        )

        # 计算总样本数用于显示进度
        total_samples = sum(
            len(sample_ids)
            for task_name in args.tasks
            for set_name, sample_ids in selected_samples.items()
            if set_name.startswith(task_name)
        )

        current_idx = 0
        results = {}
        for task_name in args.tasks:
            t_task_start = time.perf_counter()

            sub_samples = {k: v for k, v in selected_samples.items() if k.startswith(task_name)}
            for set_name, sample_ids in sub_samples.items():
                results[set_name] = {}

                if args.limit is not None and args.limit > 0:
                    sample_ids = sample_ids[: args.limit]
                for sample_id in sample_ids:
                    t_sample_start = time.perf_counter()

                    current_idx += 1
                    model.set_sample_info(
                        load_root=args.load_root,
                        group_name=set_name,
                        sample_id=sample_id,
                        current_idx=current_idx,
                        total_samples=total_samples,
                    )

                    samplewise_results = lm_eval.simple_evaluate(
                        model=model,
                        model_args=model_cfg["model_args"],
                        gen_kwargs=model_cfg.get("gen_kwargs", None),
                        apply_chat_template=model_cfg.get("apply_chat_template", False),
                        fewshot_as_multiturn=model_cfg.get("fewshot_as_multiturn", False),
                        #
                        task_manager=task_manager,
                        tasks=set_name,
                        samples={set_name: [sample_id]},
                        num_fewshot=BASE_CONFIG["num_fewshot"],
                        batch_size=BASE_CONFIG["batch_size"],
                        max_batch_size=BASE_CONFIG["max_batch_size"],
                        device=BASE_CONFIG["device"],
                        use_cache=BASE_CONFIG["use_cache"],
                        limit=BASE_CONFIG["limit"],
                        check_integrity=BASE_CONFIG["check_integrity"],
                        write_out=BASE_CONFIG["write_out"],
                        log_samples=BASE_CONFIG["log_samples"],
                        system_instruction=BASE_CONFIG["system_instruction"],
                        predict_only=BASE_CONFIG["predict_only"],
                        random_seed=BASE_CONFIG["random_seed"],
                        numpy_random_seed=BASE_CONFIG["numpy_random_seed"],
                        torch_random_seed=BASE_CONFIG["torch_random_seed"],
                        fewshot_random_seed=BASE_CONFIG["fewshot_random_seed"],
                        confirm_run_unsafe_code=BASE_CONFIG["confirm_run_unsafe_code"],
                        metadata=BASE_CONFIG["metadata"],
                        #
                        evaluation_tracker=evaluation_tracker,
                        **request_caching_arg_to_dict(cache_requests="refresh"),
                    )
                    results[set_name][sample_id] = samplewise_results["results"]

                    t_sample_total = time.perf_counter() - t_sample_start
                    tqdm.write(tabulate([[str(sample_id), f"{t_sample_total:.06f}s"]], headers=["Sample", "Time"]))

            t_task_total = time.perf_counter() - t_task_start
            tqdm.write(tabulate([[str(task_name), f"{t_task_total:.06f}s"]], headers=["Task", "Time"]))

        # Print feature coding stats
        model.print_feature_coding_stats()

        # Remove hooks to avoid memory leaks
        model.remove_hooks()
        print_aggregated_results(results)

    # 恢复 stdout 并关闭日志文件
    tqdm.write(f"\n日志已保存至: {log_filepath}")
    sys.stdout = original_stdout
    tee_logger.close()


if __name__ == "__main__":
    main()
