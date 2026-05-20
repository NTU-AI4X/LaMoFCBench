import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from PIL import Image
from tabulate import SEPARATING_LINE, tabulate
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from transformers.models.t5.modeling_t5 import BaseModelOutputWithPastAndCrossAttentions

sys.path.insert(0, "../../lmfc_utils")
from lmfc_utils.handlers.utils import compute_mse, inspect_structure, load_tensor_using_ref, load_zst_tensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HEIGHT = 1024
WIDTH = 1024


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


def tti_evaluate_fid_wo_ref(ori_image_root, rec_image_root):
    ori_image_root = Path(ori_image_root)
    assert ori_image_root.is_dir()
    ori_image_names = set([p.name for p in ori_image_root.iterdir() if p.is_file()])

    rec_image_root = Path(rec_image_root)
    assert rec_image_root.is_dir()
    rec_image_names = set([p.name for p in rec_image_root.iterdir() if p.is_file()])

    shared_image_names = sorted(ori_image_names.intersection(rec_image_names))

    ori_images = []
    rec_images = []
    for image_name in shared_image_names:
        ori_image_path = (ori_image_root / image_name).with_suffix(".png")
        if not ori_image_path.exists():
            ori_image_path = ori_image_path.with_suffix(".jpg")
        ori_image = np.asarray(Image.open(ori_image_path).convert("RGB").resize((299, 299)))
        ori_images.append(ori_image)

        rec_image_path = (rec_image_root / image_name).with_suffix(".png")
        if not rec_image_path.exists():
            rec_image_path = rec_image_path.with_suffix(".jpg")
        rec_image = np.asarray(Image.open(rec_image_path).convert("RGB").resize((299, 299)))
        rec_images.append(rec_image)
    fid_results = {"num_ori_images": len(ori_images), "num_rec_images": len(rec_images)}

    ori_images = torch.from_numpy(np.asarray(ori_images)).permute(0, 3, 1, 2)
    rec_images = torch.from_numpy(np.asarray(rec_images)).permute(0, 3, 1, 2)

    fid = FrechetInceptionDistance(
        feature=2048, reset_real_features=True, normalize=False, input_img_size=(3, 299, 299)
    )
    fid.update(ori_images, real=True)
    fid.update(rec_images, real=False)
    fid_score = fid.compute()
    fid_results["fid_score"] = fid_score.item()
    return fid_results


def set_seed(seed=42):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # =============================================================================
    # # 固定 cuDNN 行为（你已经在用）
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # # 强烈建议：关闭 TF32（尤其是在使用 Ampere 及更新架构的 GPU 时）
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    # 可选：更“狠”的确定性（可能会变慢，或遇到不支持的算子直接报错）
    # torch.use_deterministic_algorithms(True)
    # 可选：让 cuBLAS 更确定性（有时 GEMM 相关会需要）
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # =============================================================================


def get_data_infos(data_json: str, image_root: str):
    with open(data_json, "r", encoding="utf-8") as f:
        raw_data_infos = json.load(f)
    data_infos = {}
    for raw_data_info in raw_data_infos:
        image_name = raw_data_info["image_name"]
        image_caption = raw_data_info["caption"]
        data_infos[image_name] = (Path(image_root).joinpath(image_name).as_posix(), image_caption)
    return data_infos


class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)

    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)  # -1~1
        image = image * 255 * 0.5 + 0.5
        return image

    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image


def to_export_recursive(obj, to_numpy=False):
    # 1. 处理 Tensor：关键步骤 detach -> cpu -> numpy
    if isinstance(obj, torch.Tensor):
        # detach() 是必须的，如果 tensor 带有梯度，不 detach 无法转 numpy
        if to_numpy:
            return obj.detach().cpu().numpy()
        return obj.detach().cpu()

    # 2. 处理字典：递归处理 value，保持 key 不变
    elif isinstance(obj, dict):
        return {k: to_export_recursive(v) for k, v in obj.items()}

    # 3. 处理列表：递归处理每个元素
    elif isinstance(obj, list):
        return [to_export_recursive(v) for v in obj]

    # 4. 处理元组：递归处理后需要重新转回 tuple（因为 tuple 不可变）
    elif isinstance(obj, tuple):
        return tuple(to_export_recursive(v) for v in obj)

    # 5. 其他类型（int, float, str, None）：直接返回，保持原样
    else:
        return obj


class SD3CtrlNetFeatureCodingPipeline:
    # https://huggingface.co/stabilityai/stable-diffusion-3.5-large-controlnet-canny
    def __init__(
        self, load_root="", load_condition=False, load_latent=False, *, torch_dtype=torch.float16, cpu_offload=True
    ):
        self.ctrlnet_name_or_path = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
        print(f"Loading the SD Model with ControlNet: {self.ctrlnet_name_or_path}........................")
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            controlnet=SD3ControlNetModel.from_pretrained(self.ctrlnet_name_or_path, torch_dtype=torch_dtype),
            torch_dtype=torch_dtype,
        )
        self.pipe.image_processor = SD3CannyImageProcessor()
        self.model_name = "sd3.5-l-controlnet-canny"

        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(DEVICE)

        self.device = self.pipe.device
        print(f"SD3Pipeline initialized on device: {self.device} with dtype: {torch_dtype}")

        self.task_name = "cond-tti"
        if load_root:
            assert os.path.isdir(load_root), load_root
            if os.path.exists(os.path.join(load_root, "sd35cond")):
                load_root = os.path.join(load_root, "sd35cond")
        self.load_root = load_root
        print(f"load_root={load_root}")

        self.target_layer_names = []
        if self.load_root:
            # vae.encoder 是为时序迭代中的controlnet条件编码做准备的
            if load_condition:
                self.target_layer_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "vae.encoder"]
            elif load_latent:
                self.target_layer_names = ["vae.decoder"]
            else:
                raise ValueError(
                    f"Invalid load config: load_root={load_root} (load_condition={load_condition}, load_latent={load_latent})"
                )
            self.hook_handles = self.register_forward_hooks(self.target_layer_names)
            self.prompt_item_index = 0
            self.sample_fc_stats = []

    def remove_hooks(self):
        """Remove all registered forward hooks to avoid memory leaks."""
        num_hooks = len(self.hook_handles)
        for hook in self.hook_handles:
            hook.remove()
        self.hook_handles.clear()
        tqdm.write(f"Removed {num_hooks} hooks!")

    def load_feature(self, sample_name: str):
        self.sample_name = sample_name
        self.sample_mse_msg = []
        self.total_mse_sum = 0
        self.total_numel = 0

        load_path = os.path.join(self.load_root, f"{Path(sample_name).stem}.zst")
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
            tqdm.write(f"Loaded features {self.current_target_features.keys()}.")
        else:
            self.current_target_features = {**reconstructed_data}
            self.current_target_metadata = {}

    def get_hook(self, name):
        # 需要考虑文本编码器对于正负prompt都有编码
        # CLIPTextModelOutput | BaseModelOutputWithPastAndCrossAttentions | torch.Tensor
        def hook(module, inputs, outputs):
            _name = f"{name}-item{self.prompt_item_index}"
            tqdm.write(f"-->> {_name} for {self.task_name} <<--")
            inspect_structure(outputs, prefix="Embeds", print_fn=tqdm.write)
            tqdm.write("-" * 30)

            if self.load_root:
                assert self.current_target_features, "No features to load!"
                if name in ("text_encoder", "text_encoder_2") and isinstance(outputs, CLIPTextModelOutput):
                    assert outputs.text_embeds.shape[0] == 1, outputs.text_embeds.shape[0]
                    tqdm.write(f"[Feature Coding] Replace {_name}.clip_pooled_prompt_embeds...")

                    _loaded = self.current_target_features[_name]["clip_pooled_prompt_embeds"]
                    _loaded = load_tensor_using_ref(_loaded, ref=outputs.text_embeds[0])
                    _mse_sum = compute_mse(_loaded, outputs.text_embeds[0])
                    self.total_mse_sum += _mse_sum
                    _numel = _loaded.numel()
                    self.total_numel += _numel
                    self.sample_mse_msg.append([f"{_name}.clip_pooled_prompt_embeds", f"{_mse_sum / _numel:.8f}"])

                    outputs.text_embeds.zero_()
                    outputs.text_embeds = torch.stack([_loaded], dim=0)

                    assert outputs.hidden_states[-2].shape[0] == 1, outputs.hidden_states[-2].shape[0]
                    tqdm.write(f"[Feature Coding] Replace {_name}.clip_prompt_embeds...")

                    _loaded = self.current_target_features[_name]["clip_prompt_embeds"]
                    _loaded = load_tensor_using_ref(_loaded, ref=outputs.hidden_states[-2][0])
                    _mse_sum = compute_mse(_loaded, outputs.hidden_states[-2][0])
                    self.total_mse_sum += _mse_sum
                    _numel = _loaded.numel()
                    self.total_numel += _numel
                    self.sample_mse_msg.append([f"{_name}.clip_prompt_embeds", f"{_mse_sum / _numel:.8f}"])

                    # SD3.5中，默认仅利用倒数第二层的隐藏状态，不过其他的也可以直接置零
                    outputs.hidden_states = [x.zero_() for x in outputs.hidden_states]
                    outputs.hidden_states[-2] = torch.stack([_loaded], dim=0)

                # T5 Text Encoder 的情况
                elif name == "text_encoder_3" and isinstance(outputs, BaseModelOutputWithPastAndCrossAttentions):
                    assert outputs.last_hidden_state.shape[0] == 1, outputs.last_hidden_state.shape[0]
                    tqdm.write(f"[Feature Coding] Replace {_name}.t5_prompt_embeds...")

                    _loaded = self.current_target_features[_name]["t5_prompt_embeds"]
                    _loaded = load_tensor_using_ref(_loaded, ref=outputs.last_hidden_state[0])
                    _mse_sum = compute_mse(_loaded, outputs.last_hidden_state[0])
                    self.total_mse_sum += _mse_sum
                    _numel = _loaded.numel()
                    self.total_numel += _numel
                    self.sample_mse_msg.append([f"{_name}.t5_prompt_embeds", f"{_mse_sum / _numel:.8f}"])

                    outputs.last_hidden_state.zero_()
                    outputs.last_hidden_state = torch.stack([_loaded], dim=0)  # 16,128,128

                # VAE Image Encoder
                elif name == "vae.encoder" and isinstance(outputs, torch.Tensor):
                    assert outputs.shape[0] == 2, outputs.shape[0]
                    if _name in self.current_target_features:
                        real_key = _name
                    else:
                        real_key = name

                    _loaded_f0 = self.current_target_features[real_key]["vae.encoder_f0"]
                    _loaded_f0 = load_tensor_using_ref(_loaded_f0, ref=outputs[0])
                    _mse_sum = compute_mse(_loaded_f0, outputs[0])
                    self.total_mse_sum += _mse_sum
                    _numel = _loaded_f0.numel()
                    self.total_numel += _numel
                    self.sample_mse_msg.append([f"{real_key}.vae.encoder_f0", f"{_mse_sum / _numel:.8f}"])

                    _loaded_f1 = self.current_target_features[real_key]["vae.encoder_f1"]
                    _loaded_f1 = load_tensor_using_ref(_loaded_f1, ref=outputs[1])
                    _mse_sum = compute_mse(_loaded_f1, outputs[1])
                    self.total_mse_sum += _mse_sum
                    _numel = _loaded_f1.numel()
                    self.total_numel += _numel
                    self.sample_mse_msg.append([f"{real_key}.vae.encoder_f1", f"{_mse_sum / _numel:.8f}"])

                    outputs.zero_()
                    outputs = torch.stack([_loaded_f0, _loaded_f1], dim=0)  # 2,32,128,128

                else:
                    raise TypeError(type(outputs))

            self.prompt_item_index += 1
            return outputs

        return hook

    def register_forward_hooks(self, layer_names):
        print(f"Add the hook into {layer_names}...")
        all_components = self.pipe.components

        hooks = []
        for name in layer_names:
            if name == "vae.encoder":
                sub_model: AutoencoderKL = self.pipe.vae
                module = sub_model.get_submodule("encoder")
                hooks.append(module.register_forward_hook(self.get_hook(name=name)))
            elif name == "vae.decoder":
                tqdm.write("-->> Skip the vae.decoder for hook! <<--")
            else:
                if name in all_components:  # 会自动排除“vae.decoder”
                    module: nn.Module = all_components[name]
                    hooks.append(module.register_forward_hook(self.get_hook(name=name)))
        return hooks

    def get_cond_image(self, image: np.array, height=HEIGHT, width=WIDTH):
        """智能中心裁剪逻辑 (确保填满 1024x1024 不变形)"""
        h, w = image.shape[:2]
        ratio_w = width / w
        ratio_h = height / h
        if ratio_w > ratio_h:
            new_w = width
            new_h = math.ceil(h * ratio_w)
        else:
            new_h = height
            new_w = math.ceil(w * ratio_h)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 中心裁剪
        start_x = (new_w - width) // 2
        start_y = (new_h - height) // 2
        cropped = resized[start_y : start_y + height, start_x : start_x + width]

        if self.ctrlnet_name_or_path.endswith("blur"):
            gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            processed = cv2.GaussianBlur(gray, (55, 55), 0)
            mode = "blur"
        elif self.ctrlnet_name_or_path.endswith("canny"):
            # assuming img is a PIL image
            gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            processed = cv2.Canny(gray, 50, 150)
            mode = "canny"
        else:
            raise NotImplementedError(f"不支持的 ControlNet 类型: {self.ctrlnet_name_or_path}")
        return gray, processed, mode

    @torch.inference_mode()
    def text2image(self, prompts, cond_images, steps=60, h=HEIGHT, w=WIDTH):
        assert len(prompts) == len(cond_images), (len(prompts), len(cond_images))

        self.total_mse_sum = 0
        self.total_numel = 0

        # 想要计算后端的特征误差，就必须要执行前端推理
        generators = [torch.Generator(device="cpu").manual_seed(42) for _ in range(len(prompts))]
        latents = self.pipe(
            prompt=prompts,
            # negative_prompt="",
            num_inference_steps=steps,
            height=h,
            width=w,
            return_dict=False,
            generator=generators,
            max_sequence_length=77,
            output_type="latent",
            control_image=cond_images,
            controlnet_conditioning_scale=1.0,
            guidance_scale=3.5,
        )[0]

        if "vae.decoder" in self.target_layer_names:
            tqdm.write(f"-->> vae.decoder for {self.task_name} <<--")
            inspect_structure(latents, prefix="latents", print_fn=tqdm.write)
            tqdm.write("-" * 30)

            if self.load_root:
                name = "vae.decoder"
                tqdm.write(f"[Feature Coding] Replace features for layer {name}...")

                assert self.prompt_item_index == 0  # 不能同时加载前面的prompt特征
                _loaded = self.current_target_features[name]
                _loaded = load_tensor_using_ref(_loaded, ref=latents[0])
                _mse_sum = compute_mse(_loaded, latents[0])
                self.total_mse_sum += _mse_sum
                _numel = _loaded.numel()
                self.total_numel += _numel
                self.sample_mse_msg.append([name, f"{_mse_sum / _numel:.8f}"])

                latents = _loaded.unsqueeze(0)  # 补充batch维度

        image = self.latent2image(latents)

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

        return image

    @torch.inference_mode()
    def latent2image(self, latents):
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor

        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")
        return image

    def print_feature_coding_stats(self):
        """Print accumulated feature coding statistics per group."""
        if not self.sample_fc_stats:
            return

        # Print stats for each group separately
        num_samples = len(self.sample_fc_stats)
        if num_samples == 0:
            tqdm.write("No feature coding stats available.")
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


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, default="selected_100samples/data_infos.json")
    parser.add_argument("--image_root", type=str, default="selected_100samples/images")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ref_root", type=str, default="sd3.5-l-100predictions/sd35cond_recalc")
    parser.add_argument("--gen_root", type=str, required=True)
    parser.add_argument("--load_root", type=str)
    parser.add_argument("--save_cond_image", action="store_true")
    parser.add_argument("--load_condition", action="store_true")
    parser.add_argument("--load_latent", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    # fmt: on

    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    log_filename = f"sd35cond_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = Path(args.output) / log_filename
    tee_logger = TeeLogger(str(log_filepath))
    original_stdout = sys.stdout
    sys.stdout = tee_logger
    tqdm.write(f"日志将写入: {log_filepath}")

    _data_infos = get_data_infos(args.data_json, image_root=args.image_root)
    selected_keys = sorted(_data_infos.keys())
    if args.limit > 0:
        selected_keys = selected_keys[: args.limit]
    data_infos = {key: _data_infos[key] for key in selected_keys}

    gen_root = Path(args.gen_root)
    gen_root.mkdir(parents=True, exist_ok=True)

    sd3pipe = SD3CtrlNetFeatureCodingPipeline(
        load_root=args.load_root, load_condition=args.load_condition, load_latent=args.load_latent
    )

    t_task_start = time.perf_counter()
    for i, (image_name, (image_path, image_caption)) in enumerate(data_infos.items()):
        t_sample_start = time.perf_counter()
        print(f"Processing the sample-{i}/{len(data_infos)} {image_name} with '{image_caption}'...")

        assert args.image_root
        image_path = Path(args.image_root) / image_name
        image = np.array(Image.open(image_path).convert("RGB"))
        gray_image, cond_image, cond_name = sd3pipe.get_cond_image(image)
        cond_image = Image.fromarray(cond_image).convert("RGB")

        if args.save_cond_image:
            demo_image = np.concatenate([gray_image, cond_image], axis=1)
            cond_path = gen_root / f"{cond_name}-{image_name}"
            Image.fromarray(demo_image).save(cond_path)

        # 开始处理batch
        if sd3pipe.load_root:
            sd3pipe.load_feature(sample_name=image_name)

        images = sd3pipe.text2image(prompts=[image_caption], cond_images=[cond_image])
        images[0].save((gen_root / image_name).with_suffix(".png"))

        sd3pipe.prompt_item_index = 0
        t_sample_total = time.perf_counter() - t_sample_start
        tqdm.write(tabulate([[image_name, f"{t_sample_total:.06f}s"]], headers=["Sample", "Time"]))

    t_task_total = time.perf_counter() - t_task_start
    tqdm.write(tabulate([["CTTI", f"{t_task_total:.06f}s"]], headers=["Task", "Time"]))

    # Print feature coding stats
    sd3pipe.print_feature_coding_stats()

    # Remove hooks to avoid memory leaks
    sd3pipe.remove_hooks()

    fid_score = tti_evaluate_fid_wo_ref(args.ref_root, gen_root)
    tqdm.write(f"Final FID between {args.ref_root} and {gen_root}: {fid_score}")

    # 恢复 stdout 并关闭日志文件
    tqdm.write(f"\n日志已保存至: {log_filepath}")
    sys.stdout = original_stdout
    tee_logger.close()


if __name__ == "__main__":
    main()
