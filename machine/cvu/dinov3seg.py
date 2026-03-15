"""
DINOv3 Feature Coding Evaluation Script.

该脚本仅评估被codec重构后的特征自身对原始模型下游任务上的性能。
"""

import argparse
import io
import itertools
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from tabulate import SEPARATING_LINE, tabulate
from torch.utils import data
from torchvision.transforms import functional as TF
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


class ADE20K(data.Dataset):
    CROP_SIZE = 896
    SLIDE_STRDE = 596
    NUM_CLASSES = 150

    def __init__(self, data_root, use_tta=False):
        super().__init__()

        data_root = Path(data_root)
        image_root = data_root / "images" / "validation"
        annot_root = data_root / "annotations" / "validation"

        self.samples = []
        for image_path in image_root.iterdir():
            if not image_path.is_file():
                continue
            annot_path = annot_root / image_path.name.replace(".jpg", ".png")
            assert annot_path.exists(), (image_path, annot_path)
            self.samples.append((image_path, annot_path))
        tqdm.write(f"Total: {len(self.samples)} pairs of image and annot images.")
        # self.samples = self.samples[:5]

        self.transforms = make_segmentation_eval_transforms(
            img_size=ADE20K.CROP_SIZE,
            inference_mode="slide",
            use_tta=use_tta,
            tta_ratios=[0.9, 0.95, 1.0, 1.05, 1.1],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, annot_path = self.samples[index]
        image = self.load_image(image_path)
        annot = self.load_annot(annot_path)
        image, annot = self.transforms(image, annot)
        return image, annot, image_path.stem

    def load_image(self, path):
        with open(path, mode="rb") as f:
            target_data = f.read()
        f = io.BytesIO(target_data)
        return Image.open(f).convert(mode="RGB")

    def load_annot(self, path):
        with open(path, mode="rb") as f:
            target_data = f.read()
        f = io.BytesIO(target_data)
        return Image.open(f)


class FeatureCodingWrapper:
    def __init__(self, load_root: str):
        self.model = dinov3_vit7b16_ms(pretrained=True, autocast_dtype=torch.float32, check_hash=True)
        tqdm.write("[Feature Coding] Change backbone._get_intermediate_layers_not_chunked...")
        self.model.segmentation_model[0].backbone._get_intermediate_layers_not_chunked = types.MethodType(
            self.get_hook(), self.model.segmentation_model[0].backbone
        )
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
                "segmentation_model.0.backbone.blocks.9": reconstructed_data["features"]["layer.9"],
                "segmentation_model.0.backbone.blocks.19": reconstructed_data["features"]["layer.19"],
                "segmentation_model.0.backbone.blocks.29": reconstructed_data["features"]["layer.29"],
                "segmentation_model.0.backbone.blocks.39": reconstructed_data["features"]["layer.39"],
            }
        else:
            self.current_target_metadata = {}
            self.current_target_features = {**reconstructed_data}

        tqdm.write(f"Loaded features {self.current_target_features.keys()}.")
        inspect_structure(self.current_target_features)

    def get_hook(self):
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

            tqdm.write(f"-->>\tSample {self.sample_id} <<--")
            inspect_structure(outputs, prefix="Intermediate Layers Outputs", print_fn=tqdm.write)

            # Feature coding: encode and decode in memory
            if self.load_root:
                assert self.current_target_features, "No features to load!"

                for idx, output in enumerate(outputs):
                    name = f"segmentation_model.0.backbone.blocks.{blocks_to_take[idx]}"
                    tqdm.write(f"[Feature Coding] Replace {name}...")

                    _loaded = self.current_target_features[name]
                    _loaded = load_tensor_using_ref(_loaded, ref=output)
                    _mse_sum = compute_mse(_loaded, output)
                    self.total_mse_sum += _mse_sum
                    _numel = _loaded.numel()
                    self.total_numel += _numel
                    self.sample_mse_msg.append([name, f"{_mse_sum / _numel:.8f}"])

                    outputs[idx].zero_()
                    outputs[idx].copy_(_loaded)

            return outputs

        return hook

    @torch.inference_mode()
    def predict(self, x, rescale_to=(512, 512)):
        self.sample_mse_msg = []
        self.total_mse_sum = 0
        self.total_numel = 0

        with self.model.autocast_ctx():
            x = self.model.segmentation_model[0](x)  # backbone forward
            x = self.model.segmentation_model[1].predict(x, rescale_to=rescale_to)  # decoder head prediction

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


class SemanticSegmentationEvaluator:
    def __init__(self, data_root: str, load_root: str):
        self.dataset = ADE20K(data_root=data_root)
        tqdm.write(f"Initialized the dataset {self.dataset.__class__.__name__}...")
        self.dataloader = data.DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False)
        tqdm.write(f"Evaluate {len(self.dataset)} samples with {len(self.dataloader)} batches.")

        tqdm.write("Initialize the DINOv3 model...")
        self.model = FeatureCodingWrapper(load_root=load_root)

    @torch.no_grad()
    def slide_inference(self, image, num_classes, crop_size, stride, rescale_to):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to decode without padding.
        Args:
            image (tensor): the tensor should have a shape 1xCxHxW.
            num_classes (int): number of output channels
            crop_size (tuple): (h_crop, w_crop)
            stride (tuple): (h_stride, w_stride)
        Returns:
            Tensor: The output results from model of each input image.
        """
        batch_size, C, h_img, w_img = image.shape
        # As of now, the code assumes that a single image is passed at a time at inference time
        assert batch_size == 1

        h_crop, w_crop = crop_size
        if h_crop > h_img and w_crop > w_img:  # Meaning we are doing < 1.0 TTA
            h_crop, w_crop = min(h_img, w_img), min(h_img, w_img)

        h_stride, w_stride = stride
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs = []
        crop_xyxy = []
        for i, (h_idx, w_idx) in enumerate(itertools.product(range(h_grids), range(w_grids))):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = image[:, :, y1:y2, x1:x2]
            crop_imgs.append(crop_img)
            crop_xyxy.append([x1, y1, x2, y2])
        crop_imgs = torch.cat(crop_imgs, dim=0)

        crop_preds = self.model.predict(crop_imgs, rescale_to=crop_imgs.shape[2:])
        del crop_imgs

        # if decoder_head_type == "m2f":
        mask_preds, mask_clses = crop_preds["pred_masks"], crop_preds["pred_logits"]
        mask_clses = F.softmax(mask_clses, dim=-1)[..., :-1]
        mask_preds = mask_preds.sigmoid()
        crop_preds = torch.einsum("bqc,bqhw->bchw", mask_clses.to(torch.bfloat16), mask_preds.to(torch.bfloat16))
        del mask_clses, mask_preds

        preds = image.new_zeros((1, num_classes, h_img, w_img)).cpu()
        count_mat = image.new_zeros((1, 1, h_img, w_img)).to(torch.int8).cpu()
        for crop_pred, (x1, y1, x2, y2) in zip(crop_preds, crop_xyxy):
            preds += F.pad(crop_pred, (int(x1), int(preds.shape[-1] - x2), int(y1), int(preds.shape[-2] - y2))).cpu()
            count_mat[:, :, y1:y2, x1:x2] += 1
        del crop_preds
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        preds = F.interpolate(preds, size=rescale_to, mode="bilinear", align_corners=False)
        return preds

    # @torch.no_grad()
    # def whole_inference(self, x, rescale_to):
    #     pred = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    #     pred = self.model.predict(pred, rescale_to=rescale_to)

    #     # if decoder_head_type == "m2f":
    #     mask_pred, mask_cls = pred["pred_masks"], pred["pred_logits"]
    #     mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    #     mask_pred = mask_pred.sigmoid()
    #     pred = torch.einsum("bqc,bqhw->bchw", mask_cls.to(torch.float), mask_pred.to(torch.float))
    #     return pred

    @torch.no_grad()
    def evaluate(self):
        t_dataset_start = time.perf_counter()

        all_metric_values = []
        # samplewise_metric_values = {}
        for images, annot, images_stem in tqdm(self.dataloader, total=len(self.dataloader), ncols=78):
            t_sample_start = time.perf_counter()

            if not isinstance(images, (list, tuple)):
                images = [images]
            if not isinstance(images_stem, (list, tuple)):
                images_stem = [images_stem]

            images = [x.to(DEVICE) for x in images]
            annot = annot.to(DEVICE)

            annot_hw = annot.shape[-2:]
            aggregated_preds = torch.zeros(1, ADE20K.NUM_CLASSES, annot.shape[-2], annot.shape[-1])
            for image_idx, (image, stem) in enumerate(zip(images, images_stem)):
                sample_id = f"{stem}-stackedpatches"

                if self.model.load_root:
                    self.model.set_sample_info(sample_id=sample_id, current_idx=image_idx, total_samples=len(images))

                # following the paper, sliding inference is always used
                _preds = self.slide_inference(
                    image,
                    num_classes=ADE20K.NUM_CLASSES,
                    crop_size=(ADE20K.CROP_SIZE, ADE20K.CROP_SIZE),
                    stride=(ADE20K.SLIDE_STRDE, ADE20K.SLIDE_STRDE),
                    rescale_to=annot_hw,
                )

                if image_idx > 0 and image_idx >= len(images) / 2:
                    _preds = TF.hflip(_preds)
                aggregated_preds += _preds.softmax(dim=1)

            aggregated_preds = (aggregated_preds / len(images)).argmax(dim=1, keepdim=True).to(DEVICE)
            # for batch_size=1
            intersect_and_union = calculate_intersect_and_union(
                aggregated_preds[0], annot[0], num_classes=ADE20K.NUM_CLASSES, reduce_zero_label=True
            )
            all_metric_values.append(intersect_and_union)
            # samplewise_metric_values[images_stem[0]] = torch.stack([intersect_and_union])
            del images, annot, aggregated_preds, intersect_and_union

            t_sample_end = time.perf_counter()
            tqdm.write(f"Sample {sample_id} took {(t_sample_end - t_sample_start):.2f} seconds")

        all_metric_values = torch.stack(all_metric_values)
        final_metrics = calculate_segmentation_metrics(all_metric_values, metrics=["mIoU", "dice", "fscore"])
        final_metrics = {k: round(v.cpu().item() * 100, 2) for k, v in final_metrics.items()}

        # samplewise_metrics = {
        #     image_name: {
        #         k: round(v.cpu().item() * 100, 2)
        #         for k, v in calculate_segmentation_metrics(value, metrics=["mIoU", "dice", "fscore"]).items()
        #     }
        #     for image_name, value in samplewise_metric_values.items()
        # }
        # final_metrics, samplewise_metrics
        tqdm.write(f"Semantic Segmentation: {final_metrics}")

        t_dataset_end = time.perf_counter()
        tqdm.write(f"Dataset took {(t_dataset_end - t_dataset_start):.2f} seconds")

        # Print feature coding stats
        self.model.print_feature_coding_stats()


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Segmentation Evaluation")
    parser.add_argument("--data_root", type=str, required=True, help="Path to grouped images")
    parser.add_argument("--output", type=str, default="new_results", help="Output directory for logs")
    parser.add_argument("--load_root", type=str, help="Path to loading features")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    log_filename = f"dinov3seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = Path(args.output) / log_filename
    tee_logger = TeeLogger(str(log_filepath))
    original_stdout = sys.stdout
    sys.stdout = tee_logger
    tqdm.write(f"日志将写入: {log_filepath}")

    evaluator = SemanticSegmentationEvaluator(data_root=args.data_root, load_root=args.load_root)
    evaluator.evaluate()

    # 恢复 stdout 并关闭日志文件
    tqdm.write(f"\n日志已保存至: {log_filepath}")
    sys.stdout = original_stdout
    tee_logger.close()


if __name__ == "__main__":
    main()
