import io
from collections import UserDict
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import zstandard as zstd
from tabulate import SEPARATING_LINE, tabulate
from tqdm import tqdm

# =============================================================================
# Print Utilities (using tabulate)
# =============================================================================


class TablePrinter:
    """基于 tabulate 的静态工具类。无需实例化，直接调用类方法。

    Example:
        >>> TablePrinter.table({"arch": "hyperprior", "bit_depth": 10})
        >>> TablePrinter.kv([("BPFP", 0.123), ("MSE", 0.001)])
        >>> TablePrinter.boxed("Line 1", "Line 2", title="Info")
    """

    # 共享配置
    DEFAULT_FMT: str = "simple_outline"
    PRINT_FN: Callable[[str], None] = tqdm.write
    LINE_SEP: str = SEPARATING_LINE

    @classmethod
    def configure(cls, fmt: str = None, print_fn=None):
        if fmt:
            cls.DEFAULT_FMT = fmt
        if print_fn:
            cls.PRINT_FN = print_fn

    @classmethod
    def table(
        cls,
        data: Union[Dict[str, Any], List[Tuple[str, Any]], List[List[Any]]],
        headers: Union[List[str], str] = (),
        **kwargs,
    ):
        """打印通用表格。"""
        if isinstance(data, dict):
            data = list(data.items())
        table_str = tabulate(data, headers=headers, tablefmt=cls.DEFAULT_FMT, **kwargs)
        cls.PRINT_FN(table_str)

    @classmethod
    def boxed(cls, *lines: str, title: str = None, **kwargs):
        """在边框中打印多行文本。"""
        data = [[line] for line in lines]
        if title:
            data = [[title]] + data
        table_str = tabulate(data, tablefmt=cls.DEFAULT_FMT, **kwargs)
        cls.PRINT_FN(table_str)


DTYPE_TORCHTYPE_TO_BITS = {
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float32: 32,
    torch.float64: 64,
    torch.int8: 8,
    torch.uint8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.bool: 1,
}
DTYPE_STR_TO_TORCHTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


class FixedKeyDefaultDict(UserDict):
    def __init__(self, fixed_keys, *, default_factory=lambda: None):
        """
        :param fixed_keys: 允许存在的键列表（白名单）
        :param default_factory: 默认值的工厂函数，如 int, list, lambda: None
        """
        self.default_factory = default_factory
        self._allowed_keys = set(fixed_keys)  # 使用集合加速查询
        self.data = {}

        # 初始化时，根据白名单预填充默认值
        for key in self._allowed_keys:
            self.data[key] = self.default_factory()

    def __setitem__(self, key, value):
        # 核心限制：如果 key 不在白名单中，抛出错误
        if key not in self._allowed_keys:
            raise KeyError(f"禁止添加新键: '{key}'。只允许: {self._allowed_keys}")

        super().__setitem__(key, value)

    def __missing__(self, key):
        # 额外的 defaultdict 特性支持：如果某个允许的 Key 被意外删除了，再次访问时自动恢复默认值
        if key in self._allowed_keys:
            val = self.default_factory()
            self.data[key] = val
            return val

        raise KeyError(key)


def load_zst_tensor(file_path: str) -> Any:
    """Load tensor data from a .zst compressed file.

    Args:
        file_path: Path to the .zst file.

    Returns:
        Loaded data (can be Tensor, dict, list, etc.)
    """
    assert file_path.endswith(".zst"), f"Expected .zst file, got: {file_path}"

    dctx = zstd.ZstdDecompressor()
    with zstd.open(file_path, "rb", dctx=dctx) as f:
        # 读取到内存 buffer，解决 seek backwards 错误
        buffer = io.BytesIO(f.read())
        data = torch.load(buffer, map_location="cpu")
    return data


def save_zst_tensor(data: Any, file_path: str, compression_level: int = 3) -> None:
    """Save tensor data to a .zst compressed file.

    Args:
        data: Data to save (Tensor, dict, etc.)
        file_path: Path to save the .zst file.
        compression_level: Zstandard compression level (1-22).
    """
    if not file_path.endswith(".zst"):
        file_path += ".zst"

    # First save to bytes buffer
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)

    # Compress and write
    cctx = zstd.ZstdCompressor(level=compression_level)
    with open(file_path, "wb") as f:
        f.write(cctx.compress(buffer.read()))


def load_tensor(file_path: str) -> Any:
    """Load tensor from .pt or .pt.zst or .zst file.

    Args:
        file_path: Path to the tensor file.

    Returns:
        Loaded data.
    """
    if file_path.endswith(".zst"):
        return load_zst_tensor(file_path)
    elif file_path.endswith(".npy"):
        data = np.load(file_path)
        return torch.from_numpy(data)
    else:
        return torch.load(file_path, map_location="cpu", weights_only=False)


def save_tensor(data: Any, file_path: str, use_zst: bool = False) -> None:
    """Save tensor to .pt or .pt.zst file.

    Args:
        data: Data to save.
        file_path: Path to save the file.
        use_zst: Whether to use zst compression.
    """
    if use_zst or file_path.endswith(".zst"):
        save_zst_tensor(data, file_path)
    else:
        torch.save(data, file_path)


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor, return_average: bool = False) -> float:
    """Compute Mean Squared Error between two tensors.

    Args:
        original: Original tensor.
        reconstructed: Reconstructed tensor.

    Returns:
        MSE.
    """
    mse = (original.float() - reconstructed.float()) ** 2
    if return_average:
        return torch.mean(mse).item()
    return torch.sum(mse).item()


def compute_nested_mse(original: dict, reconstructed: dict) -> Tuple[float, int]:
    """Recursively compute total MSE and element count for nested structures.

    Supports Dict, Tuple, List, and Tensor. Returns (total_squared_error, total_elements).

    Args:
        original: Original nested structure.
        reconstructed: Reconstructed nested structure (must have same structure).

    Returns:
        Tuple of (total_squared_error, total_num_elements).
    """
    if isinstance(original, torch.Tensor):
        assert isinstance(reconstructed, torch.Tensor), (
            f"Structure mismatch: expected Tensor, got {type(reconstructed)}"
        )
        mse_sum = ((original.float() - reconstructed.float()) ** 2).sum().item()
        return mse_sum, original.numel()

    elif isinstance(original, dict):
        assert isinstance(reconstructed, dict), f"Structure mismatch: expected dict, got {type(reconstructed)}"
        total_mse, total_count = 0.0, 0
        for key in original.keys():
            assert key in reconstructed, f"Missing key in reconstructed: {key}"
            mse, count = compute_nested_mse(original[key], reconstructed[key])
            total_mse += mse
            total_count += count
        return total_mse, total_count

    else:
        raise TypeError(f"Unsupported type: {type(original)}")


def compute_nested_mse_detailed(original: dict, reconstructed: dict) -> Tuple[Dict[str, Dict[str, float]], float, int]:
    """Compute per-key MSE and total MSE for nested dict structures.

    Only works for flat or shallow dict structures (dict of Tensors or dict of dicts of Tensors).

    Args:
        original: Original dict structure.
        reconstructed: Reconstructed dict structure (must have same structure).

    Returns:
        Tuple of (per_key_mse_dict, total_squared_error, total_num_elements).
        per_key_mse_dict: {key: {"mse": avg_mse, "count": num_elements, "mse_sum": squared_error_sum}}
    """
    assert isinstance(original, dict), f"Expected dict, got {type(original)}"
    assert isinstance(reconstructed, dict), f"Expected dict, got {type(reconstructed)}"

    per_key_mse = {}
    total_mse_sum = 0.0
    total_count = 0

    for key in original.keys():
        assert key in reconstructed, f"Missing key in reconstructed: {key}"
        mse_sum = ((original[key].float() - reconstructed[key].float()) ** 2).sum().item()
        count = original[key].numel()
        avg_mse = mse_sum / count if count > 0 else 0.0
        per_key_mse[key] = {"mse": avg_mse, "count": count, "mse_sum": mse_sum}

        total_mse_sum += mse_sum
        total_count += count

    return per_key_mse, total_mse_sum, total_count


def get_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """Get information about a tensor.

    Args:
        tensor: Input tensor.

    Returns:
        Dict with shape, dtype, device, min, max, mean, std, numel, dtype_bits.
    """
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.float().mean().item(),
        "std": tensor.float().std().item(),
        "numel": tensor.numel(),
        "dtype_bits": DTYPE_TORCHTYPE_TO_BITS[tensor.dtype],
    }


def inspect_structure(obj, prefix="", indent=0, print_fn=tqdm.write):
    """
    递归解析并打印对象结构。

    Args:
        obj: 要解析的对象 (Tensor, list, dict, etc.)
        prefix: 当前打印的前缀 (通常是变量名或层名)
        indent: 缩进级别，用于可视化层级
        print_fn: 打印函数，默认使用 tqdm.write 以避免打断进度条
    """
    # 生成缩进空格
    space = "  " * indent

    # 1. 处理 PyTorch Tensor
    if isinstance(obj, torch.Tensor):
        print_fn(f"{space}{prefix}: [Tensor] shape={obj.shape}, dtype={obj.dtype}, device={obj.device}")

    # 2. 处理字典
    elif isinstance(obj, dict):
        print_fn(f"{space}{prefix}: [{type(obj).__name__}] with {len(obj)} keys")
        for k, v in obj.items():
            # 递归调用，前缀加上 key
            inspect_structure(v, prefix=f"key['{k}']", indent=indent + 1, print_fn=print_fn)

    # 3. 处理列表和元组
    elif isinstance(obj, (list, tuple)):
        print_fn(f"{space}{prefix}: [{type(obj).__name__}] with {len(obj)} items")
        for i, v in enumerate(obj):
            # 递归调用，前缀加上 index
            inspect_structure(v, prefix=f"item[{i}]", indent=indent + 1, print_fn=print_fn)

    # 4. 处理基本数值类型 (int, float, bool)
    elif isinstance(obj, (int, float, bool, str)):
        print_fn(f"{space}{prefix}: [{type(obj).__name__}] value={obj}")

    # 5. 处理 None
    elif obj is None:
        print_fn(f"{space}{prefix}: [None]")

    # 6. 其他未知类型
    else:
        print_fn(f"{space}{prefix}: [Unknown Type: {type(obj).__name__}] {obj}")


def load_tensor_using_ref(data, ref):
    # 当 numpy 保存 dict/list 时，经常包裹成 0-d array (shape=())
    if isinstance(data, np.ndarray) and data.ndim == 0 and data.dtype == "O":
        data = data.item()

    if isinstance(ref, torch.Tensor):
        if not isinstance(data, torch.Tensor):
            tensor = torch.as_tensor(data)
        else:
            tensor = data
        data = tensor.to(device=ref.device, non_blocking=True)

    elif isinstance(ref, dict):
        if not isinstance(data, dict):
            raise TypeError(f"Structure mismatch: Ref is dict but Cached is {type(data)}")

        restored_dict = {}
        for k, v_ref in ref.items():
            if k not in data:
                raise KeyError(f"Missing key '{k}' in cached features.")
            restored_dict[k] = load_tensor_using_ref(data[k], v_ref)
        data = restored_dict

    elif isinstance(ref, (list, tuple)):
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise TypeError(f"Structure mismatch: Ref is list/tuple but Cached is {type(data)}")

        if len(data) != len(ref):
            raise ValueError(f"Length mismatch: Ref has {len(ref)} items, Cached has {len(data)}")

        restored_list = []
        for i, item_ref in enumerate(ref):
            restored_list.append(load_tensor_using_ref(data[i], item_ref))

        # 如果原版是 tuple，转换回 tuple
        data = tuple(restored_list) if isinstance(ref, tuple) else restored_list

    return data


def recursive_check_equal(obj1, obj2, verbose=False):
    """递归检查两个嵌套结构（dict, list, tensor）是否完全一致。"""
    # 1. 类型检查：如果类型不同，直接返回 False
    if type(obj1) is not type(obj2):
        if verbose:
            print(f"Type mismatch: {type(obj1)} vs {type(obj2)}")
        return False

    # 2. 处理字典
    if isinstance(obj1, dict):
        # 键的集合必须相同
        if set(obj1.keys()) != set(obj2.keys()):
            if verbose:
                print(f"Key mismatch: {obj1.keys()} vs {obj2.keys()}")
            return False
        # 递归检查每一个 key 对应的值
        for key in obj1:
            if not recursive_check_equal(obj1[key], obj2[key], verbose):
                if verbose:
                    print(f"Value mismatch at key: '{key}'")
                return False
        return True

    # 3. 处理列表或元组
    elif isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            if verbose:
                print(f"Length mismatch: {len(obj1)} vs {len(obj2)}")
            return False
        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            if not recursive_check_equal(item1, item2, verbose):
                if verbose:
                    print(f"Value mismatch at list index: {i}")
                return False
        return True

    # 4. 处理 PyTorch Tensor
    elif isinstance(obj1, torch.Tensor):
        # 检查形状是否一致
        if obj1.shape != obj2.shape:
            if verbose:
                print(f"Tensor shape mismatch: {obj1.shape} vs {obj2.shape}")
            return False
        # 检查数值是否完全一致
        # 注意：如果张量在不同设备(CPU/GPU)，通常需要移到同一设备比较
        if obj1.device != obj2.device:
            # 如果允许跨设备比较，可以都转到 cpu()
            return torch.equal(obj1.cpu(), obj2.cpu())
        return torch.equal(obj1, obj2)

    # 5. 其他基础类型 (int, float, str, None)
    else:
        if obj1 != obj2:
            if verbose:
                print(f"Primitive mismatch: {obj1} vs {obj2}")
            return False
        return True
