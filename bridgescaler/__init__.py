from importlib.metadata import version, PackageNotFoundError

from packaging.version import parse

# 1. Internal Constants & PyTorch Checks
REQUIRED_TORCH_VERSION = "2.0.0"

def _get_torch_status():
    """Checks torch version via metadata without importing the module."""
    try:
        installed_version = version("torch")

        if parse(installed_version) < parse(REQUIRED_TORCH_VERSION):
            raise RuntimeError(
                f"PyTorch >= {REQUIRED_TORCH_VERSION} required; found {installed_version}"
            )

        return True, installed_version
    except PackageNotFoundError:
        return False, None

TORCH_AVAILABLE, TORCH_VERSION = _get_torch_status()

# 2. Base Imports
from .backend import save_scaler, load_scaler, print_scaler, read_scaler
from .group import GroupStandardScaler, GroupRobustScaler, GroupMinMaxScaler
from .deep import DeepStandardScaler, DeepMinMaxScaler, DeepQuantileTransformer
from .distributed import (DStandardScaler, DMinMaxScaler, DQuantileScaler)

# 3. Conditional Torch Imports
if TORCH_AVAILABLE:
    from .distributed_tensor import (
        DStandardScalerTensor,
        DMinMaxScalerTensor,
    )

# 4. Define Public API
__all__ = [
    # Utilities
    "save_scaler", "load_scaler", "print_scaler", "read_scaler",
    "TORCH_AVAILABLE",
    # Scalers
    "GroupStandardScaler", "GroupRobustScaler", "GroupMinMaxScaler",
    "DeepStandardScaler", "DeepMinMaxScaler", "DeepQuantileTransformer",
    "DStandardScaler", "DMinMaxScaler", "DQuantileScaler",
]

if TORCH_AVAILABLE:
    __all__ += ["DStandardScalerTensor", "DMinMaxScalerTensor"]