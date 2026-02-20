"""
IARA Kernels Package
====================
Custom GPU kernels for high-performance trading model inference.

Available Modules:
- fused_cfc: Triton kernels for CfC update step
- interface: PyTorch Autograd wrappers
"""

from src.kernels.interface import (
    FusedCfCFunction,
    FusedCfCCell,
    fused_cfc_update,
    TRITON_AVAILABLE
)

__all__ = [
    'FusedCfCFunction',
    'FusedCfCCell',
    'fused_cfc_update',
    'TRITON_AVAILABLE'
]
