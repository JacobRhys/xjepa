"""GPU-resident data layer.

The whole corpus lives in VRAM; batching is index arithmetic on device.  No
``DataLoader``, no workers, no ``pin_memory``, no prefetch, no collate.
"""

from xjepa.data.masking import (
    MaskingConfig,
    Masker,
    apply_corruption,
    random_mask,
    span_mask,
)
from xjepa.data.store import (
    DEFAULT_BUCKETS,
    TOKEN_BUDGET,
    Batch,
    BucketBatcher,
    GpuCorpus,
)

__all__ = [
    "Batch",
    "BucketBatcher",
    "GpuCorpus",
    "DEFAULT_BUCKETS",
    "TOKEN_BUDGET",
    "MaskingConfig",
    "Masker",
    "random_mask",
    "span_mask",
    "apply_corruption",
]
