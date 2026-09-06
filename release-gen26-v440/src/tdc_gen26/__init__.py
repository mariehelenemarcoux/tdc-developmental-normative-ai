"""TDC Gen2.6 current canonical clean-room implementation."""
from .canonical import CanonicalTDCGen26V440, TDCDecision
from .pytorch_model import TDCGen26Torch
from .trust import MultidimensionalTrust
from .authority import AuthorityState
from .losses import TDCCompositeLoss
from .third_factor import ThirdFactorArbiter
from .recursive_zoom import RecursiveZoomController
from .metaconfidence import MetaConfidence
from .regime import TemporalRegimeState
from .voi import budgeted_voi

__all__ = [
    "CanonicalTDCGen26V440", "TDCDecision", "TDCGen26Torch",
    "MultidimensionalTrust", "AuthorityState", "TDCCompositeLoss",
    "ThirdFactorArbiter", "RecursiveZoomController", "MetaConfidence",
    "TemporalRegimeState", "budgeted_voi",
]
__version__ = "2.6.0.post440"
REFERENCE_VERSION = "v440"
