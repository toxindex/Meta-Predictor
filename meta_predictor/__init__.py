"""
Meta-Predictor Package
A Python package for metabolite prediction using transformer models
"""

from .core.predictor import MetaPredictorWrapper
from .core.utils import canonicalise_smile, smi_tokenizer, check_smile

__version__ = "1.0.0"
__author__ = "Meta-Predictor Team"
__description__ = "Metabolite prediction using transformer models"

__all__ = [
    "MetaPredictorWrapper",
    "canonicalise_smile",
    "smi_tokenizer", 
    "check_smile"
]