"""Utility functions for watermarks and other modules."""

from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np


def hmac_sha256(key: bytes, message: bytes) -> bytes:
    """Simple HMAC-SHA256 implementation."""
    if len(key) > 64:
        key = hashlib.sha256(key).digest()
    if len(key) < 64:
        key = key + b'\x00' * (64 - len(key))
    
    o_key_pad = bytes([x ^ 0x5c for x in key])
    i_key_pad = bytes([x ^ 0x36 for x in key])
    
    inner = hashlib.sha256(i_key_pad + message).digest()
    return hashlib.sha256(o_key_pad + inner).digest()


def prf(key: bytes, message: bytes) -> bytes:
    """Pseudo-random function using HMAC-SHA256."""
    return hmac_sha256(key, message)


def safe_logits(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Safely compute logits X @ q with numerical stability."""
    try:
        result = X @ q
        return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    except Exception:
        return np.zeros(X.shape[0], dtype=np.float64)

