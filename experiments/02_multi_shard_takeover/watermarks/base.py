from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class WMArtifact:
	name: str
	meta: Dict[str, Any]
	slots: Optional[list] = None
	bits: Optional[Array] = None


class WatermarkScheme:
	"""Stateless scheme interface; state goes into WMArtifact."""
	name: str

	def prepare(
		self,
		W_prev: Dict[str, Array],
		W_curr: Dict[str, Array],
		kx: bytes,
		slots: Optional[list],
		K: int,
		rng: np.random.RandomState,
	) -> WMArtifact:
		raise NotImplementedError

	def embed(
		self,
		W_curr: Dict[str, Array],
		artifact: WMArtifact,
		alpha: float,
		steps: int = 0,
		eta: float = 0.95,
	) -> Tuple[Dict[str, Array], WMArtifact]:
		raise NotImplementedError

	def extract(
		self,
		W_curr: Dict[str, Array],
		artifact: WMArtifact,
	) -> np.ndarray:
		raise NotImplementedError

	def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
		a = a.astype(np.uint8); b = b.astype(np.uint8)
		return 1.0 - float(np.mean(a ^ b))
