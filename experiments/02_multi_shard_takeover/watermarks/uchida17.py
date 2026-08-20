from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np

from .base import WMArtifact, WatermarkScheme, Array
from utils import hmac_sha256, prf, safe_logits
import hashlib


def _rng_from(key: bytes, tag: str) -> np.random.RandomState:
	seed = int.from_bytes(hmac_sha256(key, tag.encode("utf8"))[:8], "big", signed=False) % (2**31 - 1)
	return np.random.RandomState(seed)


class Uchida17(WatermarkScheme):
	name = "uchida17"

	def prepare(
		self,
		W_prev: Dict[str, Array],
		W_curr: Dict[str, Array],
		kx: bytes,
		slots: Optional[List[Tuple[str, int]]],
		K: int,
		rng: np.random.RandomState,
	) -> WMArtifact:
		slots = slots or []
		# Build q vector to determine dimensionality
		q = self._build_q(W_curr, slots)
		B = min(len(slots), K, 128)
		# Pre-build projection matrix and target bits
		X = self._build_X(kx, q.size, B) if q.size > 0 else np.zeros((0, 0), dtype=np.float64)
		target_bits = self._build_bits(kx, B)
		meta = {"kx": kx, "K": K, "B": B, "X": X, "target_bits": target_bits}
		return WMArtifact(name=self.name, meta=meta, slots=slots)

	def _build_q(self, W_curr: Dict[str, Array], slots: List[Tuple[str, int]]) -> np.ndarray:
		rows = []
		for lname, i in slots:
			rows.append(W_curr[lname][i].reshape(-1).astype(np.float64))
		return np.concatenate(rows, axis=0) if rows else np.zeros((0,), dtype=np.float64)

	def _build_X(self, kx: bytes, D: int, B: int) -> np.ndarray:
		rng = _rng_from(kx, f"X:{D}:{B}")
		X = rng.standard_normal((B, D)).astype(np.float64)
		# Add numerical stability and clipping
		X = np.clip(X, -10.0, 10.0)  # Prevent extreme values
		# row-normalize with better stability
		norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
		X_normalized = X / norms
		# Final clipping to prevent numerical issues
		return np.clip(X_normalized, -1.0, 1.0)

	def _build_bits(self, kx: bytes, B: int) -> np.ndarray:
		rng = _rng_from(kx, f"m:{B}")
		return (rng.rand(B) > 0.5).astype(np.uint8)

	def prepare_from_q(self, q: np.ndarray, kx: bytes, K: int, rng: np.random.RandomState) -> WMArtifact:
		"""Prepare scheme from feature vector q."""
		B = min(K, 128)
		X = self._build_X(kx, q.size, B) if q.size > 0 else np.zeros((0, 0), dtype=np.float64)
		target_bits = self._build_bits(kx, B)
		meta = {"kx": kx, "K": K, "B": B, "X": X, "target_bits": target_bits}
		return WMArtifact(name=self.name, meta=meta, slots=[])

	def embed_from_q(self, q: np.ndarray, artifact: WMArtifact, alpha: float) -> Tuple[np.ndarray, WMArtifact, np.ndarray]:
		"""Embed watermark in feature vector q and return updated q, artifact, and gradients."""
		X = artifact.meta["X"]
		target_bits = artifact.meta["target_bits"]
		
		if q.size == 0 or X.size == 0:
			return q.copy(), artifact, np.zeros_like(q)
		
		# Handle dimension mismatch
		if X.shape[1] != q.size:
			print(f"[embed_from_q] Dimension mismatch: X.shape={X.shape}, q.size={q.size}")
			return q.copy(), artifact, np.zeros_like(q)
		
		# Add numerical stability
		q_safe = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
		X_safe = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
		
		try:
			# Compute current bits and gradient
			logits = X_safe @ q_safe
			logits = np.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
			current_bits = (logits >= 0).astype(np.uint8)
			
			# Gradient for bit embedding: push logits towards target sign
			dq = np.zeros_like(q_safe)
			for i, (target, logit) in enumerate(zip(target_bits, logits)):
				target_sign = 2 * target - 1  # 0->-1, 1->+1
				if target_sign * logit < 1.0:  # if not confidently correct
					grad_contribution = X_safe[i] * target_sign * alpha
					grad_contribution = np.nan_to_num(grad_contribution, nan=0.0, posinf=alpha, neginf=-alpha)
					dq += grad_contribution
			
			# Clip gradient for stability
			dq = np.clip(dq, -alpha * 10, alpha * 10)
			q_new = q_safe + dq
			return q_new, artifact, dq
			
		except Exception as e:
			print(f"[embed_from_q] Error in computation: {e}")
			return q.copy(), artifact, np.zeros_like(q)

	def extract_from_q(self, q: np.ndarray, artifact: WMArtifact) -> np.ndarray:
		"""Extract watermark bits from feature vector q."""
		X = artifact.meta["X"]
		target_bits = artifact.meta.get("target_bits", np.zeros(0, dtype=np.uint8))
		
		if q.size == 0 or X.size == 0:
			return np.zeros(len(target_bits), dtype=np.uint8)
		
		# Handle dimension mismatch
		if X.shape[1] != q.size:
			print(f"[extract_from_q] Dimension mismatch: X.shape={X.shape}, q.size={q.size}")
			return np.zeros(len(target_bits), dtype=np.uint8)
		
		# Add numerical stability
		q_safe = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
		X_safe = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
		
		try:
			logits = safe_logits(X_safe, q_safe)
			logits = np.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
			return (logits >= 0).astype(np.uint8)
		except Exception as e:
			print(f"[extract_from_q] Error in matmul: {e}")
			return np.zeros(len(target_bits), dtype=np.uint8)

	def embed(
		self,
		W_curr: Dict[str, Array],
		artifact: WMArtifact,
		alpha: float,
		steps: int = 0,
		eta: float = 0.95,
	) -> Tuple[Dict[str, Array], WMArtifact]:
		# Perform a single gradient-like nudge on raw weights to increase margin for target bits.
		kx: bytes = artifact.meta["kx"]
		slots = artifact.slots or []
		# Build concatenated q and remember slices per slot
		pieces: List[np.ndarray] = []
		slices: List[Tuple[int, int, Tuple[str, int]]] = []
		cursor = 0
		for lname, i in slots:
			v = W_curr[lname][i].reshape(-1).astype(np.float64)
			pieces.append(v)
			sz = v.size
			slices.append((cursor, cursor + sz, (lname, i)))
			cursor += sz
		q = np.concatenate(pieces, axis=0) if pieces else np.zeros((0,), dtype=np.float64)
		B = artifact.meta.get("B", min(len(slots), 128))
		X = artifact.meta.get("X")
		m = artifact.meta.get("target_bits")
		# Rebuild X if dimensions changed
		if X is None or X.shape != (B, q.size):
			X = self._build_X(kx, q.size, B)
		if m is None:
			m = self._build_bits(kx, B)
		artifact.meta.update({"X": X, "B": B, "target_bits": m})
		# Compute gradient to increase margin for target bits m
		t = (m.astype(np.int8) * 2 - 1).astype(np.float64)  # +1 for bit1, -1 for bit0
		# Iterative refinement
		W_mod = {k: v.copy() for k, v in W_curr.items()}
		max_steps = max(1, int(steps))
		for _ in range(max_steps):
			# Rebuild q from current weights
			pieces_now: List[np.ndarray] = []
			slice_meta: List[Tuple[str, int, int, int]] = []  # lname, idx, start, end
			cursor = 0
			for lname, i in slots:
				v = W_mod[lname][i].reshape(-1).astype(np.float64)
				pieces_now.append(v)
				sz = v.size
				slice_meta.append((lname, i, cursor, cursor + sz))
				cursor += sz
			q_now = np.concatenate(pieces_now, axis=0) if pieces_now else np.zeros((0,), dtype=np.float64)
			B_now = min(len(slots), 128)
			if X is None or X.shape != (B_now, q_now.size):
				X = self._build_X(kx, q_now.size, B_now)
			if m is None or len(m) != B_now:
				m = self._build_bits(kx, B_now)
			# Gradient and update
			g_q_now = X.T @ ((m.astype(np.int8) * 2 - 1).astype(np.float64))
			for lname, i, s, e in slice_meta:
				grad = g_q_now[s:e]
				if grad.size == 0:
					continue
				dir_vec = grad / (np.linalg.norm(grad) + 1e-12)
				# Apply gradient to weights
				shape = W_mod[lname][i].shape
				delta = (alpha * dir_vec.reshape(shape)).astype(W_mod[lname].dtype)
				W_mod[lname][i] = (W_mod[lname][i] + delta).astype(W_mod[lname].dtype, copy=False)
		# After refinements, compute final bits
		piecesf = []
		for (l, j) in slots:
			piecesf.append(W_mod[l][j].reshape(-1).astype(np.float64))
		qf = np.concatenate(piecesf, axis=0) if piecesf else np.zeros((0,), dtype=np.float64)
		bits = (X @ qf >= 0.0).astype(np.uint8)
		artifact.bits = bits
		return W_mod, artifact

	def extract(
		self,
		W_curr: Dict[str, Array],
		artifact: WMArtifact,
	) -> np.ndarray:
		# Standard extraction
		kx: bytes = artifact.meta["kx"]
		q = self._build_q(W_curr, artifact.slots or [])
		B = artifact.meta.get("B") or min(len(artifact.slots or []), 128)
		X = artifact.meta.get("X")
		if X is None or X.shape != (B, q.size):
			X = self._build_X(kx, q.size, B)
		logits = safe_logits(X, q)
		return (logits >= 0.0).astype(np.uint8)
