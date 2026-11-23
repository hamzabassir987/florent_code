"""Portable float/int8 inference for Florent actor policies.

Only the standard library is used: the production runtime has no numpy and a
10 ms per-unit budget. Two artifact shapes are accepted:

* ``{"policies": {"0": [layer, ...], ...}}`` - the per-unit-policy export
  produced by ``export_weights`` from a trained LibTorch checkpoint;
* ``{"layers": [...]}`` - a single flat MLP, kept for tests and small models.

Weights may be float or symmetrically quantised int8 (``scale`` present).
"""
import json
from pathlib import Path


class LearnedPolicy:
    def __init__(self, checkpoint=None):
        self.enabled = False
        self.meta = {}
        self.policies = {}      # policy index -> list of layers
        self.layers = []        # flat fallback
        self.quantized = False
        if checkpoint:
            self._load(checkpoint)

    def _load(self, checkpoint):
        path = Path(str(checkpoint))
        try:
            data = json.loads(path.read_text())
            self.meta = data.get("meta", {})
            if data.get("format") not in ("florent-policy-json-v1", "florent-policy-int8-v1"):
                return
            self.quantized = data.get("format") == "florent-policy-int8-v1"
            raw = data.get("policies")
            if isinstance(raw, dict) and raw:
                self.policies = {int(k): v for k, v in raw.items()}
            else:
                self.layers = data.get("quantized_layers" if self.quantized else "layers", [])
            self.enabled = (bool(self.policies) or bool(self.layers)) and \
                self.meta.get("reward", [-1, 0, 1]) == [-1, 0, 1]
        except (OSError, ValueError, TypeError, KeyError):
            self.enabled = False

    # -- forward ---------------------------------------------------------
    @staticmethod
    def _dense(x, layer):
        """Dense affine layer; skips zero inputs to exploit the sparse patch."""
        w = layer.get("weights", ())
        b = layer.get("bias", ())
        scale = layer.get("scale")
        if scale is not None:
            w = [[float(v) * scale for v in row] for row in w]
            b = [float(v) * scale for v in b]
        out = [float(v) for v in b] if b else [0.0] * len(w)
        nz = [(i, float(v)) for i, v in enumerate(x) if v]
        for j, row in enumerate(w):
            z = out[j]
            for i, xv in nz:
                if i < len(row):
                    z += float(row[i]) * xv
            out[j] = z
        return out

    def logits(self, observation, policy=0):
        if not self.enabled:
            return []
        layers = self.policies.get(policy, self.layers if not self.policies else [])
        if not layers:
            return []
        x = [float(v) for v in observation]
        last = len(layers) - 1
        for i, layer in enumerate(layers):
            x = self._dense(x, layer)
            if i != last:
                x = [v if v > 0.0 else 0.0 for v in x]
        return x

    def choose(self, candidates, mask=None, observation=None, policy=0, cpu_ok=True):
        """Deterministic masked argmax over the *fixed* candidate list.

        ``mask`` is aligned 1:1 with ``candidates``; candidate indices never
        renumber, so a checkpoint's head layout stays valid. Falls back to the
        wait candidate (index 0) whenever the model is unavailable.
        """
        if not candidates:
            return None
        wait = candidates[0]
        if not self.enabled or not cpu_ok or observation is None:
            return wait
        scores = self.logits(observation, policy)
        if not scores:
            return wait
        best, best_score = None, -1e30
        for i, c in enumerate(candidates):
            if mask is not None and (i >= len(mask) or not mask[i]):
                continue
            s = scores[i] if i < len(scores) else -1e30
            if s > best_score:
                best, best_score = c, s
        return best if best is not None else wait


def cpu_budget_ok(ct, reserve_us=1500):
    """Leave a reserve for the official bot logic and never use wall time."""
    try:
        return int(ct.get_cpu_time_elapsed()) < 10000 - int(reserve_us)
    except Exception:
        return True


def apply(ct, candidate):
    from .action_abi import apply as apply_candidate
    return apply_candidate(ct, candidate)
