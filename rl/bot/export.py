"""Export a portable JSON policy from a tensor checkpoint or JSON weights.

The exporter accepts a JSON source containing ``layers`` (weights and bias) and
quantizes each layer symmetrically to signed int8. A libtorch .pt cannot be
safely parsed by the restricted production interpreter, so it is rejected with
a clear message instead of being copied under a misleading deployment format.
"""
import argparse
import json
from pathlib import Path

META = {
    "format_version": 1,
    "actor_channels": 59,
    "patch": 17,
    "scalars": 40,
    "actor_input": 17091,
    "reward": [-1, 0, 1],
    "numpy": False,
}

def _quantize(layer):
    weights = layer.get("weights", [])
    bias = layer.get("bias", [])
    max_abs = max([abs(float(v)) for row in weights for v in row] + [abs(float(v)) for v in bias] + [1e-12])
    scale = max_abs / 127.0
    q = lambda v: max(-127, min(127, int(round(float(v) / scale))))
    return {"weights": [[q(v) for v in row] for row in weights], "bias": [q(v) for v in bias], "scale": scale}

def export_checkpoint(src, dst):
    source = Path(src)
    try:
        data = json.loads(source.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("export requires JSON float layers; use a tensor-to-JSON conversion step for .pt checkpoints") from exc
    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("source JSON must contain non-empty layers")
    payload = {"format": "florent-policy-int8-v1", "meta": {**META, **data.get("meta", {})}, "quantized_layers": [_quantize(x) for x in layers]}
    out = Path(dst)
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    out.with_name(out.name + ".json").write_text(json.dumps(payload["meta"], indent=2) + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("output")
    args = p.parse_args(); export_checkpoint(args.checkpoint, args.output)
