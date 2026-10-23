"""Exports IFCRN once with dynamic batch/height/width (it is fully convolutional) for the memory sweeps."""
import os, torch
from models import IFCRN
os.makedirs("models", exist_ok=True); torch.manual_seed(0)
kw = dict(opset_version=17, input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "b", 2: "h", 3: "w"}, "y": {0: "b", 2: "h", 3: "w"}})
m, x = IFCRN().eval(), torch.randn(1, 3, 256, 256)
try: torch.onnx.export(m, x, "models/ifcrn_dynamic.onnx", dynamo=False, **kw)
except TypeError: torch.onnx.export(m, x, "models/ifcrn_dynamic.onnx", **kw)
