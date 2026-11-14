"""Export the two pretrained MiniFASNet checkpoints to ONNX (one-off, needs torch).

Run:  .venv/Scripts/python scripts/export_minifas_onnx.py
"""
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party" / "minifas"
OUT = ROOT / "models"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("minifasnet_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    lib = load_module(SRC / "MiniFASNet.py")
    jobs = [
        ("2.7_80x80_MiniFASNetV2.pth", lib.MiniFASNetV2, "minifas_v2_2.7.onnx"),
        ("4_0_0_80x80_MiniFASNetV1SE.pth", lib.MiniFASNetV1SE, "minifas_v1se_4.0.onnx"),
    ]
    for ckpt, ctor, out_name in jobs:
        model = ctor(conv6_kernel=(5, 5))  # 80x80 input -> ((80+15)//16,)*2
        state = torch.load(SRC / ckpt, map_location="cpu", weights_only=True)
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        model.load_state_dict(state)
        model.eval()
        dummy = torch.zeros(1, 3, 80, 80)
        torch.onnx.export(
            model, dummy, str(OUT / out_name),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
            opset_version=17, dynamo=False,
        )
        print("wrote", OUT / out_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
