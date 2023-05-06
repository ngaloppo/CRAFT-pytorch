import argparse
from collections import OrderedDict
from craft import CRAFT

import torch
import numpy as np

from openvino.runtime import PartialShape, Layout, Dimension, serialize
from openvino.tools.mo import convert_model


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="CRAFT Text Detection")
    parser.add_argument("trained_model", type=str, help="pretrained model")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model = CRAFT()

    print("Loading weights from checkpoint (" + args.trained_model + ")")
    model.load_state_dict(
        copyStateDict(torch.load(args.trained_model, map_location="cpu"))
    )

    model.eval()

    dynamic_axes = {"input": [0, 2, 3], "output": [0, 1, 2]}

    dummy_input = torch.randn(1, 3, 768, 768)

    model.eval()

    # Convert model to an OpenVINO model so that we can embed the pre-processing
    # into the model
    mean = np.array((0.485, 0.456, 0.406)) * 255.0
    variance = np.array((0.229, 0.224, 0.225)) * 255.0
    ov_model = convert_model(
        model,
        input_shape=PartialShape(
            [Dimension(), 3, Dimension(128,1280), Dimension(128,1280)]
        ),
        mean_values=mean,
        scale_values=variance,
        layout=Layout("NCHW"),
    )

    xml_path = "craft.xml"
    print(f"Exporting to {xml_path}...")
    serialize(ov_model, xml_path=xml_path)

    onnx_fn = "craft.onnx"
    print(f"Exporting to {onnx_fn}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_fn,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
