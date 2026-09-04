from copy import deepcopy
from pathlib import Path
import re

import onnx
from onnx import helper, numpy_helper


DECODER_IF_NAME = re.compile(r"^/sam3/detr_decoder/If(?:_\d+)?$")


def _graph_attribute(node, name):
    return next(
        (attribute.g for attribute in node.attribute if attribute.name == name),
        None,
    )


def _make_squeeze_replacement(node):
    if len(node.output) != 1:
        raise ValueError(f"Expected one output from {node.name}")

    then_branch = _graph_attribute(node, "then_branch")
    else_branch = _graph_attribute(node, "else_branch")
    if then_branch is None or else_branch is None:
        raise ValueError(f"{node.name} does not contain both If branches")

    squeeze_nodes = [item for item in then_branch.node if item.op_type == "Squeeze"]
    identity_nodes = [item for item in else_branch.node if item.op_type == "Identity"]
    if len(squeeze_nodes) != 1 or len(identity_nodes) != 1:
        raise ValueError(f"Unexpected decoder squeeze pattern in {node.name}")

    squeeze = squeeze_nodes[0]
    identity = identity_nodes[0]
    if len(squeeze.input) != 2 or squeeze.input[0] != identity.input[0]:
        raise ValueError(f"Decoder If branches use different inputs in {node.name}")
    if (
        len(then_branch.output) != 1
        or len(else_branch.output) != 1
        or then_branch.output[0].name != squeeze.output[0]
        or else_branch.output[0].name != identity.output[0]
    ):
        raise ValueError(f"Unexpected decoder If branch outputs in {node.name}")

    axes_node = next(
        (
            item
            for item in then_branch.node
            if squeeze.input[1] in item.output and item.op_type == "Constant"
        ),
        None,
    )
    if axes_node is None:
        raise ValueError(f"Squeeze axes are not constant in {node.name}")

    axes_attribute = next(
        (attribute for attribute in axes_node.attribute if attribute.name == "value"),
        None,
    )
    if axes_attribute is None or numpy_helper.to_array(
        helper.get_attribute_value(axes_attribute)
    ).reshape(-1).tolist() != [-1]:
        raise ValueError(f"Expected a final-axis squeeze in {node.name}")

    axes_copy = deepcopy(axes_node)
    squeeze_copy = deepcopy(squeeze)
    axes_output = f"{node.output[0]}_trt_axes"
    axes_copy.name = f"{node.name}_trt_axes"
    axes_copy.output[0] = axes_output
    squeeze_copy.name = f"{node.name}_trt_squeeze"
    squeeze_copy.input[1] = axes_output
    squeeze_copy.output[0] = node.output[0]
    return axes_copy, squeeze_copy


def replace_decoder_squeeze_ifs(model):
    replacement_count = 0
    nodes = []
    for node in model.graph.node:
        if node.op_type == "If" and DECODER_IF_NAME.fullmatch(node.name):
            nodes.extend(_make_squeeze_replacement(node))
            replacement_count += 1
        else:
            nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return replacement_count


def fix_onnx_file_for_tensorrt(onnx_path):
    onnx_path = Path(onnx_path)
    model = onnx.load_model(onnx_path, load_external_data=False)
    replacement_count = replace_decoder_squeeze_ifs(model)
    if replacement_count == 0:
        return 0

    temporary_path = onnx_path.with_name(f"{onnx_path.stem}.trt_tmp.onnx")
    try:
        onnx.save_model(model, temporary_path)
        onnx.checker.check_model(str(temporary_path))
        temporary_path.replace(onnx_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return replacement_count
