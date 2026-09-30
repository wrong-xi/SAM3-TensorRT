import argparse
from pathlib import Path

import torch
from torch.onnx import symbolic_helper, symbolic_opset11
from PIL import Image
from transformers.models.sam3 import Sam3Config, Sam3Model, Sam3Processor

from fixed_prompt_wrapper import FixedPromptSam3Wrapper


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path(r"C:\data\model\sam3")
SAMPLE_IMAGE = Path(r"C:\data\test.png")
IMAGE_SIZE = 672
PROMPTS = ("door handle",)
VERIFY_SHARED_OUTPUTS = True
OUTPUT_DIR = Path(__file__).resolve().parent / "onnx_weights_handle"


def load_model_and_processor():
    config = Sam3Config.from_pretrained(MODEL_DIR, local_files_only=True)
    config.vision_config.backbone_config.image_size = IMAGE_SIZE

    model = Sam3Model.from_pretrained(
        MODEL_DIR,
        config=config,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()

    processor = Sam3Processor.from_pretrained(MODEL_DIR, local_files_only=True)
    processor.image_processor.size = {
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
    }
    return model, processor


def verify_against_independent_runs(
    model,
    wrapper,
    pixel_values,
    input_ids,
    attention_mask,
):
    with torch.no_grad():
        shared = wrapper(pixel_values)
        shared = shared if isinstance(shared, tuple) else (shared,)

        for prompt_index, prompt in enumerate(PROMPTS):
            reference = model(
                pixel_values=pixel_values,
                input_ids=input_ids[prompt_index : prompt_index + 1],
                attention_mask=attention_mask[prompt_index : prompt_index + 1],
            )
            expected = wrapper.select_outputs(reference)
            expected = expected if isinstance(expected, tuple) else (expected,)
            for name, actual, target in zip(wrapper.output_names, shared, expected):
                if not torch.isfinite(actual).all() or not torch.isfinite(target).all():
                    raise ValueError(f"Non-finite output: {name}")
                torch.testing.assert_close(
                    actual[prompt_index : prompt_index + 1], target,
                    rtol=1e-3, atol=1e-3,
                    msg=lambda msg: f"{name} differs for {prompt!r}: {msg}",
                )

    print("Fixed-prompt outputs match the independent full-model reference.")


def _static_squeeze(g, value, dim=None):
    # 本脚本只导出固定尺寸。使用原始 trace 的形状，避免 ONNX 中间
    # 形状推导丢失末维信息后生成 Squeeze/Identity 两种秩的 If。
    if dim is not None and symbolic_helper._is_constant(dim):
        axis = symbolic_helper._get_const(dim, "i", "dim")
        sizes = g.original_node.inputsAt(0).type().sizes()
        if sizes is not None and -len(sizes) <= axis < len(sizes):
            size = sizes[axis]
            if size == 1:
                return symbolic_helper._squeeze_helper(g, value, [axis])
            if size is not None:
                return value
    return symbolic_opset11.squeeze(g, value, dim)


class ConvPresenceHead(torch.nn.Module):
    """用等价的 1×1 Conv 导出 presence MLP，绕过 TRT 10.3 数值异常。"""

    def __init__(self, head: torch.nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 每个 token 独立计算；Linear 的 [out, in] 权重只增加卷积核维度。
        output = features.reshape(-1, features.shape[-1], 1)
        layers = (self.head.layer1, self.head.layer2, self.head.layer3)
        for index, layer in enumerate(layers):
            output = torch.nn.functional.conv1d(
                output, layer.weight.unsqueeze(-1), layer.bias,
            )
            if index < 2:
                output = torch.nn.functional.relu(output)
        return output.reshape(*features.shape[:-1], output.shape[1])


@torch.no_grad()
def export_onnx(wrapper, pixel_values, onnx_path):
    decoder = wrapper.sam3.detr_decoder
    original_head = decoder.presence_head
    torch.onnx.register_custom_op_symbolic("aten::squeeze", _static_squeeze, 17)
    try:
        if wrapper.output_mode == "instance":
            reference = wrapper(pixel_values)
            decoder.presence_head = ConvPresenceHead(original_head).eval()
            for name, actual, expected in zip(
                wrapper.output_names, wrapper(pixel_values), reference
            ):
                torch.testing.assert_close(
                    actual, expected, rtol=1e-3, atol=1e-3,
                    msg=lambda msg, name=name: f"Conv presence changes {name}: {msg}",
                )
            print("Conv presence head matches the original model outputs.")
        torch.onnx.export(
            wrapper,
            (pixel_values,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=list(wrapper.output_names),
            dynamo=False,
            opset_version=17,
        )
    finally:
        decoder.presence_head = original_head
        torch.onnx.unregister_custom_op_symbolic("aten::squeeze", 17)


def main():
    parser = argparse.ArgumentParser(description="导出固定 door handle 提示的 SAM3")
    parser.add_argument("--mode", choices=("semantic", "instance"), default="semantic")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    model, processor = load_model_and_processor()

    image = Image.open(SAMPLE_IMAGE).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    prompt_inputs = processor(text=list(PROMPTS), return_tensors="pt").to(DEVICE)

    pixel_values = image_inputs["pixel_values"]
    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs["attention_mask"]

    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    wrapper = FixedPromptSam3Wrapper(
        model,
        text_embeds,
        attention_mask,
        output_mode=args.mode,
    ).to(DEVICE).eval()

    print("prompts:", PROMPTS)
    print("pixel_values:", tuple(pixel_values.shape))
    print("text_embeds:", tuple(text_embeds.shape))

    if VERIFY_SHARED_OUTPUTS:
        verify_against_independent_runs(
            model,
            wrapper,
            pixel_values,
            input_ids,
            attention_mask,
        )

    with torch.no_grad():
        outputs = wrapper(pixel_values)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    for name, tensor in zip(wrapper.output_names, outputs):
        print(f"{name}:", tuple(tensor.shape))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / f"sam3_door_handle_{args.mode}.onnx"
    with torch.no_grad():
        export_onnx(wrapper, pixel_values, onnx_path)
    print(f"Exported to {onnx_path}")


if __name__ == "__main__":
    main()
