import argparse
import json
import numpy as np
from pathlib import Path

import torch
from torch.onnx import symbolic_helper, symbolic_opset11
from PIL import Image

from fixed_prompt_wrapper import FixedPromptSam3Wrapper, encode_fixed_text_features


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path(r"C:\data\model\sam3")
SAMPLE_IMAGE = Path(r"C:\data\test.png")
IMAGE_SIZE = 672
PROMPTS = ("door handle",)
VERIFY_SHARED_OUTPUTS = True
OUTPUT_DIR = Path(__file__).resolve().parent / "onnx_weights_handle"


def load_model_and_processor(model_dir=MODEL_DIR):
    from transformers.models.sam3 import Sam3Config, Sam3Model, Sam3Processor

    config = Sam3Config.from_pretrained(model_dir, local_files_only=True)
    config.vision_config.backbone_config.image_size = IMAGE_SIZE

    model = Sam3Model.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()

    processor = Sam3Processor.from_pretrained(model_dir, local_files_only=True)
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
    prompts=PROMPTS,
):
    with torch.no_grad():
        shared = wrapper(pixel_values)
        shared = shared if isinstance(shared, tuple) else (shared,)

        for prompt_index, prompt in enumerate(prompts):
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


def compare_trt(output_json, reference_path, output_dir):
    """两类都比较，presence 分支错误不能以 engine 构建成功代替验证。"""
    records = json.loads(output_json.read_text(encoding="utf-8"))
    actual = {item["name"]: np.asarray(item["values"], dtype=np.float32) for item in records}
    report = {}
    with np.load(reference_path) as reference:
        for name in reference.files:
            expected = reference[name]
            values = actual[name].reshape(expected.shape)
            if not np.isfinite(values).all():
                raise ValueError(f"TRT non-finite output: {name}")
            rows = []
            for index in range(expected.shape[0]):
                difference = np.abs(values[index] - expected[index])
                row = {"class_index": index, "max_abs_error": float(difference.max()),
                       "mean_abs_error": float(difference.mean())}
                if name == "presence_logits":
                    row.update(reference=expected[index].tolist(), trt=values[index].tolist())
                if name == "pred_logits":
                    row.update(reference_top_query=int(expected[index].argmax()),
                               trt_top_query=int(values[index].argmax()))
                rows.append(row)
            report[name] = rows
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trt-comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="导出固定单/双类别 SAM3，或检查 TRT 数值")
    parser.add_argument("--mode", choices=("semantic", "instance"), default="semantic")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--classes", choices=("handle", "door-handle"), default="handle",
                        help="固定类别顺序；door-handle 为 door、door handle")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--image", type=Path, default=SAMPLE_IMAGE)
    parser.add_argument("--check-trt-output", type=Path,
                        help="只比较 trtexec --exportOutput JSON，不加载模型")
    parser.add_argument("--reference", type=Path, help="导出时生成的 reference.npz")
    args = parser.parse_args()
    if args.check_trt_output:
        if not args.reference:
            parser.error("--check-trt-output requires --reference")
        compare_trt(args.check_trt_output, args.reference, args.output_dir)
        return
    prompts = ("door", "door handle") if args.classes == "door-handle" else PROMPTS
    model, processor = load_model_and_processor(args.model_dir)

    image = Image.open(args.image).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    prompt_inputs = processor(text=list(prompts), return_tensors="pt").to(DEVICE)

    pixel_values = image_inputs["pixel_values"]
    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs["attention_mask"]

    with torch.no_grad():
        text_embeds = encode_fixed_text_features(model, input_ids, attention_mask)

    wrapper = FixedPromptSam3Wrapper(
        model,
        text_embeds,
        attention_mask,
        output_mode=args.mode,
    ).to(DEVICE).eval()

    print("prompts:", prompts)
    print("pixel_values:", tuple(pixel_values.shape))
    print("text_embeds:", tuple(wrapper.text_embeds.shape))

    if VERIFY_SHARED_OUTPUTS:
        verify_against_independent_runs(
            model,
            wrapper,
            pixel_values,
            input_ids,
            attention_mask,
            prompts,
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
    # 供 Jetson 用完全相同的浮点输入检查 TRT；不保存原图。
    pixel_values.detach().float().cpu().numpy().tofile(args.output_dir / "pixel_values.bin")
    np.savez_compressed(args.output_dir / "reference.npz", **{
        name: tensor.detach().float().cpu().numpy()
        for name, tensor in zip(wrapper.output_names, outputs)
    })
    contract = {"classes": list(prompts), "image_size": IMAGE_SIZE,
                "source_image": str(args.image.resolve()),
                "image_encoder_calls": 1, "outputs": {
                    name: list(tensor.shape) for name, tensor in zip(wrapper.output_names, outputs)}}
    (args.output_dir / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported to {onnx_path}")


if __name__ == "__main__":
    main()
