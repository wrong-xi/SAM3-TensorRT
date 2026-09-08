from pathlib import Path

import torch
from PIL import Image
from transformers.models.sam3 import Sam3Config, Sam3Model, Sam3Processor

from fixed_prompt_wrapper import FixedPromptSam3Wrapper
from onnx_tensorrt_fix import fix_onnx_file_for_tensorrt


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
        shared_semantic, shared_presence = wrapper(pixel_values)

        for prompt_index, prompt in enumerate(PROMPTS):
            reference = model(
                pixel_values=pixel_values,
                input_ids=input_ids[prompt_index : prompt_index + 1],
                attention_mask=attention_mask[prompt_index : prompt_index + 1],
            )
            torch.testing.assert_close(
                shared_semantic[prompt_index : prompt_index + 1],
                reference.semantic_seg,
                rtol=1e-3,
                atol=1e-3,
                msg=lambda msg: f"semantic logits differ for {prompt!r}: {msg}",
            )
            torch.testing.assert_close(
                shared_presence[prompt_index : prompt_index + 1],
                reference.presence_logits,
                rtol=1e-3,
                atol=1e-3,
                msg=lambda msg: f"presence logits differ for {prompt!r}: {msg}",
            )

    print("Fixed-prompt outputs match the independent full-model reference.")


def export_onnx(wrapper, pixel_values, onnx_path):
    torch.onnx.export(
        wrapper,
        (pixel_values,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["semantic_logits", "presence_logits"],
        dynamo=False,
        opset_version=17,
    )


def main():
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
        semantic_logits, presence_logits = wrapper(pixel_values)
    print("semantic_logits:", tuple(semantic_logits.shape))
    print("presence_logits:", tuple(presence_logits.shape))

    OUTPUT_DIR.mkdir(exist_ok=True)
    onnx_path = OUTPUT_DIR / "sam3_door_handle.onnx"
    export_onnx(wrapper, pixel_values, onnx_path)
    replacement_count = fix_onnx_file_for_tensorrt(onnx_path)
    print(
        "TensorRT compatibility fix:",
        f"replaced {replacement_count} decoder squeeze If node(s)",
    )
    print(f"Exported to {onnx_path}")


if __name__ == "__main__":
    main()
