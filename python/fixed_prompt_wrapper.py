import torch


@torch.no_grad()
def encode_fixed_text_features(model, input_ids, attention_mask):
    """按独立参考相同的 batch=1 编码，避免双提示批处理的数值差异。"""
    features = [model.get_text_features(
        input_ids=input_ids[index:index + 1],
        attention_mask=attention_mask[index:index + 1],
    ) for index in range(input_ids.shape[0])]
    if isinstance(features[0], torch.Tensor):
        return torch.cat(features, dim=0)
    # 新版 HF 的固定文本接口返回 ModelOutput。
    return type(features[0])(
        pooler_output=torch.cat([value.pooler_output for value in features], dim=0))


class FixedPromptSam3Wrapper(torch.nn.Module):
    """复用固定文本特征，导出语义或未经筛选的实例输出。"""

    def __init__(
        self,
        sam3: torch.nn.Module,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_mode: str = "semantic",
    ) -> None:
        super().__init__()
        if output_mode not in ("semantic", "instance"):
            raise ValueError("output_mode must be semantic or instance")
        self.output_mode = output_mode
        # 新版 HF 返回带 pooler_output 的对象；旧版直接返回 Tensor。
        self.text_output_type = None if isinstance(text_embeds, torch.Tensor) else type(text_embeds)
        if self.text_output_type is not None:
            text_embeds = text_embeds.pooler_output
        if text_embeds.shape[0] not in (1, 2) or text_embeds.shape[0] != attention_mask.shape[0]:
            raise ValueError("Expected one handle prompt or two door/handle prompts")
        self.sam3 = sam3
        self.register_buffer("text_embeds", text_embeds)
        self.register_buffer("attention_mask", attention_mask)

    @property
    def output_names(self):
        return ("semantic_logits",) if self.output_mode == "semantic" else (
            "pred_masks", "pred_logits", "presence_logits", "pred_boxes"
        )

    def select_outputs(self, outputs):
        # 原始 logits 保留给 C++；筛选、缩放和身份关联不进入 ONNX。
        if self.output_mode == "semantic":
            return outputs.semantic_seg
        return tuple(getattr(outputs, name) for name in self.output_names)

    def forward(self, pixel_values: torch.Tensor):
        # 图像主干只运行一次。每条提示独立解码，避免跨提示 batch 广播改变原单类行为。
        vision = self.sam3.get_vision_features(pixel_values=pixel_values)
        results = []
        for index in range(self.text_embeds.shape[0]):
            text = self.text_embeds[index:index + 1]
            if self.text_output_type is not None:
                text = self.text_output_type(pooler_output=text)
            outputs = self.sam3(
                vision_embeds=vision, text_embeds=text,
                attention_mask=self.attention_mask[index:index + 1],
            )
            selected = self.select_outputs(outputs)
            results.append(selected if isinstance(selected, tuple) else (selected,))
        combined = tuple(torch.cat(parts, dim=0) for parts in zip(*results))
        return combined[0] if self.output_mode == "semantic" else combined
