import torch


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
        if text_embeds.shape[0] != 1 or attention_mask.shape[0] != 1:
            raise ValueError("This branch requires exactly one fixed door handle prompt")
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
        vision = self.sam3.get_vision_features(pixel_values=pixel_values)
        outputs = self.sam3(
            vision_embeds=vision,
            text_embeds=self.text_embeds,
            attention_mask=self.attention_mask,
        )
        return self.select_outputs(outputs)
