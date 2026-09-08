import torch


class FixedPromptSam3Wrapper(torch.nn.Module):
    """Evaluate one fixed prompt using precomputed text features."""

    def __init__(
        self,
        sam3: torch.nn.Module,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        if text_embeds.shape[0] != 1 or attention_mask.shape[0] != 1:
            raise ValueError("This branch requires exactly one fixed door handle prompt")
        self.sam3 = sam3
        self.register_buffer("text_embeds", text_embeds)
        self.register_buffer("attention_mask", attention_mask)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vision = self.sam3.get_vision_features(pixel_values=pixel_values)
        outputs = self.sam3(
            vision_embeds=vision,
            text_embeds=self.text_embeds,
            attention_mask=self.attention_mask,
        )
        presence_logits = outputs.presence_logits.reshape(1, 1)
        return outputs.semantic_seg, presence_logits
