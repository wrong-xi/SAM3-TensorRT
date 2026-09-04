import torch
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput


def _repeat_prompt_batch(tensor, prompt_count):
    if tensor is None:
        return None
    return tensor.repeat((prompt_count,) + (1,) * (tensor.ndim - 1))


def _repeat_tensor_tuple(tensors, prompt_count):
    if tensors is None:
        return None
    return tuple(_repeat_prompt_batch(tensor, prompt_count) for tensor in tensors)


class FixedPromptSam3Wrapper(torch.nn.Module):
    """Run the image encoder once and evaluate all fixed prompts as one batch."""

    def __init__(self, sam3, text_embeds, attention_mask):
        super().__init__()
        self.sam3 = sam3
        self.register_buffer("text_embeds", text_embeds)
        self.register_buffer("attention_mask", attention_mask)

    def forward(self, pixel_values):
        vision = self.sam3.get_vision_features(pixel_values=pixel_values)
        prompt_count = self.text_embeds.shape[0]
        vision_batch = Sam3VisionEncoderOutput(
            last_hidden_state=_repeat_prompt_batch(
                vision.last_hidden_state, prompt_count
            ),
            fpn_hidden_states=_repeat_tensor_tuple(
                vision.fpn_hidden_states, prompt_count
            ),
            fpn_position_encoding=_repeat_tensor_tuple(
                vision.fpn_position_encoding, prompt_count
            ),
            hidden_states=_repeat_tensor_tuple(vision.hidden_states, prompt_count),
            attentions=_repeat_tensor_tuple(vision.attentions, prompt_count),
        )

        outputs = self.sam3(
            vision_embeds=vision_batch,
            text_embeds=self.text_embeds,
            attention_mask=self.attention_mask,
        )
        presence_logits = outputs.presence_logits.reshape(prompt_count, 1)
        return outputs.semantic_seg, presence_logits
