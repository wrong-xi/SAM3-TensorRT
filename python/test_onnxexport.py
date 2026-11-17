import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers.models.sam3.modeling_sam3 import Sam3DecoderMLP

from onnxexport import ConvPresenceHead, compare_trt
from fixed_prompt_wrapper import FixedPromptSam3Wrapper, encode_fixed_text_features


class PresenceExportTests(unittest.TestCase):
    def test_conv_preserves_values_and_shape(self):
        torch.manual_seed(7)
        original = Sam3DecoderMLP(256, 256, 1, 3).eval()
        adapted = ConvPresenceHead(original).eval()
        for shape in ((1, 1, 256), (2, 3, 256), (6, 2, 1, 256)):
            with self.subTest(shape=shape), torch.no_grad():
                features = torch.randn(shape)
                expected = original(features)
                torch.testing.assert_close(adapted(features), expected)

                buffer = io.BytesIO()
                torch.onnx.export(
                    adapted, (features,), buffer, dynamo=False,
                    opset_version=17, input_names=["features"],
                    output_names=["presence"],
                )
                model = onnx.load_model_from_string(buffer.getvalue())
                onnx.checker.check_model(model)
                # 完整 TRT 图中 Gemm 仍有异常，必须确实导出三层 Conv。
                operations = [node.op_type for node in model.graph.node]
                self.assertEqual(operations.count("Conv"), 3)
                self.assertNotIn("Gemm", operations)
                self.assertNotIn("MatMul", operations)
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = ort.InferenceSession(
                    buffer.getvalue(), sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                actual = session.run(None, {"features": features.numpy()})[0]
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(
                    actual, expected.numpy(), rtol=1e-4, atol=1e-6,
                )


class FakeSam3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_calls = 0

    def get_vision_features(self, pixel_values):
        self.encoder_calls += 1
        return pixel_values.mean().reshape(1, 1)

    def forward(self, vision_embeds, text_embeds, attention_mask):
        text = getattr(text_embeds, "pooler_output", text_embeds)
        value = vision_embeds + (text * attention_mask[..., None]).sum().reshape(1, 1)
        return SimpleNamespace(pred_masks=value.reshape(1,1,1,1).expand(1,2,2,2),
            pred_logits=value.expand(1,2), presence_logits=value,
            pred_boxes=value.reshape(1,1,1).expand(1,2,4),
            semantic_seg=value.reshape(1,1,1,1))


class DualPromptTests(unittest.TestCase):
    def test_shared_encoder_independent_classes_and_onnx(self):
        image = torch.ones(1,3,4,4)
        text = torch.tensor([[[1.,2.]], [[4.,8.]]])
        for structured in (False, True):
            with self.subTest(structured=structured):
                model = FakeSam3()
                embeds = SimpleNamespace(pooler_output=text) if structured else text
                wrapper = FixedPromptSam3Wrapper(model,embeds,torch.ones(2,1),"instance")
                outputs = wrapper(image)
                self.assertEqual(model.encoder_calls,1)
                self.assertEqual([tuple(x.shape) for x in outputs],
                                 [(2,2,2,2),(2,2),(2,1),(2,2,4)])
                for index in range(2):
                    single = FixedPromptSam3Wrapper(FakeSam3(),text[index:index+1],
                                                   torch.ones(1,1),"instance")
                    for actual, expected in zip(outputs,single(image)):
                        torch.testing.assert_close(actual[index:index+1],expected)
                buffer = io.BytesIO()
                torch.onnx.export(wrapper,(image,),buffer,dynamo=False,opset_version=17,
                    input_names=["pixel_values"],output_names=list(wrapper.output_names))
                onnx.checker.check_model(onnx.load_model_from_string(buffer.getvalue()))
                session = ort.InferenceSession(buffer.getvalue(),providers=["CPUExecutionProvider"])
                for actual, expected in zip(session.run(None,{"pixel_values":image.numpy()}),outputs):
                    np.testing.assert_allclose(actual,expected.numpy())

    def test_text_encoding_uses_independent_single_prompt_batches(self):
        class TextModel:
            def __init__(self, structured):
                self.structured, self.batches = structured, []

            def get_text_features(self, input_ids, attention_mask):
                self.batches.append(input_ids.shape[0])
                # 模拟 batch=2 数值变化；逐提示编码必须避开这条路径。
                value = input_ids[..., None].float() * attention_mask[..., None]
                if input_ids.shape[0] > 1:
                    value += .01
                return SimpleNamespace(pooler_output=value) if self.structured else value

        ids = torch.tensor([[1,2],[3,4]])
        mask = torch.tensor([[1,0],[1,1]])
        for structured in (False, True):
            for count in (1,2):
                with self.subTest(structured=structured, count=count):
                    model = TextModel(structured)
                    result = encode_fixed_text_features(model,ids[:count],mask[:count])
                    self.assertEqual(model.batches,[1]*count)
                    actual = result.pooler_output if structured else result
                    torch.testing.assert_close(actual,ids[:count,:,None].float()*mask[:count,:,None])

    def test_prompt_contract(self):
        with self.assertRaises(ValueError):
            FixedPromptSam3Wrapper(FakeSam3(),torch.ones(3,1,2),torch.ones(3,1))
        with self.assertRaises(ValueError):
            FixedPromptSam3Wrapper(FakeSam3(),torch.ones(2,1,2),torch.ones(1,1))

    def test_trt_comparison_keeps_both_presence_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/"reference.npz",presence_logits=np.array([[4.],[5.]],np.float32))
            (root/"output.json").write_text(json.dumps([
                {"name":"presence_logits","values":[4.01,5.02]}]))
            compare_trt(root/"output.json",root/"reference.npz",root/"report")
            report = json.loads((root/"report/trt-comparison.json").read_text())
            self.assertEqual(len(report["presence_logits"]),2)
            self.assertAlmostEqual(report["presence_logits"][1]["max_abs_error"],.02,places=5)


if __name__ == "__main__":
    unittest.main()
