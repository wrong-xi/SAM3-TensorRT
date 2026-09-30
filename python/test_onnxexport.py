import io
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers.models.sam3.modeling_sam3 import Sam3DecoderMLP

from onnxexport import ConvPresenceHead


class PresenceExportTests(unittest.TestCase):
    def test_conv_preserves_values_and_shape(self):
        torch.manual_seed(7)
        original = Sam3DecoderMLP(256, 256, 1, 3).eval()
        adapted = ConvPresenceHead(original).eval()
        for shape in ((1, 1, 256), (2, 3, 256)):
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


if __name__ == "__main__":
    unittest.main()
