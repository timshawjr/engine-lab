from __future__ import annotations

import unittest

from app.engine.availability import AvailabilityMatrix, CompileResult, ModelAvailability


class AvailabilityMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = AvailabilityMatrix(
            generated_at="2026-09-24T00:00:00+00:00",
            fingerprint="abc123",
            cache_hit=False,
            models={
                "detector": ModelAvailability(
                    model_id="detector",
                    devices={
                        "NPU": CompileResult(True, ("NPU",), None, 10.0),
                        "GPU": CompileResult(False, (), "unsupported", None),
                        "CPU": CompileResult(True, ("CPU",), None, 20.0),
                    },
                )
            },
        )

    def test_cache_payload_round_trip(self) -> None:
        payload = self.matrix.to_dict()
        self.assertEqual(payload["schema"], 1)
        restored = AvailabilityMatrix.from_dict(payload, cache_hit=True)
        self.assertTrue(restored.cache_hit)
        self.assertTrue(restored.supports("detector", "NPU"))
        self.assertFalse(restored.supports("detector", "GPU"))
        self.assertTrue(restored.supports("detector", "AUTO:NPU,CPU"))

    def test_available_device_means_at_least_one_model(self) -> None:
        self.assertEqual(self.matrix.available_devices, ("NPU", "CPU"))


if __name__ == "__main__":
    unittest.main()
