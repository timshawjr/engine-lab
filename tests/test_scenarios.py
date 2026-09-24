from __future__ import annotations

import unittest
from pathlib import Path

from app.engine.device_policy import DeviceMode, DevicePolicy
from app.engine.pipelines import build_stage_assignments, load_scenario_catalog


ROOT = Path(__file__).resolve().parents[1]


class ScenarioCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_scenario_catalog(
            ROOT / "config" / "scenarios.json",
            ROOT / "config" / "models.json",
            ROOT / "media",
        )

    def test_exact_four_scenario_order(self) -> None:
        self.assertEqual(
            tuple(scenario.id for scenario in self.catalog.values()),
            ("retail", "smart_city", "medical", "gov_defense"),
        )

    def test_scenario_media_and_thresholds(self) -> None:
        for scenario in self.catalog.values():
            path = self.catalog.media_path(scenario.id, camera_index=None)
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())
            self.assertTrue(scenario.event_rules)
            self.assertGreaterEqual(len(scenario.ticker), 4)
            for zone in scenario.zones:
                x, y, width, height = zone.roi
                self.assertGreaterEqual(min(x, y, width, height), 0.0)
                self.assertLessEqual(x + width, 1.0)
                self.assertLessEqual(y + height, 1.0)

    def test_retail_stage_assignment(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        assignments = build_stage_assignments(policy, self.catalog["retail"], 0)
        by_stage = {assignment.stage: assignment for assignment in assignments}
        self.assertEqual(by_stage["detector"].requested_device, "NPU")
        self.assertEqual(by_stage["classifier"].requested_device, "GPU")
        self.assertEqual(by_stage["zone_event"].requested_device, "CPU")

    def test_split_and_disabled_preference_fallback(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPLIT)
        retail = self.catalog["retail"]
        assignments = {
            item.stage: item for item in build_stage_assignments(policy, retail, 0)
        }
        self.assertEqual(assignments["detector"].requested_device, "NPU")
        self.assertEqual(assignments["classifier"].requested_device, "GPU")
        policy.toggle_gpu()
        assignments = {
            item.stage: item for item in build_stage_assignments(policy, retail, 0)
        }
        self.assertEqual(assignments["classifier"].requested_device, "CPU")
        self.assertEqual(assignments["classifier"].intended_device, "GPU")

    def test_unavailable_model_device_uses_explicit_fallback(self) -> None:
        class FakeAvailability:
            @staticmethod
            def supports(model_id: str, device: str) -> bool:
                return not (model_id == "yolo11n-fp16" and device == "NPU")

        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        assignments = {
            item.stage: item
            for item in build_stage_assignments(
                policy,
                self.catalog["retail"],
                0,
                availability=FakeAvailability(),
            )
        }
        self.assertEqual(assignments["detector"].requested_device, "GPU")
        self.assertEqual(assignments["detector"].intended_device, "NPU")

    def test_medical_person_detector_prefers_gpu(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        assignments = {
            item.stage: item
            for item in build_stage_assignments(policy, self.catalog["medical"], 0)
        }
        self.assertEqual(assignments["person_detector"].requested_device, "GPU")
        self.assertEqual(assignments["pose"].requested_device, "NPU")

    def test_auto_uses_reduced_priority(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.AUTO)
        policy.toggle_npu()
        assignments = {
            item.stage: item for item in build_stage_assignments(
                policy,
                self.catalog["retail"],
                0,
            )
        }
        self.assertEqual(assignments["detector"].requested_device, "AUTO:GPU,CPU")


if __name__ == "__main__":
    unittest.main()
