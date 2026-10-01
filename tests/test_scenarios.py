from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from app.engine.device_policy import DeviceMode, DevicePolicy
from app.engine.stages import Detection, LetterboxTransform, postprocess_ssd
from app.engine.pipelines import build_stage_assignments, load_scenario_catalog
from app.scenarios.retail import classification_candidates


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
            ("retail", "metro", "health", "federal"),
        )

    def test_every_model_has_a_task_label(self) -> None:
        payload = json.loads(
            (ROOT / "config" / "models.json").read_text(encoding="utf-8")
        )
        for model in payload["models"]:
            self.assertTrue(model.get("task_label"), model["id"])

    def test_every_vertical_uses_all_three_engines(self) -> None:
        for scenario in self.catalog.values():
            devices = {s.device_pref for s in scenario.stages}
            self.assertTrue(
                {"NPU", "GPU", "CPU"} <= devices, f"{scenario.id}: {devices}"
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

    def test_classifier_never_submits_person(self) -> None:
        """An ImageNet or vocabulary label for a person crop is always noise."""
        detections = (
            Detection(100.0, 50.0, 300.0, 380.0, "person", 0.95),
            Detection(500.0, 120.0, 560.0, 180.0, "bowl", 0.80),
        )
        selected = classification_candidates(
            detections,
            confidence_min=0.3,
            top_k=3,
            detector_labels={"person", "bowl", "banana"},
        )
        self.assertEqual([item.label for item in selected], ["bowl"])
        self.assertNotIn("person", {item.label for item in selected})

    def test_classifier_allowlist_fails_closed(self) -> None:
        """An unlisted detector class is never submitted.

        This is what stops the self-checkout machine being submitted as an
        ``oven``/``microwave`` and then named a grocery product.
        """
        detections = (Detection(100.0, 100.0, 300.0, 300.0, "microwave", 0.99),)
        self.assertEqual(
            classification_candidates(
                detections,
                confidence_min=0.3,
                top_k=3,
                detector_labels=("banana", "apple"),
            ),
            (),
        )

    def test_classifier_skips_sub_minimum_crop_size(self) -> None:
        """A crop too small to name is skipped rather than upscaled."""
        too_small = (Detection(500.0, 120.0, 540.0, 150.0, "banana", 0.99),)
        self.assertEqual(
            classification_candidates(
                too_small,
                confidence_min=0.3,
                top_k=3,
                detector_labels=("banana",),
                min_crop_pixels=48.0,
            ),
            (),
        )
        big_enough = (Detection(30.0, 600.0, 240.0, 900.0, "banana", 0.99),)
        self.assertEqual(
            len(
                classification_candidates(
                    big_enough,
                    confidence_min=0.3,
                    top_k=3,
                    detector_labels=("banana",),
                    min_crop_pixels=48.0,
                )
            ),
            1,
        )

    def test_retail_names_products_from_a_declared_vocabulary(self) -> None:
        """Retail detects real products and refines with CLIP, not ImageNet.

        The checkout footage is what this scenario is measured on:
        product-detection-0001 names the actual SKUs (``ruffles``, ``mtn_dew``,
        ...) that the COCO detector could only approximate (``apple``,
        ``suitcase``, ``oven``). CLIP then refines each product crop against
        the declared category vocabulary. The vocabulary is the contract, so
        only declared names can reach the overlay.
        """
        stages = {stage.stage: stage for stage in self.catalog["retail"].stages}
        self.assertIn("detector", stages)
        self.assertEqual(stages["detector"].model_id, "product-detection-0001")
        self.assertIn("product_classifier", stages)
        self.assertEqual(
            stages["product_classifier"].model_id, "clip-vision-patch32"
        )
        self.assertEqual(stages["product_classifier"].device_pref, "GPU")
        rules = self.catalog["retail"].event_rules
        self.assertNotIn("classification_min", rules)
        self.assertIn("zero_shot_min", rules)
        # person is a real shelf signal but must never be refined as a product.
        self.assertNotIn("person", rules["classify_detector_labels"])
        self.assertNotIn("person", rules["business_event_labels"])
        # Both gates carry the product-detection labels, so they admit exactly
        # the SKUs the detector can name. COCO names and the background/undefined
        # classes must not remain in the event rules.
        for product in ("ruffles", "mtn_dew", "best_foods", "sprite", "pringles"):
            self.assertIn(product, rules["classify_detector_labels"])
            self.assertIn(product, rules["business_event_labels"])
        for coco_name in ("apple", "banana", "suitcase", "oven", "bowl"):
            self.assertNotIn(coco_name, rules["classify_detector_labels"])
            self.assertNotIn(coco_name, rules["business_event_labels"])
        for excluded in ("background_label", "undefined"):
            self.assertNotIn(excluded, rules["classify_detector_labels"])
            self.assertNotIn(excluded, rules["business_event_labels"])
        # No sampling: the fast pipeline runs every frame at source rate.
        self.assertNotIn("detector_cadence", rules)

    def test_metro_types_vehicles_with_clip_not_imagenet(self) -> None:
        """Metro names vehicle crops with CLIP, not a weak ImageNet top-1.

        The crossroad detector already answers the counting question, and an
        ImageNet top-1 over a vehicle crop was only ever a weak candidate
        (``whistle``, ``carton``, ``fly``). The stage that was added is a
        zero-shot ``product_classifier`` that names a vehicle crop against a
        declared vocabulary, so the ImageNet-only fields stay absent.
        """
        stages = {stage.stage for stage in self.catalog["metro"].stages}
        self.assertNotIn("classifier", stages)
        self.assertIn("product_classifier", stages)
        classifier = next(
            stage
            for stage in self.catalog["metro"].stages
            if stage.stage == "product_classifier"
        )
        self.assertEqual(classifier.model_id, "clip-vision-patch32")
        self.assertEqual(classifier.device_pref, "GPU")
        rules = self.catalog["metro"].event_rules
        for removed in ("classify_output_labels", "classification_min"):
            self.assertNotIn(removed, rules)
        # The montage's crosswalks are full of pedestrians, and the declared
        # metro vocabulary carries a "person" entry, so person crops are
        # named too; vehicle crops remain the counting answer.
        self.assertEqual(rules["classify_detector_labels"], ["vehicle", "person"])
        # Retail and metro are the two CLIP scenarios; health and
        # federal name nothing.
        self.assertEqual(
            [
                s.id
                for s in self.catalog.values()
                if "product_classifier" in {x.stage for x in s.stages}
            ],
            ["retail", "metro"],
        )

    def test_classifier_gates_are_configured_on_every_scenario(self) -> None:
        for scenario in self.catalog.values():
            rules = scenario.event_rules
            if "classify_detector_labels" not in rules:
                continue
            # A label must be stable before it is shown, and a detector class
            # must be declared before its crop is submitted.
            self.assertGreaterEqual(int(rules["classify_min_frames"]), 2)
            self.assertTrue(rules["classify_detector_labels"])
            # person is banned from the ImageNet classifier path in code
            # (app/scenarios/retail.py), but metro runs the zero-shot CLIP
            # stage, whose declared vocabulary carries a "person" entry for the
            # montage's crosswalks.
            if scenario.id != "metro":
                self.assertNotIn("person", rules["classify_detector_labels"])
            self.assertIn("zero_shot_min", rules)
            self.assertGreater(float(rules["zero_shot_min"]), 0.0)

    def test_retail_stage_assignment(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        assignments = build_stage_assignments(policy, self.catalog["retail"], 0)
        by_stage = {assignment.stage: assignment for assignment in assignments}
        self.assertEqual(by_stage["detector"].requested_device, "NPU")
        self.assertEqual(
            by_stage["product_classifier"].requested_device, "GPU"
        )
        self.assertEqual(by_stage["zone_event"].requested_device, "CPU")

    def test_gpu_off_fails_gpu_stage_over_to_npu(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        policy.toggle_gpu()
        assignments = {
            item.stage: item
            for item in build_stage_assignments(policy, self.catalog["retail"], 0)
        }
        self.assertEqual(
            assignments["product_classifier"].requested_device, "NPU"
        )
        self.assertEqual(
            assignments["product_classifier"].intended_device, "GPU"
        )
        self.assertEqual(assignments["zone_event"].requested_device, "CPU")
        # The NPU-preferred detector must stay on the NPU.
        self.assertEqual(assignments["detector"].requested_device, "NPU")

    def test_npu_off_fails_npu_preferred_stage_to_gpu(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        policy.toggle_npu()
        health = self.catalog["health"]
        health_assignments = {
            item.stage: item
            for item in build_stage_assignments(policy, health, 0)
        }
        self.assertEqual(health_assignments["pose"].requested_device, "GPU")
        self.assertEqual(health_assignments["pose"].intended_device, "NPU")

    def test_gpu_and_npu_off_falls_back_to_cpu(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        policy.toggle_gpu()
        policy.toggle_npu()
        assignments = {
            item.stage: item
            for item in build_stage_assignments(policy, self.catalog["health"], 0)
        }
        self.assertEqual(assignments["person_detector"].requested_device, "CPU")
        self.assertEqual(assignments["pose"].requested_device, "CPU")

    def test_split_and_disabled_preference_fallback(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPLIT)
        health = self.catalog["health"]
        assignments = {
            item.stage: item for item in build_stage_assignments(policy, health, 0)
        }
        # Split keeps every device on, so each stage gets its own preference.
        self.assertEqual(assignments["person_detector"].requested_device, "GPU")
        self.assertEqual(assignments["pose"].requested_device, "NPU")
        # With the GPU off, a GPU-preferred stage moves to the NPU and records
        # what it originally asked for.
        policy.toggle_gpu()
        assignments = {
            item.stage: item for item in build_stage_assignments(policy, health, 0)
        }
        self.assertEqual(assignments["person_detector"].requested_device, "NPU")
        self.assertEqual(assignments["person_detector"].intended_device, "GPU")

    def test_unavailable_model_device_uses_explicit_fallback(self) -> None:
        class FakeAvailability:
            @staticmethod
            def supports(model_id: str, device: str) -> bool:
                return not (
                    model_id == "person-vehicle-bike-detection-crossroad-1016"
                    and device == "NPU"
                )

        policy = DevicePolicy(mode=DeviceMode.SPLIT, density=1)
        assignments = {
            item.stage: item
            for item in build_stage_assignments(
                policy,
                self.catalog["metro"],
                0,
                availability=FakeAvailability(),
            )
        }
        # metro's detector prefers NPU and the policy allows it, but this
        # model cannot run there, so it must fall back and still record what it
        # originally asked for.
        self.assertEqual(assignments["detector"].intended_device, "NPU")
        self.assertNotEqual(assignments["detector"].requested_device, "NPU")

    def test_health_person_detector_prefers_gpu(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        assignments = {
            item.stage: item
            for item in build_stage_assignments(policy, self.catalog["health"], 0)
        }
        self.assertEqual(assignments["person_detector"].requested_device, "GPU")
        self.assertEqual(assignments["pose"].requested_device, "NPU")

    def test_auto_uses_reduced_priority(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.AUTO)
        policy.toggle_npu()
        assignments = {
            item.stage: item for item in build_stage_assignments(
                policy,
                self.catalog["health"],
                0,
            )
        }
        self.assertEqual(
            assignments["person_detector"].requested_device, "AUTO:GPU,CPU"
        )

    def test_government_spreads_person_vehicle_and_plate(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD, density=1)
        assignments = {
            item.stage: item
            for item in build_stage_assignments(
                policy,
                self.catalog["federal"],
                0,
            )
        }
        # person-detection-retail-0013 detects nothing on the NPU (0 detections
        # over 499 frames in review; CPU/GPU both give 66), so the person
        # model moved to GPU and crossroad-1016 — measured at 303 detections
        # over 40 frames on the NPU — took the NPU slot. federal still
        # declares a stage on each of NPU, GPU and CPU.
        self.assertEqual(assignments["perimeter_detector"].requested_device, "GPU")
        self.assertEqual(assignments["vehicle_detector"].requested_device, "NPU")
        self.assertEqual(assignments["plate_detector"].requested_device, "GPU")
        policy.toggle_gpu()
        assignments = {
            item.stage: item
            for item in build_stage_assignments(
                policy,
                self.catalog["federal"],
                0,
            )
        }
        self.assertEqual(assignments["perimeter_detector"].requested_device, "NPU")
        self.assertEqual(assignments["vehicle_detector"].requested_device, "NPU")
        self.assertEqual(assignments["plate_detector"].requested_device, "NPU")

    def test_ssd_normalized_boxes_are_converted_to_source_pixels(self) -> None:
        output = np.array(
            [[[1.0, 1.0, 0.9, 0.25, 0.20, 0.50, 0.70]]],
            dtype=np.float32,
        )
        transform = LetterboxTransform(
            scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
            original_width=800,
            original_height=600,
            model_width=320,
            model_height=544,
        )
        detections = postprocess_ssd(
            output,
            labels={1: "person"},
            transform=transform,
            confidence_threshold=0.3,
            iou_threshold=0.5,
            max_detections=10,
        )
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].x1, 200.0)
        self.assertAlmostEqual(detections[0].y1, 120.0)
        self.assertAlmostEqual(detections[0].x2, 400.0)
        self.assertAlmostEqual(detections[0].y2, 420.0)


if __name__ == "__main__":
    unittest.main()
