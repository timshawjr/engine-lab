from __future__ import annotations

import unittest
from pathlib import Path

from app.engine.events import EventTracker
from app.engine.pipelines import load_scenario_catalog
from app.engine.stages import Detection, Keypoint


ROOT = Path(__file__).resolve().parents[1]

#: Retail runs the 720x404 store-aisle clip. Zone ROIs are normalized, so the
#: tracker must be told the source size to test membership at all.
RETAIL_SIZE = (720, 404)
#: Smart city runs the 768x432 traffic montage.
SOURCE_SIZE = (768, 432)


class EventTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_scenario_catalog(
            ROOT / "config" / "scenarios.json",
            ROOT / "config" / "models.json",
            ROOT / "media",
        )

    def test_retail_emits_the_declared_product_name(self) -> None:
        """Retail names the item from the declared vocabulary, not COCO.

        The measured case: YOLO reports ``bowl`` on every pass, but the crop is a
        stainless steel saucepan and CLIP names it ``pot``. The event stream must
        carry ``pot``.
        """
        tracker = EventTracker(self.catalog["retail"], 0)
        rules = self.catalog["retail"].event_rules
        stable_frames = rules["classify_min_frames"]
        # 720x404 aisle source; this box centre lands inside the right_shelf zone.
        on_shelf = Detection(430, 80, 500, 200, "bowl", 0.8, "pot", 0.7)

        emitted: list[str] = []
        labels: list[str] = []
        for index in range(stable_frames):
            tracked, events = tracker.update(
                (on_shelf,), now=100.0 + index * 0.2, source_size=RETAIL_SIZE
            )
            emitted.extend(event.type for event in events)
            labels.extend(event.label for event in events)
        self.assertEqual(tracked[0].track_id, 1)
        self.assertIn("object_classified", emitted)
        # The event carries the declared product name, not the COCO class.
        self.assertIn("pot", labels)
        self.assertNotIn("bowl", labels)

    def test_unstable_classification_is_never_surfaced(self) -> None:
        """A label that changes every frame must never reach the operator.

        This is the measured retail defect: a stationary bowl flipped through
        Chihuahua / crash_helmet / chest 239 times in 40 seconds.
        """
        tracker = EventTracker(self.catalog["retail"], 0)
        stable_frames = self.catalog["retail"].event_rules["classify_min_frames"]
        noisy = ["pot", "plate", "cup", "glass", "vase", "kettle"]
        surfaced: list[str] = []
        events: list[str] = []
        for index, label in enumerate(noisy * 3):
            box = (430, 80, 500, 200)
            tracked, batch = tracker.update(
                (Detection(*box, "bowl", 0.8, label, 0.9),),
                now=100.0 + index * 0.05, source_size=RETAIL_SIZE,
            )
            events.extend(event.type for event in batch)
            surfaced.extend(item.classification for item in tracked)
        # No churning label may ever be shown, and none may be emitted.
        self.assertNotIn("object_classified", events)
        self.assertEqual([value for value in surfaced if value], [])
        self.assertGreaterEqual(len(noisy), stable_frames)

    def test_classification_emitted_once_per_label_per_track(self) -> None:
        """A stable label emits one event, not one per cooldown window."""
        tracker = EventTracker(self.catalog["retail"], 0)
        stable_frames = self.catalog["retail"].event_rules["classify_min_frames"]
        emitted: list[str] = []
        for index in range(stable_frames + 30):
            box = (430, 80, 500, 200)
            _, events = tracker.update(
                (Detection(*box, "bowl", 0.8, "pot", 0.8),),
                now=100.0 + index * 1.0, source_size=RETAIL_SIZE,
            )
            emitted.extend(event.type for event in events)
        self.assertEqual(emitted.count("object_classified"), 1)

    def test_zone_count_emits_once_per_track(self) -> None:
        """A track dwelling in a lane is counted once, not every frame.

        Regression: the counter had no per-track guard and re-fired each frame,
        producing 2,459 ``vehicle_counted`` events in 25 seconds.
        """

        size = (768, 432)
        tracker = EventTracker(self.catalog["metro"], 1)
        # centre (650, 380) -> (0.846, 0.880): inside outgoing_lane [0.0, 0.75, 1.0, 0.25]
        vehicle = Detection(600, 310, 700, 450, "vehicle", 0.9)
        emitted = 0
        for index in range(40):
            _, events = tracker.update((vehicle,), now=300.0 + index * 0.05, source_size=size)
            emitted += sum(1 for event in events if event.type == "vehicle_counted")
        self.assertEqual(emitted, 1)

    def test_machinery_false_positives_cannot_raise_business_events(self) -> None:
        """Regression: YOLO reports shelf hardware as household objects.

        Measured on the retail footage the detector produced ``microwave``,
        ``oven`` and ``tv`` on what is shelf hardware. Those classes are not in
        the declared inventory, so they must never raise a retail business event.
        """

        on_shelf = (430, 80, 500, 200)
        for label in ("microwave", "oven", "tv", "laptop", "toilet"):
            fresh = EventTracker(self.catalog["retail"], 0)
            _, events = fresh.update(
                (Detection(*on_shelf, label, 0.9),),
                now=100.0,
                source_size=RETAIL_SIZE,
            )
            self.assertNotIn(
                "object_classified",
                {event.type for event in events},
                f"{label} must not raise a retail classification event",
            )
        # A declared product class still does.
        allowed = EventTracker(self.catalog["retail"], 0)
        for index in range(self.catalog["retail"].event_rules["classify_min_frames"]):
            _, events = allowed.update(
                (Detection(*on_shelf, "bowl", 0.9, "pot", 0.9),),
                now=100.0 + index * 0.2,
                source_size=RETAIL_SIZE,
            )
        self.assertIn("object_classified", {event.type for event in events})

    def test_retail_event_label_is_the_plain_product_name(self) -> None:
        """Retail must not prefix the label; it comes from a declared vocabulary."""
        tracker = EventTracker(self.catalog["retail"], 0)
        stable_frames = self.catalog["retail"].event_rules["classify_min_frames"]
        events: list[BusinessEvent] = []
        for index in range(stable_frames):
            box = (30, 600, 240, 900)
            box = (430, 80, 500, 200)
            _, batch = tracker.update(
                (Detection(*box, "bowl", 0.8, "pot", 0.6),),
                now=100.0 + index * 0.2,
                source_size=RETAIL_SIZE,
            )
            events.extend(batch)
        labels = [event.label for event in events if event.type == "object_classified"]
        self.assertTrue(labels)
        self.assertEqual(labels[0], "pot")
        self.assertNotIn("candidate", labels[0])

    def test_zone_membership_uses_pixel_coordinates(self) -> None:
        """Regression: zone ROIs are normalized, detections are in pixels.

        Comparing a pixel centre straight against a 0..1 ROI made every zone
        test fail, so ``person_counted`` and ``vehicle_counted`` could never
        fire in the live app. These boxes are real 768x432 traffic pixels and
        must be matched.
        """

        size = (768, 432)
        tracker = EventTracker(self.catalog["metro"], 1)
        # centre (380, 210) -> (0.495, 0.486): inside the crosswalk band [0.0, 0.3, 1.0, 0.5]
        person = Detection(330, 80, 430, 340, "person", 0.9)
        tracker.update((person,), now=200.0, source_size=size)
        _, events = tracker.update((person,), now=200.5, source_size=size)
        self.assertIn("person_counted", {event.type for event in events})
        # centre (650, 380) -> (0.846, 0.880): inside outgoing_lane [0.0, 0.75, 1.0, 0.25]
        vehicle = Detection(600, 310, 700, 450, "vehicle", 0.9)
        tracker.update((vehicle,), now=201.0, source_size=size)
        _, events = tracker.update((vehicle,), now=201.5, source_size=size)
        self.assertIn("vehicle_counted", {event.type for event in events})

    def test_zone_membership_needs_a_source_size(self) -> None:
        """Without a frame size the tracker must not claim a zone hit."""

        tracker = EventTracker(self.catalog["metro"], 1)
        person = Detection(330, 80, 430, 340, "person", 0.9)
        tracker.update((person,), now=200.0)
        _, events = tracker.update((person,), now=200.5)
        self.assertNotIn("person_counted", {event.type for event in events})

    def test_plate_event(self) -> None:
        tracker = EventTracker(self.catalog["federal"], 2)
        plate = Detection(0.4, 0.4, 0.6, 0.6, "license plate", 0.9)
        _, events = tracker.update((plate,), now=300.0, source_size=SOURCE_SIZE)
        self.assertIn("plate_detected", {event.type for event in events})

    def test_medical_posture_alert_respects_dwell(self) -> None:
        tracker = EventTracker(self.catalog["health"], 0)
        person = Detection(0.3, 0.2, 0.7, 0.9, "person", 0.9)
        keypoints = (
            Keypoint("left_shoulder", 0.3, 0.4, 0.9),
            Keypoint("right_shoulder", 0.7, 0.4, 0.9),
            Keypoint("left_hip", 0.3, 0.6, 0.9),
            Keypoint("right_hip", 0.7, 0.6, 0.9),
        )
        person = person.with_keypoints(keypoints)
        _, events = tracker.update((person,), now=400.0, posture_angles=(90.0,))
        self.assertNotIn("posture_alert", {event.type for event in events})
        _, events = tracker.update((person,), now=401.0, posture_angles=(90.0,))
        self.assertIn("posture_alert", {event.type for event in events})

    def test_tracker_reconnects_after_motion_and_one_dropout(self) -> None:
        tracker = EventTracker(self.catalog["retail"], 0)
        first = Detection(0.10, 0.10, 0.30, 0.30, "bowl", 0.9)
        second = Detection(0.15, 0.10, 0.35, 0.30, "bowl", 0.9)
        reconnected = Detection(0.20, 0.10, 0.40, 0.30, "bowl", 0.9)
        tracked, _ = tracker.update((first,), now=100.0)
        track_id = tracked[0].track_id
        tracked, _ = tracker.update((second,), now=100.1)
        self.assertEqual(tracked[0].track_id, track_id)
        tracker.update((), now=100.2)
        tracked, _ = tracker.update((reconnected,), now=100.3)
        self.assertEqual(tracked[0].track_id, track_id)


if __name__ == "__main__":
    unittest.main()
