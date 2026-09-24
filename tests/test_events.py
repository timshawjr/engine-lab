from __future__ import annotations

import unittest
from pathlib import Path

from app.engine.events import EventTracker
from app.engine.pipelines import load_scenario_catalog
from app.engine.stages import Detection, Keypoint


ROOT = Path(__file__).resolve().parents[1]


class EventTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_scenario_catalog(
            ROOT / "config" / "scenarios.json",
            ROOT / "config" / "models.json",
            ROOT / "media",
        )

    def test_retail_classification_and_pickup(self) -> None:
        tracker = EventTracker(self.catalog["retail"], 0)
        shelf = Detection(0.05, 0.25, 0.25, 0.55, "bowl", 0.8, "mixing bowl", 0.7)
        tracked, events = tracker.update((shelf,), now=100.0)
        self.assertEqual(tracked[0].track_id, 1)
        self.assertIn("object_classified", {event.type for event in events})
        transition = Detection(0.12, 0.30, 0.32, 0.60, "bowl", 0.8, "mixing bowl", 0.7)
        tracker.update((transition,), now=100.4)
        bridge = Detection(0.20, 0.50, 0.40, 0.80, "bowl", 0.8, "mixing bowl", 0.7)
        tracker.update((bridge,), now=100.6)
        moved = Detection(0.20, 0.60, 0.40, 0.90, "bowl", 0.8, "mixing bowl", 0.7)
        tracker.update((moved,), now=100.8)
        _, events = tracker.update((moved,), now=101.5)
        self.assertIn("object_picked_up", {event.type for event in events})

    def test_smart_city_zone_counts(self) -> None:
        tracker = EventTracker(self.catalog["smart_city"], 1)
        person = Detection(0.40, 0.20, 0.55, 0.80, "person", 0.9)
        _, events = tracker.update((person,), now=200.0)
        self.assertNotIn("person_counted", {event.type for event in events})
        _, events = tracker.update((person,), now=200.5)
        self.assertIn("person_counted", {event.type for event in events})
        vehicle = Detection(0.78, 0.20, 0.95, 0.80, "vehicle", 0.9)
        _, events = tracker.update((vehicle,), now=201.0)
        self.assertNotIn("vehicle_counted", {event.type for event in events})
        _, events = tracker.update((vehicle,), now=201.5)
        self.assertIn("vehicle_counted", {event.type for event in events})

    def test_plate_event(self) -> None:
        tracker = EventTracker(self.catalog["gov_defense"], 2)
        plate = Detection(0.4, 0.4, 0.6, 0.6, "license plate", 0.9)
        _, events = tracker.update((plate,), now=300.0)
        self.assertIn("plate_detected", {event.type for event in events})

    def test_medical_posture_alert_respects_dwell(self) -> None:
        tracker = EventTracker(self.catalog["medical"], 0)
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


if __name__ == "__main__":
    unittest.main()
