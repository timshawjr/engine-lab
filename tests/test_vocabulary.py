"""Per-scenario CLIP vocabulary declaration and resolution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.engine.pipelines import load_scenario_catalog
from app.engine.stages import load_zero_shot_vocabulary
from tools.clip_vocabulary import RETAIL_VOCABULARY, VOCABULARIES


ROOT = Path(__file__).resolve().parents[1]


class VocabularyDeclarationTests(unittest.TestCase):
    def test_manufacturing_vocabulary_is_a_binary_compliance_question(self):
        """PPE is asked as compliant / not compliant, never as a garment list.

        A five-way vocabulary (hard_hat, safety_vest, work_overalls,
        safety_gloves, plain_clothes) measured 19 label changes in 60 sampled
        frames on the warehouse clip, which is the instability that got the
        ImageNet stage removed. The binary form measured 0 changes and 60/60
        correct. The test exists so a garment list cannot be reintroduced.
        """
        labels = set(VOCABULARIES["manufacturing"])
        self.assertEqual(labels, {"ppe_worn", "no_ppe"})
        for garment in ("hard_hat", "safety_vest", "work_overalls", "safety_gloves"):
            self.assertNotIn(garment, labels)

    def test_every_declared_scenario_vocabulary_is_baked_and_current(self):
        """A stage that declares a vocabulary needs its own baked pair.

        load_zero_shot_vocabulary() falls back to the default pair when a named
        one is missing, and the default is the retail grocery list. Without this
        check a machine that never ran the bake would put "mtn_dew" on a
        warehouse worker. The artifacts are gitignored, so this is a real
        provisioning check, not a formality.
        """
        catalog = load_scenario_catalog(
            ROOT / "config" / "scenarios.json",
            ROOT / "config" / "models.json",
            ROOT / "media",
        )
        checked = 0
        for scenario in catalog.values():
            for stage in scenario.stages:
                if not stage.vocabulary:
                    continue
                checked += 1
                name = stage.vocabulary
                model_dir = ROOT / "models" / stage.model_id
                index = model_dir / f"vocabulary_{name}.json"
                embeddings = model_dir / f"text_embeddings_{name}.npy"
                self.assertTrue(
                    index.is_file(),
                    f"{scenario.id}:{stage.stage} missing {index.name}; rebuild with "
                    f"<dev-venv>\\Scripts\\python tools\\build_clip_zero_shot.py "
                    f"--scenario {name}",
                )
                self.assertTrue(embeddings.is_file(), f"missing {embeddings.name}")
                baked = json.loads(index.read_text(encoding="utf-8"))
                self.assertEqual(
                    list(baked.get("labels", [])),
                    list(VOCABULARIES[name]),
                    f"{name} bake is stale relative to tools/clip_vocabulary.py; "
                    "rebuild it",
                )
        self.assertGreater(checked, 0, "no scenario declares a vocabulary")

    def test_retail_vocabulary_names_groceries(self):
        labels = set(VOCABULARIES["retail"])
        self.assertIn("bananas", labels)
        self.assertNotIn("mixing bowl", labels)

    def test_retail_vocabulary_has_no_kitchenware(self):
        """The checkout clip is groceries, so no dishware name may remain."""

        labels = set(VOCABULARIES["retail"])
        for kitchenware in ("pot", "mixing bowl", "storage container"):
            self.assertNotIn(kitchenware, labels)

    def test_metro_vocabulary_names_road_users(self):
        labels = set(VOCABULARIES["metro"])
        self.assertIn("car", labels)
        self.assertIn("person", labels)
        self.assertNotIn("mixing bowl", labels)

    def test_retail_alias_keeps_existing_imports_working(self):
        self.assertIs(RETAIL_VOCABULARY, VOCABULARIES["retail"])


class ZeroShotVocabularyLoadingTests(unittest.TestCase):
    DEFAULT_LABELS = ("default one", "default two")

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)
        self.path = self._write_pair(None, self.DEFAULT_LABELS)

    def _write_pair(self, name: str | None, labels: tuple[str, ...]) -> Path:
        """Bake a vocabulary pair the way tools/build_clip_zero_shot.py does."""

        suffix = f"_{name}" if name else ""
        embeddings = np.zeros((2 * len(labels), 4), dtype=np.float32)
        embeddings[:, 0] = 1.0
        np.save(self.directory / f"text_embeddings{suffix}.npy", embeddings)
        path = self.directory / f"vocabulary{suffix}.json"
        path.write_text(
            json.dumps(
                {
                    "checkpoint": "openai/clip-vit-base-patch32",
                    "labels": list(labels),
                    "template_counts": [2] * len(labels),
                    "templates_per_label": True,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_named_vocabulary_is_loaded_when_present(self):
        self._write_pair("retail", ("bananas", "bread"))
        _embeddings, labels, counts = load_zero_shot_vocabulary(
            self.path, name="retail"
        )
        self.assertEqual(labels, ("bananas", "bread"))
        self.assertEqual(counts, (2, 2))

    def test_missing_vocabulary_falls_back(self):
        _embeddings, labels, counts = load_zero_shot_vocabulary(
            self.path, name="does-not-exist"
        )
        self.assertTrue(labels)  # default used, no exception
        self.assertEqual(labels, self.DEFAULT_LABELS)
        self.assertEqual(counts, (2, 2))

    def test_name_none_uses_the_default(self):
        _embeddings, labels, _counts = load_zero_shot_vocabulary(self.path)
        self.assertEqual(labels, self.DEFAULT_LABELS)
        _embeddings, labels, _counts = load_zero_shot_vocabulary(
            self.path, name=None
        )
        self.assertEqual(labels, self.DEFAULT_LABELS)

    def test_half_missing_named_pair_falls_back(self):
        """A named index without its embeddings must not half-load."""

        self._write_pair("metro", ("car", "bus"))
        (self.directory / "text_embeddings_metro.npy").unlink()
        _embeddings, labels, _counts = load_zero_shot_vocabulary(
            self.path, name="metro"
        )
        self.assertEqual(labels, self.DEFAULT_LABELS)


class ScenarioVocabularyWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_scenario_catalog(
            ROOT / "config" / "scenarios.json",
            ROOT / "config" / "models.json",
            ROOT / "media",
        )

    def test_every_clip_stage_declares_a_known_vocabulary(self):
        declared = {
            stage.vocabulary
            for scenario in self.catalog.values()
            for stage in scenario.stages
            if stage.model_id == "clip-vision-patch32"
        }
        self.assertEqual(declared, {"retail", "metro", "manufacturing"})
        for name in declared:
            self.assertIn(name, VOCABULARIES)


if __name__ == "__main__":
    unittest.main()
