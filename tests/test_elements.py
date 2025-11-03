import json
import tempfile
import unittest
from pathlib import Path

from clause_miner import (
    ElementDef,
    ParsedJP,
    chunk_and_rank,
    load_elements,
    to_element_defs,
)


class ElementLoaderTests(unittest.TestCase):
    def test_load_elements_backward_compat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "elements_old.json"
            payload = {
                "elements": [
                    {"id": "E1", "synonyms": ["alpha", "beta", "alpha"], "label": "ignored"},
                    {"id": "E2", "term": "gamma"},
                ]
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            records = load_elements(str(path))
            defs = to_element_defs(records)

            self.assertEqual(defs[0].term, "alpha")
            self.assertEqual(defs[0].synonyms, ["beta"])
            self.assertEqual(defs[0].role, "must")
            self.assertEqual(defs[1].term, "gamma")
            self.assertEqual(defs[1].synonyms, [])


class ScoringTests(unittest.TestCase):
    def test_must_not_excludes_chunk(self) -> None:
        parsed = ParsedJP(
            title="",
            abstract="",
            claims=[],
            description="foo feature with stateless only mode",
            paragraphs=["foo feature", "stateless only mode"],
        )
        elements = [
            ElementDef(id="M1", role="must", term="foo", synonyms=[], weight=1.0),
            ElementDef(id="NG", role="must_not", term="stateless", synonyms=["only"], weight=1.0),
        ]
        ranked = chunk_and_rank(parsed, elements, chunk_tokens=50, overlap_tokens=0, include_text=False)
        self.assertEqual(ranked["results"]["M1"], [])
        self.assertEqual(ranked["results"]["NG"], [])

    def test_synonym_variant_selection(self) -> None:
        parsed = ParsedJP(
            title="",
            abstract="",
            claims=[],
            description="The device includes an antenna module.",
            paragraphs=["The device includes an antenna module."],
        )
        elements = [
            ElementDef(
                id="E1",
                role="must",
                term="アンテナ",
                synonyms=["antenna", "アンテナ装置"],
                weight=1.0,
            )
        ]
        ranked = chunk_and_rank(parsed, elements, chunk_tokens=50, overlap_tokens=0, include_text=False)
        results = ranked["results"]["E1"]
        self.assertTrue(results, "Expected at least one result")
        self.assertEqual(results[0]["matched_variant"], "antenna")
        self.assertGreater(results[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
