import json
import tempfile
import unittest
from pathlib import Path

from clause_miner import (
    Cue,
    ElementDef,
    ParsedJP,
    chunk_and_rank,
    load_elements,
    to_element_defs,
)


def make_parsed(paragraphs: list[str]) -> ParsedJP:
    return ParsedJP(
        title="",
        abstract="",
        claims=[],
        description=" ".join(paragraphs),
        paragraphs=paragraphs,
    )


class LoaderTests(unittest.TestCase):
    def test_backward_compat_promotes_term(self) -> None:
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

            self.assertEqual(defs[0].id, "E1")
            self.assertEqual(defs[0].cues[0].term, "alpha")
            self.assertEqual(defs[0].cues[0].synonyms, ["beta"])
            self.assertEqual(defs[1].cues[0].term, "gamma")
            self.assertEqual(defs[1].cues[0].synonyms, [])


class ScoringTests(unittest.TestCase):
    def test_cues_are_and_within_element(self) -> None:
        parsed = make_parsed(["セッションを開始する。ユーザに通知する。"])
        element = ElementDef(
            id="E1",
            weight=1.0,
            cues=[
                Cue(term="セッション", synonyms=["session"]),
                Cue(term="認証", synonyms=["authentication"]),
            ],
        )
        ranked = chunk_and_rank(parsed, [element], chunk_tokens=200, overlap_tokens=0, include_text=False)
        self.assertFalse(ranked["results"]["E1"], "Chunk lacking one cue must not satisfy element")

    def test_all_elements_must_satisfy_chunk(self) -> None:
        paragraphs = ["セッション管理を行い、認証も処理する。", "改ざん検出のために署名検証を行う。"]
        parsed = make_parsed(paragraphs)
        elements = [
            ElementDef(
                id="E1",
                weight=1.0,
                cues=[
                    Cue(term="セッション管理", synonyms=["session"]),
                    Cue(term="ユーザ認証", synonyms=["認証"]),
                ],
            ),
            ElementDef(
                id="E2",
                weight=1.0,
                cues=[Cue(term="改ざん検出", synonyms=["署名検証"])],
            ),
        ]
        ranked = chunk_and_rank(parsed, elements, chunk_tokens=400, overlap_tokens=0, include_text=False)
        self.assertTrue(ranked["results"]["E1"], "All cues satisfied -> should produce result")
        self.assertTrue(ranked["results"]["E2"], "All elements satisfied -> should produce result")

        # Remove second cue mention -> results should vanish
        parsed_missing = make_parsed(["セッション管理を行うが、改ざん検出はしない。"])
        ranked_missing = chunk_and_rank(parsed_missing, elements, chunk_tokens=400, overlap_tokens=0, include_text=False)
        self.assertFalse(ranked_missing["results"]["E1"])
        self.assertFalse(ranked_missing["results"]["E2"])

    def test_synonym_variant_selection(self) -> None:
        parsed = make_parsed(["The device includes an antenna module."])
        element = ElementDef(
            id="E1",
            weight=1.0,
            cues=[Cue(term="アンテナ", synonyms=["antenna", "アンテナ装置"])],
        )
        ranked = chunk_and_rank(parsed, [element], chunk_tokens=100, overlap_tokens=0, include_text=False)
        self.assertTrue(ranked["results"]["E1"])
        match = ranked["results"]["E1"][0]["matched"][0]
        self.assertEqual(match["variant"], "antenna")
        self.assertGreater(match["score"], 0.0)

    def test_matched_contains_all_cues(self) -> None:
        paragraphs = ["セッション管理とユーザ認証を同時に実行する。"]
        parsed = make_parsed(paragraphs)
        element = ElementDef(
            id="E1",
            weight=1.0,
            cues=[
                Cue(term="セッション管理", synonyms=["session"]),
                Cue(term="ユーザ認証", synonyms=["認証"]),
            ],
        )
        ranked = chunk_and_rank(parsed, [element], chunk_tokens=200, overlap_tokens=0, include_text=False)
        result = ranked["results"]["E1"][0]
        self.assertEqual(len(result["matched"]), 2)
        cue_terms = {m["cue_term"] for m in result["matched"]}
        self.assertEqual(cue_terms, {"セッション管理", "ユーザ認証"})
        self.assertIn("score_norm", result)


if __name__ == "__main__":
    unittest.main()
