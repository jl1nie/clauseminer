# ClauseMiner CLI

ClauseMiner extracts evidence snippets from JP/US/WO patent publications. It parses a document, chunks the description, evaluates each chunk with BM25, and returns per-element matches that satisfy role-based filters (`must`, `should`, `must_not`). The text parser is exposed for tests and tooling; production inputs are provided as JSON.

## 1. Function & I/O Schema

### Workflow

1. Load patent text (UTF-8) from JSON input. A minimal request body is:
   ```json
   {
     "doc_id": "JP4743919B2",
     "text": "<publication body here>"
   }
   ```
   Use `parse_jpo_plaintext()` to split the text into title, abstract, claims, description, and paragraphs.
2. Load extraction elements via `load_elements(path)` and convert to internal definitions with `to_element_defs(...)`.
3. Rank chunks with `chunk_and_rank(...)`. For CLI usage:
   ```bash
   python clause_miner.py \
     --elements elements.json \
     --input input.txt \
     --output output.json
   ```
   (`--input`/`--output` accept files; omit to use stdin/stdout.)

### Elements JSON

`elements.json` must follow the new schema (legacy files are auto-upgraded):

| Field      | Required | Type        | Description / Statistical meaning |
|------------|----------|-------------|------------------------------------|
| `id`       | ✅        | string      | Stable identifier used as the result key. Must be unique. |
| `role`     | ❌ (default `must`) | string (`must` \| `should` \| `must_not`) | Drives eligibility: all `must` elements must hit (score ≥ τ), `should` contribute to final score, `must_not` filter chunks when score > 0. |
| `term`     | ✅/auto   | string      | Representative phrase used for scoring; if omitted the first synonym is promoted. |
| `synonyms` | ❌        | list[str]   | Additional variants evaluated independently; the maximum BM25 score is taken so variants do not double-count. Empty list allowed. |
| `weight`   | ❌ (default 1.0) | float | Multiplier applied to the raw BM25 element score before aggregation. Can be used to up/down-weight specific concepts; effectively a linear scaling factor. |
| `top_k`    | ❌        | int (>0)    | Per-element override for the number of top chunks to return. Falls back to global CLI `--top-k` otherwise. |

Loader rules:
- Missing `term` falls back to the first synonym.
- Missing `role` defaults to `must`.
- Synonyms are deduplicated; `label` (legacy) is ignored.
- Validation ensures every element has at least one non-empty phrase.

```json
{
  "elements": [
    {
      "id": "E1",
      "role": "must",
      "term": "セッション管理",
      "synonyms": ["session", "セッション", "セッショントークン"],
      "weight": 1.0,
      "top_k": 5
    },
    {
      "id": "NG1",
      "role": "must_not",
      "term": "完全にステートレス",
      "synonyms": ["stateless only"]
    }
  ]
}
```

### Output JSON

The CLI decides mode automatically:

*Long mode* (paragraphs ≥ 150 or description tokens > 60 000):
```json
{
  "meta": {
    "mode": "chunked",
    "num_chunks": 190,
    "normalize": "zscore",
    "tau_must": 0.0,
    "desc_paragraphs": 240,
    "desc_tokens": 112695,
    "chunk_params": {"chunk_tokens": 600, "overlap_tokens": 120}
  },
  "title": "...",
  "abstract": "...",
  "claims": ["...", "..."],
  "results": {
    "E1": [
      {
        "element_id": "E1",
        "role": "must",
        "element_term": "タッチスクリーンディスプレイ",
        "matched_variant": "タッチスクリーンディスプレイ",
        "chunk_idx": 14,
        "start_para": 15,
        "end_para": 16,
        "section": "課題/Problem to be Solved",
        "score": 24.67,
        "score_norm": 2.07,
        "chunk_score_total": 39.70,
        "chunk": "<chunk text>"
      }
    ],
    "NG1": []
  }
}
```

*Short/Mid mode* (below thresholds):
```json
{
  "meta": {
    "mode": "full",
    "desc_paragraphs": 42,
    "desc_tokens": 3800
  },
  "title": "...",
  "abstract": "...",
  "claims": ["...", "..."],
  "description": "..."
}
```

#### Field reference

**`meta`**
- `mode`: `"chunked"` when long-mode runs, `"full"` otherwise.
- `num_chunks`: count of generated chunks.
- `avgdl`: average document length used inside BM25 (sum of chunk term-frequency lengths divided by chunk count).
- `normalize`: normalization method selected (`none`, `zscore`, `minmax`).
- `tau_must`: threshold applied to `must` elements (default 0.0).
- `desc_paragraphs`, `desc_tokens`: statistics from the parsed description that drive mode selection.
- `chunk_params`: present in chunked mode; echoes chunk/overlap sizes.

**Per-element results (`results[element_id][]`)**

| Field | Description / Statistical meaning |
|-------|-----------------------------------|
| `element_id` | Echoes the source element ID. |
| `role` | Role used during eligibility (`must`/`should`). |
| `element_term` | Canonical phrase used for scoring. |
| `matched_variant` | Variant (term or synonym) that achieved the maximum BM25 score for this chunk. |
| `chunk_idx` | Index of the chunk within the processed document. |
| `start_para`, `end_para` | Paragraph boundaries (inclusive, zero-based). |
| `section` | Majority section label across the chunk’s paragraphs. |
| `score` | Weighted BM25 score: `BM25_max_variant × weight`. Higher is better. |
| `score_norm` | Normalized score using the configured method; with `zscore` it is `(score - μ) / σ` inside the element’s result list. |
| `chunk_score_total` | Sum of weighted scores over all `must` and `should` elements for the chunk; useful for ranking across elements. |
| `chunk` | (Optional) Raw chunk text, present when `--include-text` is `true` (default in chunked mode). |
| `chunk_highlight` | (Optional) Highlighted snippet when `--highlight` is enabled. |

## 2. Testing

### Test Suite

Run the built-in tests with the standard library runner (no extra deps):
```bash
python -m unittest tests.test_elements
```

`tests/test_elements.py` covers:
- Backward-compatible element loading.
- Must/must_not gating behaviour.
- Synonym variant selection.

### Preparing Test Data

1. Patent text JSON: place the source text (e.g., `JP4743919B2.txt`) under `tests/data/` or supply it inline when calling `parse_jpo_plaintext()`.
2. Elements JSON: use `tests/data/elements_jp4743919.json` as a template for new specs. Ensure each element has `id`, `role`, `term`, and optional `synonyms`, `weight`, `top_k`.
3. When adding new fixtures, keep them UTF-8 encoded and reference them from the tests to avoid network fetches.
4. For statistical verification, inspect `score`, `score_norm`, and `chunk_score_total` in the output to ensure scaling and normalization match expectations (`score_norm` will be a z-score unless `--normalize` overrides it).
