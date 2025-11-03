# -*- coding: utf-8 -*-
"""
Evidence Pipeline — full CLI (JP patent oriented)
- Parser: JP publication text -> title / abstract / claims / description / paragraphs
- Chunker: paragraph-aware chunking with target size + overlap
- Scorer: BM25 (synonyms-max per element), per-element normalization
- Highlight: optional (disabled by default)
- MAIN policy:
    * LONG docs (>=150 paragraphs or >60,000 tokens):
        - Return RESULTS ONLY (no raw description, no chunk list in top-level),
          BUT each result includes the matched chunk TEXT.
    * SHORT/MID docs:
        - Return RAW DESCRIPTION ONLY (no results).
- CLI: stdin/stdout or --input/--output. See usage below.

Usage
-----
python evidence_pipeline_full_auto.py \
  --elements elements.json \
  --chunk-tokens 600 --overlap-tokens 120 --top-k 3 \
  --normalize zscore \
  --input input.txt --output output.json

Or with stdin/stdout:
cat input.txt | python evidence_pipeline_full_auto.py --elements elements.json > output.json
"""

from __future__ import annotations
import math, re, unicodedata, json, sys, argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from pathlib import Path

# ------------------------------
# Text normalization
# ------------------------------
def normalize_text(text: str) -> str:
    """NFKC, collapse whitespace, lowercase ASCII (JP left as-is)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    # control chars -> space, collapse whitespace
    t = re.sub(r"[\u0000-\u001F\u007F]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # lowercase ASCII; make sure '-' is last inside class to avoid ranges
    t = re.sub(r"[A-Z0-9%./-]+", lambda m: m.group(0).lower(), t)
    return t

# ------------------------------
# Parser (JP heuristics)
# ------------------------------
SECTION_PATTERNS = {
    "title": [r"【?発明の名称】?", r"【?名称】?", r"\btitle\b"],
    "abstract": [r"【?要約】?", r"【?概要】?", r"\babstract\b"],
    "claims": [r"【?特許請求の範囲】?", r"【?請求項】?", r"特許請求の範囲", r"\bclaims?\b"],
    "description": [r"【?発明の詳細な説明】?", r"発明の詳細な説明", r"【?詳細な説明】?", r"\bdetailed description\b"],
}
CLAIM_ITEM_PATTERNS = [
    r"【請求項\s*([0-9０-９]+)】",
    r"^\s*(?:claim|clause)\s*(\d+)\b",
]

@dataclass
class ParsedJP:
    title: str
    abstract: str
    claims: List[str]
    description: str
    paragraphs: List[str]

def _locate_sections(text: str) -> Dict[str, List[Tuple[int, re.Match]]]:
    hits: Dict[str, List[Tuple[int, re.Match]]] = {k: [] for k in SECTION_PATTERNS}
    for sec, pats in SECTION_PATTERNS.items():
        for pat in pats:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                hits[sec].append((m.start(), m))
        hits[sec].sort(key=lambda x: x[0])
    return hits

def parse_jpo_plaintext(raw_text: str) -> ParsedJP:
    """Split JP publication plain text into sections and paragraphs."""
    text = normalize_text(raw_text)
    text = text.replace("\ufeff", "")
    marks = _locate_sections(text)

    # Drop abstract markers that appear after claims/description headers (false positives).
    guard_candidates = []
    for sec in ("claims", "description"):
        if marks[sec]:
            guard_candidates.append(marks[sec][0][0])
    guard = min(guard_candidates) if guard_candidates else None
    if guard is not None and marks["abstract"]:
        marks["abstract"] = [(pos, m) for pos, m in marks["abstract"] if pos < guard]

    # unify all markers to compute section boundaries
    all_marks: List[Tuple[int, str]] = []
    for sec, arr in marks.items():
        for pos, _m in arr:
            all_marks.append((pos, sec))
    all_marks.sort(key=lambda x: x[0])

    def first(sec: str) -> int | None:
        return marks[sec][0][0] if marks[sec] else None

    def slice_between(a: int, b: int | None) -> str:
        return text[a:b].strip() if b is not None else text[a:].strip()

    # Title
    title = ""
    ti = first("title")
    if ti is not None:
        seg = text[ti: ti + 200]
        seg = re.sub(r"^.*?(発明の名称|名称|title)】?\s*", "", seg, flags=re.IGNORECASE)
        title = seg.split(" ", 1)[0][:120]
    else:
        title = text[:120].split(" ", 1)[0]

    # Abstract
    abstract = ""
    ai = first("abstract")
    if ai is not None:
        next_after = next((pos for pos, _sec in all_marks if pos > ai), None)
        body = slice_between(ai, next_after)
        abstract = re.sub(r"^.*?(要約|概要|abstract)】?\s*", "", body, flags=re.IGNORECASE).strip()

    # Claims
    claims_block = ""
    ci = first("claims")
    if ci is not None:
        end_at = next((pos for pos, sec in all_marks if pos > ci and sec in ("description", "abstract", "title")), None)
        claims_block = slice_between(ci, end_at)
    claims: List[str] = []
    if claims_block:
        items: List[Tuple[int, re.Match]] = []
        for pat in CLAIM_ITEM_PATTERNS:
            for m in re.finditer(pat, claims_block, flags=re.IGNORECASE | re.MULTILINE):
                items.append((m.start(), m))
        items.sort(key=lambda x: x[0])
        if items:
            for i, (pos, _m) in enumerate(items):
                endp = items[i + 1][0] if i + 1 < len(items) else None
                seg = claims_block[pos:endp]
                seg = re.sub(r"^【?請求項\s*[0-9０-９]+】?\s*", "", seg)
                claims.append(seg.strip())
        else:
            # Fallback: split by simple numbered lines like "(1) " or "1. "
            tmp = re.split(r"(?:(?:^|\s))(?:\(\d+\)|\d+\.)\s+", claims_block)
            claims = [t.strip() for t in tmp if t.strip()]

    # Description
    di = first("description")
    if di is not None:
        description = slice_between(di, None)
        description = re.sub(r"^.*?(発明の詳細な説明|詳細な説明|detailed description)】?\s*", "", description, flags=re.IGNORECASE).strip()
    else:
        description = text[text.find(claims_block) + len(claims_block):].strip() if claims_block else text

    # Paragraphs (coarse split by 。 or .)
    paragraphs = [p.strip() for p in re.split(r"(?<=。)\s+|(?<=\.)\s+", description) if p.strip()]
    return ParsedJP(title=title, abstract=abstract, claims=claims, description=description, paragraphs=paragraphs)

# ------------------------------
# Section tagging (JP/EN headers)
# ------------------------------
SECTION_TAG_PATTERNS = [
    # JP headers
    (re.compile(r"^【?技術分野】?"), "技術分野/Technical Field"),
    (re.compile(r"^【?背景技術】?"), "背景技術/Background Art"),
    (re.compile(r"^【?課題】?|^【?発明が解決しようとする課題】?"), "課題/Problem to be Solved"),
    (re.compile(r"^【?解決手段】?"), "解決手段/Solution"),
    (re.compile(r"^【?効果】?|^【?発明の効果】?"), "効果/Effects"),
    (re.compile(r"^【?実施形態】?"), "実施形態/Embodiments"),
    (re.compile(r"^【?実施例】?"), "実施例/Examples"),
    (re.compile(r"^【?比較例】?"), "比較例/Comparative Examples"),
    (re.compile(r"^【?変形例】?"), "変形例/Modifications"),
    (re.compile(r"^【?図面の簡単な説明】?"), "図面の簡単な説明/Brief Description of Drawings"),
    (re.compile(r"^【?産業上の利用可能性】?"), "産業上の利用可能性/Industrial Applicability"),
    # EN headers (\b only)
    (re.compile(r"^(technical field)\b", re.IGNORECASE), "技術分野/Technical Field"),
    (re.compile(r"^(background(?: art)?)\b", re.IGNORECASE), "背景技術/Background Art"),
    (re.compile(r"^(summary|summary of the invention)\b", re.IGNORECASE), "概要/Summary"),
    (re.compile(r"^(problem to be solved)\b", re.IGNORECASE), "課題/Problem to be Solved"),
    (re.compile(r"^(solution|means for solving the problem)\b", re.IGNORECASE), "解決手段/Solution"),
    (re.compile(r"^(effects?|advantages?)\b", re.IGNORECASE), "効果/Effects"),
    (re.compile(r"^(brief description of the drawings)\b", re.IGNORECASE), "図面の簡単な説明/Brief Description of Drawings"),
    (re.compile(r"^(detailed description|description of (?:the )?(?:embodiments|invention))\b", re.IGNORECASE), "詳細説明/Detailed Description"),
    (re.compile(r"^(embodiments?)\b", re.IGNORECASE), "実施形態/Embodiments"),
    (re.compile(r"^(examples?)\b", re.IGNORECASE), "実施例/Examples"),
    (re.compile(r"^(comparative examples?)\b", re.IGNORECASE), "比較例/Comparative Examples"),
    (re.compile(r"^(modifications?|variations?)\b", re.IGNORECASE), "変形例/Modifications"),
    (re.compile(r"^(industrial applicability)\b", re.IGNORECASE), "産業上の利用可能性/Industrial Applicability"),
]

def tag_paragraphs(paragraphs: List[str]) -> List[str]:
    tags: List[str] = []
    current = "その他/Other"
    for p in paragraphs:
        for rex, label in SECTION_TAG_PATTERNS:
            if rex.search(p):
                current = label
                break
        tags.append(current)
    return tags

# ------------------------------
# Tokenizer & BM25
# ------------------------------
WORD_RE = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*|[µ%℃°]+|[0-9]+(?:\.[0-9]+)?")

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    tokens: List[str] = []
    spans = []
    for m in WORD_RE.finditer(s):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    # residual for CJK
    res = []
    last = 0
    for a, b in spans:
        if last < a:
            res.append(s[last:a])
        last = b
    if last < len(s):
        res.append(s[last:])
    residual = "".join(res).replace(" ", "")
    for i in range(len(residual)):
        if i + 2 <= len(residual): tokens.append(residual[i:i+2])
        if i + 3 <= len(residual): tokens.append(residual[i:i+3])
    return tokens

@dataclass
class Chunk:
    idx: int
    start_para: int
    end_para: int
    text: str
    term_freq: Dict[str, int] = field(default_factory=dict)

def build_chunks(paragraphs: List[str], target_tokens: int = 600, overlap_tokens: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    cur_tokens = 0
    cur_texts: List[str] = []
    cur_start = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]; toks = tokenize(p)
        if cur_tokens == 0: cur_start = i
        cur_texts.append(p); cur_tokens += len(toks)
        if cur_tokens >= target_tokens or i == len(paragraphs) - 1:
            text = " ".join(cur_texts)
            tf: Dict[str, int] = {}
            for t in tokenize(text): tf[t] = tf.get(t, 0) + 1
            chunks.append(Chunk(idx=len(chunks), start_para=cur_start, end_para=i, text=text, term_freq=tf))
            # overlap seed
            if overlap_tokens > 0 and i < len(paragraphs) - 1:
                tail_texts = []; tail_tokens = 0; j = i
                while j >= cur_start and tail_tokens < overlap_tokens:
                    pt = paragraphs[j]
                    tail_texts.insert(0, pt)
                    tail_tokens += len(tokenize(pt)); j -= 1
                cur_texts = tail_texts[:]; cur_tokens = tail_tokens; cur_start = j + 1
            else:
                cur_texts = []; cur_tokens = 0; cur_start = i + 1
        i += 1
    return chunks

@dataclass
class BM25Index:
    chunks: List[Chunk]
    df: Dict[str, int]
    avgdl: float
    N: int
    k1: float = 1.2
    b: float = 0.75
    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
    def score(self, tf: int, dl: int, term: str) -> float:
        if tf <= 0: return 0.0
        denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
        return self.idf(term) * (tf * (self.k1 + 1)) / denom

def build_bm25_index(chunks: List[Chunk]) -> BM25Index:
    df: Dict[str, int] = {}
    for ch in chunks:
        for t in ch.term_freq.keys():
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(c.term_freq) for c in chunks) / max(1, len(chunks))
    return BM25Index(chunks=chunks, df=df, avgdl=avgdl, N=len(chunks))

# ------------------------------
# Elements & scoring
# ------------------------------
def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_elements(path: str) -> List[Dict[str, Any]]:
    """
    Load JSON, validate the structure, and normalize into a flat dict list.

    Backward compatibility:
        - If cues missing, upgrade old format {term?, synonyms}.
        - If 'term' missing -> use first synonym.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "elements" in raw:
        items = raw["elements"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("elements.json must be a dict with 'elements' or a list")

    if not isinstance(items, list):
        raise ValueError("elements list must be an array")

    normalized: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Element at index {idx} is not an object")
        elem_id = str(item.get("id") or "").strip()
        if not elem_id:
            raise ValueError(f"Element at index {idx} missing 'id'")
        if elem_id in seen_ids:
            raise ValueError(f"Duplicate element id '{elem_id}'")
        seen_ids.add(elem_id)

        weight = float(item.get("weight", 1.0))

        top_k_val = item.get("top_k")
        top_k: Optional[int]
        if top_k_val is None:
            top_k = None
        else:
            try:
                top_k = int(top_k_val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Element '{elem_id}' top_k must be an integer") from exc
            if top_k <= 0:
                raise ValueError(f"Element '{elem_id}' top_k must be > 0")

        cues_raw = item.get("cues")
        cues: List[Dict[str, Any]]
        if cues_raw is None:
            term = str(item.get("term") or "").strip()
            synonyms_raw = item.get("synonyms", [])
            if synonyms_raw is None:
                synonyms_raw = []
            if isinstance(synonyms_raw, str):
                synonyms_raw = [synonyms_raw]
            if not isinstance(synonyms_raw, list):
                raise ValueError(f"Element '{elem_id}' synonyms must be a list or string")
            synonyms = [str(s).strip() for s in synonyms_raw if str(s).strip()]
            if not term:
                if synonyms:
                    term = synonyms[0]
                else:
                    raise ValueError(f"Element '{elem_id}' requires a 'term' or non-empty synonyms")
            cue_synonyms = _dedupe_preserve_order(s for s in synonyms if s != term)
            cues = [{"term": term, "synonyms": cue_synonyms}]
        else:
            if not isinstance(cues_raw, list) or not cues_raw:
                raise ValueError(f"Element '{elem_id}' cues must be a non-empty list")
            cues = []
            for cidx, cue in enumerate(cues_raw, start=1):
                if not isinstance(cue, dict):
                    raise ValueError(f"Element '{elem_id}' cue #{cidx} is not an object")
                term = str(cue.get("term") or "").strip()
                if not term:
                    raise ValueError(f"Element '{elem_id}' cue #{cidx} missing 'term'")
                syn_raw = cue.get("synonyms", [])
                if syn_raw is None:
                    syn_raw = []
                if isinstance(syn_raw, str):
                    syn_raw = [syn_raw]
                if not isinstance(syn_raw, list):
                    raise ValueError(f"Element '{elem_id}' cue #{cidx} synonyms must be list or string")
                syn = [str(s).strip() for s in syn_raw if str(s).strip() and str(s).strip() != term]
                cues.append({"term": term, "synonyms": _dedupe_preserve_order(syn)})

        normalized.append({
            "id": elem_id,
            "weight": weight,
            "top_k": top_k,
            "cues": cues,
        })

    return normalized


@dataclass
class Cue:
    term: str
    synonyms: List[str]

    def variants(self) -> List[str]:
        return [self.term] + self.synonyms


@dataclass
class ElementDef:
    id: str
    weight: float
    cues: List[Cue]
    top_k: Optional[int] = None


def to_element_defs(objs: List[Dict[str, Any]]) -> List[ElementDef]:
    elements: List[ElementDef] = []
    for obj in objs:
        cues = [Cue(term=c["term"], synonyms=list(c.get("synonyms", []))) for c in obj["cues"]]
        if not cues:
            raise ValueError(f"Element '{obj['id']}' must define at least one cue")
        elements.append(
            ElementDef(
                id=obj["id"],
                weight=float(obj.get("weight", 1.0)),
                cues=cues,
                top_k=obj.get("top_k"),
            )
        )
    return elements


def phrase_to_terms(phrase: str) -> List[str]:
    return tokenize(phrase)


def bm25_phrase_score(index: BM25Index, ch: Chunk, phrase: str) -> float:
    return sum(index.score(ch.term_freq.get(t, 0), len(ch.term_freq), t) for t in phrase_to_terms(phrase))


def score_cue(index: BM25Index, ch: Chunk, cue: Cue) -> Tuple[float, str]:
    best_score = 0.0
    best_variant = ""
    for variant in cue.variants():
        score = bm25_phrase_score(index, ch, variant)
        if score > best_score:
            best_score = score
            best_variant = variant
    return best_score, best_variant

# ------------------------------
# Normalization utilities
# ------------------------------
def normalize_scores(values: List[float], method: str) -> List[float]:
    if not values:
        return []
    if method == "none":
        return values[:]
    if method == "minmax":
        lo, hi = min(values), max(values)
        if hi - lo <= 1e-12:
            return [0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]
    # default: zscore
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values]

# ------------------------------
# Highlighter (optional)
# ------------------------------
def _build_phrase_regex(phrase: str) -> re.Pattern:
    p = unicodedata.normalize("NFKC", phrase or "")
    p = re.escape(p)
    p = p.replace(r"\ ", r"\s+")
    return re.compile(p, flags=re.IGNORECASE)

def _highlight_sentence(sentence: str, phrase_pat: re.Pattern) -> str | None:
    if not sentence: return None
    s_norm = unicodedata.normalize("NFKC", sentence)
    parts = []; last = 0
    hits = list(phrase_pat.finditer(s_norm))
    if not hits: return None
    for m in hits[:3]:
        a, b = m.span()
        parts.append(s_norm[last:a]); parts.append("<<"); parts.append(s_norm[a:b]); parts.append(">>"); last = b
    parts.append(s_norm[last:])
    return "".join(parts)

# ------------------------------
# Pipeline: chunk + rank
# ------------------------------
def chunk_and_rank(
    parsed: ParsedJP,
    elements: List[ElementDef],
    chunk_tokens: int = 600,
    overlap_tokens: int = 120,
    top_k: int = 3,
    include_text: bool = True,
    overlap_ratio: float | None = None,
    suppress_adjacent: bool = True,
    adjacent_window: int = 1,
    adjacent_penalty: float = 0.7,
    highlight: bool = False,
    normalize_method: str = "zscore",
    tau_cue: float = 0.0,
) -> Dict[str, Any]:
    if overlap_ratio is not None:
        overlap_tokens = max(0, int(round(chunk_tokens * float(overlap_ratio))))
    chunks = build_chunks(parsed.paragraphs, target_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    index = build_bm25_index(chunks)
    para_tags = tag_paragraphs(parsed.paragraphs)

    def chunk_section(ch: Chunk) -> str:
        sl = para_tags[ch.start_para: ch.end_para + 1]
        return Counter(sl).most_common(1)[0][0] if sl else "不明/Unknown"

    results: Dict[str, List[Dict[str, Any]]] = {e.id: [] for e in elements}

    for ch in chunks:
        per_element_matches: Dict[str, Dict[str, Any]] = {}
        chunk_satisfied = True
        chunk_total_score = 0.0
        for e in elements:
            cue_matches: List[Dict[str, Any]] = []
            cue_scores_sum = 0.0
            element_ok = True
            first_variant_for_highlight = ""
            for cue in e.cues:
                cue_score, variant = score_cue(index, ch, cue)
                if not first_variant_for_highlight and variant:
                    first_variant_for_highlight = variant
                cue_matches.append({
                    "cue_term": cue.term,
                    "variant": variant,
                    "score": cue_score,
                })
                if cue_score <= tau_cue:
                    element_ok = False
                cue_scores_sum += cue_score
            if not element_ok:
                chunk_satisfied = False
                break
            element_score = cue_scores_sum * e.weight
            per_element_matches[e.id] = {
                "matches": cue_matches,
                "score": element_score,
                "highlight_variant": first_variant_for_highlight,
            }
            chunk_total_score += element_score

        if not chunk_satisfied or chunk_total_score <= 0:
            continue

        section = chunk_section(ch)
        chunk_text = ch.text if include_text else None

        for e in elements:
            elem_info = per_element_matches[e.id]
            entry: Dict[str, Any] = {
                "element_id": e.id,
                "chunk_idx": ch.idx,
                "start_para": ch.start_para,
                "end_para": ch.end_para,
                "section": section,
                "score": elem_info["score"],
                "chunk_score_total": chunk_total_score,
                "matched": elem_info["matches"],
            }
            if include_text:
                entry["chunk"] = chunk_text
            if highlight and elem_info["highlight_variant"] and chunk_text:
                phrase_pat = _build_phrase_regex(elem_info["highlight_variant"])
                highlighted = _highlight_sentence(chunk_text, phrase_pat)
                if highlighted:
                    entry["chunk_highlight"] = highlighted
            results[e.id].append(entry)

    # Adjacency suppression & normalization per element
    for e in elements:
        scored = results[e.id]
        scored.sort(key=lambda x: x["score"], reverse=True)
        if suppress_adjacent and scored:
            for i in range(len(scored)):
                ci = scored[i]["chunk_idx"]
                for j in range(i + 1, len(scored)):
                    cj = scored[j]["chunk_idx"]
                    if abs(cj - ci) <= adjacent_window:
                        scored[j]["score"] *= adjacent_penalty
            scored.sort(key=lambda x: x["score"], reverse=True)

        vals = [d["score"] for d in scored]
        vals_n = normalize_scores(vals, normalize_method)
        for d, nv in zip(scored, vals_n):
            d["score_norm"] = nv

        per_element_top_k = e.top_k if e.top_k is not None else top_k
        results[e.id] = scored[:per_element_top_k]

    chunks_out = [{"idx": ch.idx, "start_para": ch.start_para, "end_para": ch.end_para} for ch in index.chunks]

    return {
        "meta": {
            "num_chunks": len(index.chunks),
            "avgdl": index.avgdl,
            "normalize": normalize_method,
            "tau_cue": tau_cue,
        },
        "chunks": chunks_out,
        "results": results,
    }

# ------------------------------
# CLI main
# ------------------------------
def main():
    DEFAULT_CHUNK_TOKENS = 600
    DEFAULT_OVERLAP_TOKENS = 120
    DEFAULT_TOP_K = 3
    DEFAULT_NORMALIZE = "zscore"
    DEFAULT_HIGHLIGHT = False
    DEFAULT_THRESH_PARAS = 150
    DEFAULT_THRESH_TOKENS = 60000

    ap = argparse.ArgumentParser(description="Evidence pipeline (JP patents; long=results only / short=raw only)")
    ap.add_argument("--elements", required=True, help="Path to elements.json")
    ap.add_argument("--input", help="Input txt (default: stdin)")
    ap.add_argument("--output", help="Output json (default: stdout)")
    ap.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    ap.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--include-text", action="store_true", default=True,
                    help="Include chunk text inside results (long-doc mode).")
    ap.add_argument("--highlight", action="store_true", default=DEFAULT_HIGHLIGHT,
                    help="Enable highlights (optional).")
    ap.add_argument("--normalize", choices=["none", "zscore", "minmax"],
                    default=DEFAULT_NORMALIZE, help="Score normalization method")
    ap.add_argument("--auto-chunk-threshold-paragraphs", type=int, default=DEFAULT_THRESH_PARAS,
                    help="Consider document LONG if paragraph count >= this")
    ap.add_argument("--auto-chunk-threshold-tokens", type=int, default=DEFAULT_THRESH_TOKENS,
                    help="Consider document LONG if token count > this")
    ap.add_argument("--force-mode", choices=["auto", "full", "chunked"], default="auto",
                    help="full=always raw, chunked=always results, auto=threshold-based")
    args = ap.parse_args()

    # load
    text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    elements = to_element_defs(load_elements(args.elements))

    # parse & measure
    parsed = parse_jpo_plaintext(text)
    para_count = len(parsed.paragraphs)
    desc_tokens = len(tokenize(parsed.description))

    if args.force_mode == "full":
        mode_long = False
    elif args.force_mode == "chunked":
        mode_long = True
    else:
        mode_long = (
            (args.auto_chunk_threshold_paragraphs > 0 and para_count >= args.auto_chunk_threshold_paragraphs) or
            (args.auto_chunk_threshold_tokens > 0 and desc_tokens > args.auto_chunk_threshold_tokens)
        )

    meta_base = {
        "desc_paragraphs": para_count,
        "desc_tokens": desc_tokens,
        "threshold_paragraphs": args.auto_chunk_threshold_paragraphs,
        "threshold_tokens": args.auto_chunk_threshold_tokens,
        "normalize": args.normalize,
        "mode": "chunked" if mode_long else "full",
    }

    if mode_long:
        # Long doc: results ONLY, include chunk TEXT inside results
        ranked = chunk_and_rank(
            parsed, elements,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            top_k=args.top_k,
            include_text=args.include_text,
            highlight=args.highlight,
            normalize_method=args.normalize,
        )
        out = {
            "meta": {**ranked.get("meta", {}), **meta_base,
                     "chunk_params": {"chunk_tokens": args.chunk_tokens, "overlap_tokens": args.overlap_tokens}},
            "title": parsed.title,
            "abstract": parsed.abstract,
            "claims": parsed.claims,
            "results": ranked.get("results", {}),
        }
    else:
        # Short/Mid doc: raw only (no results)
        out = {
            "meta": meta_base,
            "title": parsed.title,
            "abstract": parsed.abstract,
            "claims": parsed.claims,
            "description": parsed.description,
        }

    js = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(js, encoding="utf-8")
    else:
        print(js)

if __name__ == "__main__":
    main()
