"""Socratic Git CLI entrypoint."""

import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import select
from urllib import request as urllib_request
from urllib import error as urllib_error
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from socratic_git.miner import (
    InvalidGitRepositoryError,
    Repo,
    extract_commits,
    get_file_blame,
    get_blame_for_line,
    get_introducing_commit,
    get_symbol_or_lines_from_query,
    find_symbol_definition,
    find_symbol_in_repo,
    extract_changed_files,
    clean_diff,
    get_local_history_window,
    get_file_content_at_commit,
    find_introducing_commit_by_predicate,
    find_symbol_definition_in_text,
    detect_structured_changes,
)
from socratic_git.rag import (
    SentenceTransformer,
    append_commits,
    configure_table,
    index_commits,
    lancedb,
    open_existing_table,
    retrieve,
)
from socratic_git.utils import print_evidence, table_name_for_repo


def _generate_answer(*args, **kwargs):
    """Lazy-load LLM backend to avoid MLX init on non-LLM commands."""
    from socratic_git.llm import generate_answer

    return generate_answer(*args, **kwargs)


def _llm_disabled():
    """Allow opt-out in constrained environments without changing default behavior."""
    return os.environ.get("SOCRATIC_SKIP_LLM", "").strip() == "1"


def sanitize_llm_output(text):
    cleaned = text or ""
    if "<|im_" in cleaned:
        cleaned = cleaned.split("<|im_", 1)[0]
    lines_local = cleaned.splitlines()
    kept = []
    in_code = False
    for ln in lines_local:
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            kept.append(ln)
    cleaned = "\n".join(kept).strip()
    cleaned = cleaned.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    return cleaned


def _truncate_sentences(text, max_sentences=3):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]
    if not parts:
        return ""
    return " ".join(parts[:max_sentences]).strip()


def normalize_text(s):
    text = (s or "").lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


GITHUB_MOTIVE_TERMS = (
    "because",
    "due to",
    "reason",
    "why",
    "原因",
    "为了",
    "to reduce",
    "in order to",
    "latency",
    "timeout",
    "flaky",
    "stability",
    "performance",
)

GITHUB_CHANGE_KEYS = {
    "timeout",
    "threshold",
    "retry",
    "retries",
    "backoff",
    "ttl",
    "limit",
    "window",
    "batch",
    "max",
    "min",
    "debounce",
}

GITHUB_QUERY_KEYWORDS = {
    "timeout",
    "retry",
    "retries",
    "threshold",
    "latency",
    "performance",
    "stability",
    "error",
    "fail",
    "flaky",
}


def detect_query_intent(query):
    text = (query or "").lower()
    keys = ["timeout", "retry", "retries", "threshold", "limit", "backoff", "window", "batch", "max", "min", "ttl", "debounce"]
    matched_keys = [k for k in keys if re.search(rf"\b{re.escape(k)}\b", text)]

    param_patterns = (
        r"\bchanged\s+from\b",
        r"\bincrease\b",
        r"\bdecrease\b",
        r"\b\d+\s*(?:to|->)\s*\d+\b",
    )
    if matched_keys or any(re.search(p, text) for p in param_patterns):
        return {"intent": "param_change", "keys": matched_keys}
    if any(w in text for w in ("regression", "bug", "fail", "failure", "flaky")):
        return {"intent": "bug_regression", "keys": matched_keys}
    if any(w in text for w in ("introduce", "introduced", "added", "add feature")):
        return {"intent": "introduce_feature", "keys": matched_keys}
    return {"intent": "general", "keys": matched_keys}


def _build_parameter_change_analysis(intent_info, structured_changes, changed_files, evidence_text):
    if (intent_info or {}).get("intent") != "param_change":
        return ""

    keys = set((intent_info or {}).get("keys", []))
    selected = []
    for ch in structured_changes or []:
        key = (ch.get("key") or "").lower()
        if keys and key not in keys:
            continue
        selected.append(ch)
    if not selected:
        selected = (structured_changes or [])[:5]
    else:
        selected = selected[:5]

    file_lows = [f.lower() for f in (changed_files or [])]
    ev = (evidence_text or "").lower()
    has_test_ci = any(any(tok in f for tok in ("test", "spec", "ci")) for f in file_lows)
    has_network = any(any(tok in f for tok in ("network", "db", "http", "client")) for f in file_lows) or any(
        tok in ev for tok in ("network", "db", "http", "client", "latency")
    )
    has_retry_backoff = any((c.get("key") or "").lower() in {"retry", "retries", "backoff"} for c in selected) or any(
        tok in ev for tok in ("retry", "retries", "backoff")
    )

    hypotheses = []
    if has_test_ci:
        hypotheses.append("May relate to test stability (test-related files changed), but not explicitly stated.")
    if has_network:
        hypotheses.append("May relate to dependency/network latency handling, but not explicitly stated.")
    if has_retry_backoff:
        hypotheses.append("May improve resilience under transient failures (retry/backoff signals), but not explicitly stated.")
    if not hypotheses:
        hypotheses.append("Possibly performance/stability tuning; motive not explicitly stated.")
    hypotheses = hypotheses[:3]

    lines = ["## Parameter Change Analysis", "- Detected changes:"]
    if selected:
        for ch in selected:
            lines.append(
                f"  - {ch.get('key', '')}: {ch.get('from', '')} -> {ch.get('to', '')} "
                f"(file={ch.get('file', '')}, hunk={ch.get('hunk', '')})"
            )
    else:
        lines.append("  - (none)")
    lines.append("- Likely rationale (Hypotheses):")
    for i, h in enumerate(hypotheses, start=1):
        lines.append(f"  {i}) {h}")
    lines.append("- Evidence:")
    lines.append(f"  - changed_files: {', '.join(changed_files or []) if changed_files else '(none)'}")
    if selected:
        for ch in selected:
            lines.append(f"  - change: {ch.get('key', '')} {ch.get('from', '')}->{ch.get('to', '')}")
    else:
        lines.append("  - change: (none)")
    return "\n".join(lines)


def _extract_motive_sentences(text, max_sentences=2):
    keywords = (
        "because",
        "due to",
        "to prevent",
        "to improve",
        "in order to",
        "fix",
        "mitigate",
        "原因",
        "为了",
    )
    src = (text or "").strip()
    if not src:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", src)
    out = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in keywords):
            out.append(s)
        if len(out) >= max_sentences:
            break
    return out


def issue_relevance_score(query: str, structured_changes: list, introducing_commit: dict, issue_title: str, issue_body: str) -> dict:
    """Score GitHub issue/PR relevance for motive evidence guard."""
    stop = {
        "the", "and", "for", "with", "this", "that", "why", "was", "were", "what", "when",
        "where", "which", "how", "from", "into", "line", "file", "added", "changed", "in",
    }
    q_tokens = [t for t in _tokenize_query(query or "") if len(t) >= 3 and t not in stop]
    text = normalize_text(f"{issue_title or ''} {issue_body or ''}")
    breakdown = {
        "query_key_match": 0,
        "structured_key_match": 0,
        "file_or_module_match": 0,
        "short_text_penalty": 0,
    }
    query_hit_count = sum(1 for t in q_tokens if re.search(rf"\b{re.escape(t)}\b", text))
    if query_hit_count >= 2:
        breakdown["query_key_match"] = 2

    struct_keys = {(c.get("key") or "").lower() for c in (structured_changes or []) if c.get("key")}
    if struct_keys:
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in struct_keys):
            breakdown["structured_key_match"] = 2

    intro_files = (introducing_commit or {}).get("changed_files", []) or []
    for fp in intro_files:
        base = os.path.basename(fp).lower()
        stem = os.path.splitext(base)[0]
        module = (fp or "").replace("\\", "/").lower().replace("/", ".")
        if (
            (base and base in text)
            or (stem and re.search(rf"\b{re.escape(stem)}\b", text))
            or (module and module in text)
        ):
            breakdown["file_or_module_match"] = 1
            break

    if len((issue_body or "").strip()) < 40:
        breakdown["short_text_penalty"] = -1

    score = max(0, min(5, sum(breakdown.values())))
    verdict = "accept" if score >= 3 else "reject"
    return {"score": score, "breakdown": breakdown, "verdict": verdict}


def build_motive_evidence(repo_path, query, evidence) -> dict:
    """Build layered motive evidence from already-computed signals."""
    selected = evidence.get("selected_github_evidence", []) or []
    introducing = evidence.get("introducing_commit", {}) or {}
    retrieved = evidence.get("retrieved_commits", []) or []
    structured = evidence.get("structured_changes", []) or []
    changed_files = evidence.get("changed_files", []) or []
    impact_prop_raw = evidence.get("impact_propagation", {}) or {}
    cochange_top_raw = evidence.get("cochange_top", {}) or {}
    timeline_items_raw = evidence.get("timeline_items", []) or []

    # Sanitize noisy signals:
    # - Co-change: only code/config partitions.
    # - Impact: keep counts only.
    # - Timeline: keep aggregate counters only.
    impact_level_1 = len(impact_prop_raw.get("level_1", []) or [])
    impact_level_2 = len(impact_prop_raw.get("level_2", []) or [])
    cochange_code = cochange_top_raw.get("top_code", []) if isinstance(cochange_top_raw, dict) else []
    cochange_config = cochange_top_raw.get("top_config", []) if isinstance(cochange_top_raw, dict) else []
    timeline_count = len(timeline_items_raw)
    timeline_intro_count = 0
    timeline_structured_count = 0
    for it in timeline_items_raw:
        sigs = set(it.get("signals", []) or [])
        if "introducing" in sigs:
            timeline_intro_count += 1
        if "structured_change" in sigs:
            timeline_structured_count += 1

    explicit = []
    github_relevance = []
    rejected_refs = []
    for item in selected[:3]:
        num = item.get("number")
        title = item.get("title", "") or ""
        body = item.get("body_snippet", "") or ""
        rel = issue_relevance_score(
            query=query or "",
            structured_changes=structured or [],
            introducing_commit=introducing or {},
            issue_title=title,
            issue_body=body,
        )
        github_relevance.append(
            {"number": num, "score": rel.get("score", 0), "breakdown": rel.get("breakdown", {}), "verdict": rel.get("verdict", "reject")}
        )
        if rel.get("verdict") != "accept":
            rejected_refs.append({"number": num, "score": rel.get("score", 0)})
            continue
        for s in _extract_motive_sentences(title + "\n" + body, max_sentences=2):
            explicit.append({"source": f"github_issue #{num}", "sentence": s})

    intro_hash = (introducing.get("hash", "") or "")[:8]
    intro_msg = introducing.get("message", "") or ""
    for s in _extract_motive_sentences(intro_msg, max_sentences=1):
        explicit.append({"source": f"commit_message {intro_hash}", "sentence": s})

    for ctx in retrieved[:3]:
        c = ctx.get("commit", {}) if isinstance(ctx, dict) else {}
        h = (c.get("hash", "") or "")[:8]
        m = c.get("message", "") or ""
        for s in _extract_motive_sentences(m, max_sentences=1):
            explicit.append({"source": f"commit_message {h}", "sentence": s})

    # diff comments with explicit motive words
    intro_diff = introducing.get("diff", "") or ""
    for ln in intro_diff.splitlines():
        s = ln.strip()
        if not (s.startswith("+#") or s.startswith("+//") or s.startswith("#") or s.startswith("//")):
            continue
        for sent in _extract_motive_sentences(s, max_sentences=1):
            explicit.append({"source": f"diff_comment {intro_hash}", "sentence": sent})

    # de-dup explicit
    seen_exp = set()
    explicit_dedup = []
    for e in explicit:
        k = (e.get("source", ""), e.get("sentence", ""))
        if k in seen_exp:
            continue
        seen_exp.add(k)
        explicit_dedup.append(e)
    explicit = explicit_dedup[:5]

    strong = []
    for ch in structured[:2]:
        strong.append(
            f"structured_change: {ch.get('key','')} {ch.get('from','')} -> {ch.get('to','')} (file={ch.get('file','')})"
        )
    lowers = [f.lower() for f in changed_files]
    test_or_ci_touched = any(any(t in f for t in ("tests/", "test", "spec", "ci", "github/workflows")) for f in lowers)
    strong.append(f"test_or_ci_touched: {'true' if test_or_ci_touched else 'false'}")
    strong.append(f"impact_propagation: level1={impact_level_1}, level2={impact_level_2}")
    lifts = []
    for row in (cochange_code or []) + (cochange_config or []):
        try:
            lifts.append(float(row.get("lift", 0) or 0))
        except Exception:
            continue
    min_lift = max([v for v in lifts if v >= 2.0], default=0.0)
    strong.append(
        f"cochange_related: top_code={len(cochange_code)}, top_config={len(cochange_config)} "
        f"(min_lift={round(min_lift, 3) if min_lift else 0})"
    )
    strong.append(
        f"timeline_stats: count={timeline_count}, introducing={timeline_intro_count}, structured={timeline_structured_count}"
    )

    # Motive score (0-10), deterministic and debuggable.
    score_breakdown = {
        "explicit_motive": 6 if explicit else 0,
        "structured_change": 2 if structured else 0,
        "test_or_ci_touched": 1 if test_or_ci_touched else 0,
        "impact_level1_positive": 1 if impact_level_1 > 0 else 0,
        "cochange_lift_ge_2": 1 if min_lift >= 2.0 else 0,
    }
    motive_score = min(10, sum(score_breakdown.values()))

    missing = []
    intro_low = (intro_msg or "").strip().lower()
    if not intro_low or intro_low.startswith(("update", "updating", "merge", "wip", "fix", "cleanup")):
        missing.append("commit message is generic")
    if not selected:
        missing.append("no linked issue/pr evidence")
    for r in rejected_refs[:3]:
        missing.append(f"github evidence rejected (low relevance score={r.get('score', 0)})")
    if not explicit:
        missing.append("no explicit motive sentence found")
    if not strong:
        missing.append("no strong non-text signals found")

    return {
        "explicit": explicit,
        "strong": strong,
        "missing": missing,
        "motive_score": motive_score,
        "score_breakdown": score_breakdown,
        "github_relevance": github_relevance,
    }


def render_motive_evidence_lines(motive_evidence):
    rel_items = ((motive_evidence or {}).get("github_relevance", []) or [])
    if rel_items:
        rel_text = ", ".join(
            [
                f"#{x.get('number')} score={x.get('score')} verdict={x.get('verdict')} breakdown={x.get('breakdown')}"
                for x in rel_items
            ]
        )
    else:
        rel_text = "(none)"
    lines = [
        "## Motive Evidence (Layered)",
        f"- motive_score: {int((motive_evidence or {}).get('motive_score', 0))}/10",
        f"- score_breakdown: {(motive_evidence or {}).get('score_breakdown', {})}",
        f"- github_relevance: {rel_text}",
        "",
        "### A) Explicit Motive (High)",
    ]
    explicit = motive_evidence.get("explicit", []) if motive_evidence else []
    if explicit:
        for e in explicit:
            lines.append(f"- source: {e.get('source','')}")
            lines.append(f"  sentence: \"{e.get('sentence','')}\"")
    else:
        lines.append("- (none)")

    lines.extend(["", "### B) Strong Signals (Medium)"])
    strong = motive_evidence.get("strong", []) if motive_evidence else []
    if strong:
        for s in strong:
            lines.append(f"- {s}")
    else:
        lines.append("- (none)")

    lines.extend(["", "### C) Unknown / Missing Evidence (Low)"])
    missing = motive_evidence.get("missing", []) if motive_evidence else []
    if missing:
        for m in missing:
            lines.append(f"- missing: {m}")
    else:
        lines.append("- (none)")
    return lines


def build_answer_template(
    findings,
    motive,
    has_explicit_motive,
    structured_changes,
    evidence_text,
    missing_items=None,
    intent_info=None,
    changed_files=None,
    selected_github_evidence=None,
    why_chain="",
    motive_evidence=None,
):
    """Build deterministic answer with confidence and missing-evidence section."""
    selected = selected_github_evidence or []
    layered = motive_evidence or {}
    explicit_layer = layered.get("explicit", []) or []
    strong_layer = layered.get("strong", []) or []
    missing_layer = layered.get("missing", []) or []
    missing = list(missing_items or [])
    missing.extend([m for m in missing_layer if m not in missing])
    evidence_low = (evidence_text or "").lower()
    if not has_explicit_motive and "generic" not in evidence_low:
        if "Commit message is generic; no clear motivation statement." not in missing:
            missing.append("Commit message is generic; no clear motivation statement.")
    if not selected and "pr-" not in evidence_low and "issue" not in evidence_low and "http" not in evidence_low:
        if "No linked ticket/PR evidence available." not in missing:
            missing.append("No linked ticket/PR evidence available.")
    if not any(tok in evidence_low for tok in ("test", "ci", "log")):
        if "No test/CI/log evidence in diff." not in missing:
            missing.append("No test/CI/log evidence in diff.")

    no_commit_motive_text = "Motive is not explicitly stated in commit/diff evidence."
    motive_text = (motive or "").strip() or no_commit_motive_text

    # Confidence calibration:
    # - Finding Confidence: deterministic facts (commit/blame/structured changes) => usually High.
    # - Motive Confidence: conservative by default.
    strong_keys = {"timeout", "retry", "retries", "threshold", "backoff", "ttl"}
    explicit_motive_terms = {"because", "due to", "fix timeout", "flaky", "performance", "latency", "stability", "原因", "为了"}
    medium_signal_terms = {"timeout error", "retry", "deadline exceeded", "connection reset", "rate limit"}

    finding_confidence = "High" if (findings or "").strip() else "Medium"

    has_structured_numeric = any(
        (ch.get("key", "") in strong_keys)
        and (str(ch.get("from", "")).replace(".", "", 1).isdigit())
        and (str(ch.get("to", "")).replace(".", "", 1).isdigit())
        for ch in (structured_changes or [])
    )
    has_explicit_motive_text = has_explicit_motive or any(t in evidence_low for t in explicit_motive_terms)
    has_key_match = any((ch.get("key", "") or "").lower() in evidence_low for ch in (structured_changes or []))
    has_medium_signal = any(t in evidence_low for t in medium_signal_terms)
    generic_message = any(k in evidence_low for k in (" merge ", " updating ", " update ", "wip"))

    github_has_motive_and_key = any(i.get("has_motive_word") and i.get("has_key_match") for i in selected)
    github_has_any = bool(selected)

    # Motive confidence defaults to Low; upgraded cautiously.
    if has_explicit_motive_text and has_key_match:
        motive_confidence = "High"
    elif has_structured_numeric and has_medium_signal:
        motive_confidence = "Medium"
    elif has_structured_numeric and has_explicit_motive_text and has_key_match:
        motive_confidence = "Medium"
    else:
        motive_confidence = "Low"
    if generic_message and not has_explicit_motive_text:
        motive_confidence = "Low"
    if github_has_any and not github_has_motive_and_key:
        if "Issue/PR exists but does not explicitly state motive." not in missing:
            missing.append("Issue/PR exists but does not explicitly state motive.")

    best_evidence = selected[0] if selected else None
    best_sentence = ""
    if best_evidence:
        best_sentence = _truncate_sentences(best_evidence.get("body_snippet", "") or best_evidence.get("title", ""), max_sentences=1)

    if explicit_layer:
        if motive_confidence == "Low":
            motive_confidence = "Medium"
        top = explicit_layer[0]
        motive_text = f"Motive is explicitly stated in {top.get('source','evidence')}: {top.get('sentence','')}"
    elif strong_layer:
        motive_confidence = "Low"
        if not selected and "no linked issue/pr evidence" not in " ".join(missing).lower():
            missing.append("no linked issue/pr evidence")
    else:
        motive_confidence = "Low"

    # Phase 17: bind confidence to motive_score (0~10).
    motive_score = layered.get("motive_score", None)
    if isinstance(motive_score, (int, float)):
        if motive_score >= 7:
            motive_confidence = "High"
        elif motive_score >= 4:
            motive_confidence = "Medium"
        else:
            motive_confidence = "Low"
        if not explicit_layer and motive_confidence == "High":
            motive_confidence = "Medium"

    if motive_confidence in ("Medium", "High") and best_evidence and best_evidence.get("has_motive_word") and best_evidence.get("has_key_match"):
        motive_text = f"Motive is explicitly stated in linked Issue/PR #{best_evidence.get('number')}: {best_sentence or '(evidence text unavailable)'}"
    elif github_has_any:
        motive_text = "Issue/PR exists but does not explicitly state motive."
    elif not has_explicit_motive and ("because" not in normalize_text(motive_text) and "reason" not in normalize_text(motive_text)):
        motive_text = no_commit_motive_text

    signals = []
    if structured_changes:
        ch = structured_changes[0]
        signals.append(f"structured_change:{ch.get('key','')} {ch.get('from','')}->{ch.get('to','')}")
    else:
        signals.append("structured_change:none")
    file_lows = [f.lower() for f in (changed_files or [])]
    if any(any(tok in f for tok in ("test", "spec", "ci")) for f in file_lows):
        signals.append("changed_files:test")
    elif any(any(tok in f for tok in ("network", "db", "http", "client")) for f in file_lows):
        signals.append("changed_files:network_or_db")
    else:
        signals.append("changed_files:none")
    if explicit_layer:
        signals.append("text_signal:explicit_motive")
    elif has_medium_signal:
        signals.append("text_signal:related_runtime_signal")
    else:
        signals.append("text_signal:none")

    missing_line = "; ".join(missing) if missing else "No critical missing evidence identified."
    param_section = _build_parameter_change_analysis(intent_info, structured_changes, changed_files or [], evidence_text)
    why_chain_text = why_chain.strip() or "line=(unknown) -> commit=(unknown) -> refs=(none) -> issue_title=(none) -> evidence_sentence=(none)"
    return (
        "Findings (Deterministic):\n"
        f"{findings}\n\n"
        "Motive (Evidence-Driven, Optional):\n"
        f"{motive_text}\n\n"
        "Why Chain:\n"
        f"- {why_chain_text}\n\n"
        f"{param_section + chr(10) + chr(10) if param_section else ''}"
        f"Finding Confidence: {finding_confidence}\n"
        f"Motive Confidence: {motive_confidence}\n"
        f"Motive Evidence Signals: {signals}\n"
        f"Missing Evidence: {missing_line}"
    )


def _tokenize_query(text):
    return [t for t in re.split(r"[^A-Za-z0-9_]+", (text or "").lower()) if t]


def _github_query_terms(query, structured_changes):
    q_norm = normalize_text(query)
    q_tokens = set(_tokenize_query(q_norm))
    query_keys = {k for k in (GITHUB_QUERY_KEYWORDS | GITHUB_CHANGE_KEYS) if re.search(rf"\b{re.escape(k)}\b", q_norm)}
    query_keys.update({t for t in q_tokens if t in (GITHUB_QUERY_KEYWORDS | GITHUB_CHANGE_KEYS)})
    structured_keys = {(c.get("key") or "").lower() for c in (structured_changes or []) if c.get("key")}
    return query_keys, structured_keys


def _score_github_item(item, query_keys, structured_keys):
    title = item.get("title", "") or ""
    body = item.get("body_snippet", "") or ""
    txt = normalize_text(f"{title} {body}")
    score = 0
    query_hits = [k for k in sorted(query_keys) if re.search(rf"\b{re.escape(k)}\b", txt)]
    structured_hits = [k for k in sorted(structured_keys) if re.search(rf"\b{re.escape(k)}\b", txt)]
    motive_hits = [w for w in GITHUB_MOTIVE_TERMS if w in txt]
    has_query_key = bool(query_hits)
    has_structured_key = bool(structured_hits)
    has_motive_word = bool(motive_hits)
    too_short = len((title + " " + body).strip()) < 50
    score_query = min(len(query_hits) * 2, 6)
    score_struct = min(len(structured_hits) * 2, 6)
    score_motive = 2 if has_motive_word else 0
    score_short = -1 if too_short else 0
    score += score_query
    score += score_struct
    score += score_motive
    if has_motive_word:
        score += 0
    score += score_short
    out = dict(item)
    out["score"] = score
    out["has_query_key"] = has_query_key
    out["has_key_match"] = has_structured_key or has_query_key
    out["has_motive_word"] = has_motive_word
    out["query_key_hits"] = query_hits
    out["structured_key_hits"] = structured_hits
    out["motive_word_hits"] = motive_hits
    out["score_breakdown"] = {
        "query_key_match": score_query,
        "structured_key_match": score_struct,
        "motive_word": score_motive,
        "short_text_penalty": score_short,
    }
    return out


def select_github_evidence(query, structured_changes, items, topn=3):
    query_keys, structured_keys = _github_query_terms(query, structured_changes)
    scored = [_score_github_item(i, query_keys, structured_keys) for i in (items or [])]
    scored.sort(key=lambda x: (x.get("score", 0), x.get("number", 0)), reverse=True)
    return scored[:topn]


def build_why_chain(line_no, commit_hash, selected_github):
    line_part = str(line_no) if line_no else "unknown"
    commit_part = commit_hash[:12] if commit_hash else "unknown"
    if not selected_github:
        return f"line={line_part} -> commit={commit_part} -> refs=(none) -> issue_title=(none) -> evidence_sentence=(none)"
    best = selected_github[0]
    refs = f"#{best.get('number')}"
    title = (best.get("title", "") or "").strip() or "(none)"
    body = (best.get("body_snippet", "") or "").strip()
    sentence = _truncate_sentences(body, max_sentences=1) or "(none)"
    return f"line={line_part} -> commit={commit_part} -> refs={refs} -> issue_title={title} -> evidence_sentence={sentence}"


def classify_file(path: str) -> str:
    p = (path or "").replace("\\", "/").lower()
    if any(seg in p for seg in ("node_modules/", "dist/", "target/", "build/", "out/")):
        return "BUILD"
    ext = os.path.splitext(p)[1]
    if ext in {".py", ".scala", ".js", ".ts", ".java", ".cpp", ".cc", ".c", ".hpp", ".h"}:
        return "CODE"
    if ext in {".json", ".yaml", ".yml", ".ini", ".toml"}:
        return "CONFIG"
    if ext in {".parquet", ".csv"}:
        return "DATA"
    if ext in {".joblib", ".pdf", ".png", ".jpg", ".bin", ".pkl", ".safetensors"}:
        return "ARTIFACT"
    return "OTHER"


def compute_cochange(repo_path, target_files, n_commits=300, mode="code"):
    """Compute cross-file co-change stats grouped by file class."""
    repo = Repo(repo_path)
    norm_targets = {str(t).replace("\\", "/").strip() for t in (target_files or []) if str(t).strip()}
    if not norm_targets:
        compute_cochange.last_window_commits = 0
        return {"top_code": [], "top_config": [], "ignored": []}

    total = 0
    target_any = 0
    file_count = {}
    co_count = {}
    class_map = {}

    window_days = os.environ.get("SOCRATIC_WINDOW_DAYS", "").strip()
    iter_kwargs = {}
    if window_days:
        try:
            iter_kwargs["since"] = f"{int(window_days)} days ago"
        except Exception:
            pass
    else:
        iter_kwargs["max_count"] = max(1, int(n_commits))
    iter_kwargs["no_merges"] = True
    for commit in repo.iter_commits(**iter_kwargs):
        files = {f.replace("\\", "/") for f in commit.stats.files.keys()}
        if not files:
            continue
        total += 1
        has_target = any(t in files for t in norm_targets)
        for f in files:
            class_map[f] = classify_file(f)
            file_count[f] = file_count.get(f, 0) + 1
        if has_target:
            target_any += 1
            for f in files:
                if f in norm_targets:
                    continue
                co_count[f] = co_count.get(f, 0) + 1

    compute_cochange.last_window_commits = total
    if total == 0 or target_any == 0:
        return {"top_code": [], "top_config": [], "ignored": []}

    rows = []
    for f, cc in co_count.items():
        p_other = file_count.get(f, 0) / total if total else 0.0
        p_other_given_target = cc / target_any if target_any else 0.0
        lift = (p_other_given_target / p_other) if p_other > 0 else 0.0
        rows.append(
            {
                "file": f,
                "class": class_map.get(f, "OTHER"),
                "co_count": cc,
                "lift": round(lift, 3),
                "p_other": round(p_other, 4),
                "p_other_given_target": round(p_other_given_target, 4),
            }
        )

    rows.sort(key=lambda x: (x["co_count"], x["lift"]), reverse=True)
    if mode == "all":
        pool = rows
    else:
        pool = [r for r in rows if r.get("class") in {"CODE", "CONFIG"}]

    top_code = [r for r in pool if r.get("class") == "CODE"][:5]
    top_config = [r for r in pool if r.get("class") == "CONFIG"][:5]
    ignored = [r for r in rows if r.get("class") in {"DATA", "ARTIFACT", "BUILD"}][:5]
    return {"top_code": top_code, "top_config": top_config, "ignored": ignored}


def _changed_files_for_hash(repo_path, commit_hash):
    if not commit_hash:
        return []
    try:
        repo = Repo(repo_path)
        raw = repo.git.show("--name-only", "--pretty=format:", commit_hash)
        files = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln not in files:
                files.append(ln.replace("\\", "/"))
        return files
    except Exception:
        return []


def get_cochange_evidence(repo_path, target_files, other_file, n_commits=300, max_commits=3):
    """Return commit snippets where target_files and other_file co-occur."""
    repo = Repo(repo_path)
    targets = {str(t).replace("\\", "/").strip() for t in (target_files or []) if str(t).strip()}
    other = (other_file or "").replace("\\", "/").strip()
    if not targets or not other:
        return []

    out = []
    window_days = os.environ.get("SOCRATIC_WINDOW_DAYS", "").strip()
    iter_kwargs = {}
    if window_days:
        try:
            iter_kwargs["since"] = f"{int(window_days)} days ago"
        except Exception:
            pass
    else:
        iter_kwargs["max_count"] = max(1, int(n_commits))
    iter_kwargs["no_merges"] = True
    for commit in repo.iter_commits(**iter_kwargs):
        files = {f.replace("\\", "/") for f in commit.stats.files.keys()}
        if not files:
            continue
        if other in files and any(t in files for t in targets):
            msg = (commit.message or "").strip().splitlines()[0] if commit.message else ""
            out.append(
                {
                    "hash": commit.hexsha[:8],
                    "date": commit.committed_datetime.date().isoformat(),
                    "message": msg,
                }
            )
            if len(out) >= max(1, int(max_commits)):
                break
    return out


def _read_text_safe(repo_path, rel_path):
    try:
        p = rel_path
        if not os.path.isabs(p):
            p = os.path.join(repo_path, rel_path)
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _run_symbol_rg_candidates(repo_path: str, symbol: str, limit: int = 50):
    """Collect symbol hit candidates via rg (file,line)."""
    if not symbol:
        return []
    if shutil.which("rg") is None:
        return []
    pattern = rf"\b{re.escape(symbol)}\b"
    try:
        proc = subprocess.run(
            [
                "rg",
                "-n",
                "--no-heading",
                "-S",
                "--max-filesize",
                "1M",
                "--glob",
                "!node_modules/*",
                "--glob",
                "!dist/*",
                "--glob",
                "!build/*",
                pattern,
                repo_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return []
    out = []
    for ln in (proc.stdout or "").splitlines():
        # path:line:content
        m = re.match(r"^(.*?):(\d+):(.*)$", ln)
        if not m:
            continue
        fp = m.group(1)
        try:
            rel = os.path.relpath(fp, repo_path).replace("\\", "/")
        except Exception:
            rel = fp.replace("\\", "/")
        out.append(
            {
                "file": rel,
                "line": int(m.group(2)),
                "text": m.group(3),
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return out


def _lsp_pack_message(payload: dict) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _lsp_read_message(fd: int, buffer: bytes, timeout_sec: float = 5.0):
    """Read one LSP framed message from fd, returning (obj|None, remaining_buffer)."""
    deadline = time.time() + max(0.1, timeout_sec)
    buf = buffer
    while time.time() < deadline:
        sep = buf.find(b"\r\n\r\n")
        if sep != -1:
            header = buf[:sep].decode("ascii", errors="ignore")
            rest = buf[sep + 4 :]
            m = re.search(r"Content-Length:\s*(\d+)", header, flags=re.IGNORECASE)
            if not m:
                # skip malformed header
                buf = rest
                continue
            length = int(m.group(1))
            if len(rest) >= length:
                body = rest[:length]
                remaining = rest[length:]
                try:
                    return json.loads(body.decode("utf-8", errors="ignore")), remaining
                except Exception:
                    return None, remaining
        remain = deadline - time.time()
        if remain <= 0:
            break
        r, _, _ = select.select([fd], [], [], remain)
        if not r:
            continue
        try:
            chunk = os.read(fd, 8192)
        except Exception:
            chunk = b""
        if not chunk:
            break
        buf += chunk
    return None, buf


def _lsp_try_definition(repo_path: str, symbol: str, candidates, preferred_file: str = "", tool_name: str = ""):
    """Try real LSP textDocument/definition. Returns (file, line, details)."""
    if not tool_name:
        return "", None, "no lsp tool selected"
    if not candidates:
        return "", None, "no candidates for lsp request"

    # choose query position: prefer candidate in preferred file, else first
    chosen = None
    if preferred_file:
        pf = preferred_file.replace("\\", "/")
        chosen = next((c for c in candidates if c.get("file", "") == pf), None)
    if chosen is None:
        chosen = candidates[0]
    src_rel = chosen.get("file", "")
    src_abs = os.path.abspath(os.path.join(repo_path, src_rel))
    src_uri = Path(src_abs).as_uri()
    src_line = max(0, int(chosen.get("line", 1)) - 1)
    src_text = chosen.get("text", "") or ""
    char_idx = src_text.find(symbol) if symbol else -1
    if char_idx < 0:
        char_idx = 0

    server_cmd = []
    if tool_name == "pyright-langserver":
        server_cmd = [tool_name, "--stdio"]
    elif tool_name == "typescript-language-server":
        server_cmd = [tool_name, "--stdio"]
    elif tool_name == "gopls":
        server_cmd = [tool_name]
    else:
        return "", None, f"unsupported lsp tool: {tool_name}"

    proc = None
    try:
        proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_path,
        )
        if proc.stdin is None or proc.stdout is None:
            return "", None, f"{tool_name} stdio unavailable"

        fd = proc.stdout.fileno()
        buf = b""
        req_id = 1
        init_req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": Path(os.path.abspath(repo_path)).as_uri(),
                "capabilities": {},
            },
        }
        proc.stdin.write(_lsp_pack_message(init_req))
        proc.stdin.flush()

        # wait initialize response
        init_ok = False
        while True:
            msg, buf = _lsp_read_message(fd, buf, timeout_sec=5.0)
            if msg is None:
                break
            if msg.get("id") == req_id and ("result" in msg or "error" in msg):
                init_ok = "result" in msg
                break
        if not init_ok:
            return "", None, f"{tool_name} initialize failed"

        # initialized notification
        proc.stdin.write(_lsp_pack_message({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        proc.stdin.flush()

        # didOpen (best effort)
        try:
            with open(src_abs, "r", encoding="utf-8", errors="ignore") as f:
                src_content = f.read()
            ext = os.path.splitext(src_abs)[1].lower()
            lang_id_map = {
                ".py": "python",
                ".ts": "typescript",
                ".tsx": "typescriptreact",
                ".js": "javascript",
                ".jsx": "javascriptreact",
                ".go": "go",
            }
            did_open = {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": src_uri,
                        "languageId": lang_id_map.get(ext, "plaintext"),
                        "version": 1,
                        "text": src_content,
                    }
                },
            }
            proc.stdin.write(_lsp_pack_message(did_open))
            proc.stdin.flush()
        except Exception:
            pass

        # definition request
        req_id = 2
        def_req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "textDocument/definition",
            "params": {
                "textDocument": {"uri": src_uri},
                "position": {"line": src_line, "character": char_idx},
            },
        }
        proc.stdin.write(_lsp_pack_message(def_req))
        proc.stdin.flush()

        def_result = None
        while True:
            msg, buf = _lsp_read_message(fd, buf, timeout_sec=6.0)
            if msg is None:
                break
            if msg.get("id") == req_id:
                def_result = msg.get("result")
                break

        if not def_result:
            return "", None, f"{tool_name} definition empty"

        location = None
        if isinstance(def_result, list) and def_result:
            location = def_result[0]
        elif isinstance(def_result, dict):
            location = def_result
        if not isinstance(location, dict):
            return "", None, f"{tool_name} definition parse failed"

        if "targetUri" in location and "targetRange" in location:
            uri = location.get("targetUri", "")
            start = ((location.get("targetRange") or {}).get("start") or {})
            line0 = int(start.get("line", 0))
        else:
            uri = location.get("uri", "")
            start = ((location.get("range") or {}).get("start") or {})
            line0 = int(start.get("line", 0))
        if not uri:
            return "", None, f"{tool_name} definition missing uri"

        if uri.startswith("file://"):
            parsed = urlparse(uri)
            abs_target = parsed.path or ""
            try:
                from urllib.parse import unquote

                abs_target = unquote(abs_target)
            except Exception:
                pass
            if abs_target.startswith("/") and os.name == "nt":
                abs_target = abs_target.lstrip("/")
        else:
            abs_target = uri
        try:
            rel_target = os.path.relpath(abs_target, repo_path).replace("\\", "/")
        except Exception:
            rel_target = abs_target.replace("\\", "/")
        return rel_target, line0 + 1, f"lsp definition via {tool_name}"
    except Exception as exc:
        return "", None, f"{tool_name} error: {exc}"
    finally:
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass


def _probe_tool_version(cmd):
    """Return (ok, version_text_or_reason)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=8,
        )
    except Exception as exc:
        return False, f"exec_error:{exc}"
    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    if proc.returncode == 0 and text:
        return True, text.splitlines()[0][:160]
    if proc.returncode == 0:
        return True, "(no version text)"
    return False, text.splitlines()[0][:160] if text else f"exit={proc.returncode}"


def _definition_line_heuristic(file_text: str, symbol: str):
    """Find likely definition line for a symbol in multi-language source text."""
    if not file_text or not symbol:
        return None
    patterns = [
        rf"(?m)^\s*def\s+{re.escape(symbol)}\s*\(",
        rf"(?m)^\s*class\s+{re.escape(symbol)}\b",
        rf"(?m)^\s*function\s+{re.escape(symbol)}\b",
        rf"(?m)^\s*(?:const|let|var)\s+{re.escape(symbol)}\s*=\s*(?:\(|function\b)",
        rf"(?m)^\s*(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z_][\w<>\[\]]*\s+{re.escape(symbol)}\s*\(",
        rf"(?m)^\s*(?:def|val|var)\s+{re.escape(symbol)}\b",  # Scala
    ]
    for pat in patterns:
        m = re.search(pat, file_text)
        if m:
            return file_text[: m.start()].count("\n") + 1
    return None


def resolve_symbol_in_repo(repo_path: str, symbol: str, preferred_file: str = "", mode: str = "heuristic"):
    """
    Resolve symbol definition across repo.
    Returns (file, line, meta).
    """
    meta = {"mode": mode, "status": "fallback", "details": ""}
    if not symbol:
        meta["details"] = "no symbol"
        return "", None, meta

    # 1) Always start with rg candidates.
    candidates = _run_symbol_rg_candidates(repo_path, symbol, limit=50)
    if not candidates:
        meta["details"] = "rg found no candidates"
        return "", None, meta

    # Prioritize preferred file candidates.
    ordered = candidates
    if preferred_file:
        pf = preferred_file.replace("\\", "/")
        ordered = sorted(candidates, key=lambda c: 0 if c.get("file", "") == pf else 1)

    # 2) LSP mode: try real textDocument/definition first.
    used_tool = ""
    if mode == "lsp":
        exts = {os.path.splitext(c.get("file", ""))[1].lower() for c in ordered}
        if ".py" in exts and shutil.which("pyright-langserver"):
            used_tool = "pyright-langserver"
        elif (".ts" in exts or ".js" in exts) and shutil.which("typescript-language-server"):
            used_tool = "typescript-language-server"
        elif ".go" in exts and shutil.which("gopls"):
            used_tool = "gopls"
        else:
            meta["details"] = "lsp tools not found; fallback to heuristic"
        if used_tool:
            f, ln, det = _lsp_try_definition(repo_path, symbol, ordered, preferred_file=preferred_file, tool_name=used_tool)
            if f and ln:
                meta["status"] = "ok"
                meta["details"] = det
                return f, ln, meta
            meta["details"] = f"{det}; fallback to heuristic"

    # 3) Pick best definition line from candidates.
    for c in ordered:
        rel = c.get("file", "")
        txt = _read_text_safe(repo_path, rel)
        line = _definition_line_heuristic(txt, symbol)
        if line:
            if mode == "lsp" and used_tool:
                meta["status"] = "ok"
                meta["details"] = f"rg candidates + {used_tool} guidance"
            elif mode == "lsp":
                meta["status"] = "fallback"
                if not meta["details"]:
                    meta["details"] = "lsp requested; heuristic used"
            else:
                meta["status"] = "ok"
                meta["details"] = "heuristic definition match"
            return rel, line, meta

    # 4) Last fallback: first rg hit.
    first = ordered[0]
    rel = first.get("file", "")
    line = first.get("line")
    if mode == "lsp":
        meta["status"] = "fallback"
        if not meta["details"]:
            meta["details"] = "lsp requested; used first rg candidate"
    else:
        meta["status"] = "ok"
        meta["details"] = "used first rg candidate"
    return rel, line, meta


def _extract_function_names(text):
    names = set()
    for m in re.finditer(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", text):
        names.add(m.group(1))
    for m in re.finditer(r"(?m)\bfunction\s+([A-Za-z_]\w*)\s*\(", text):
        names.add(m.group(1))
    for m in re.finditer(r"(?m)^\s*(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z_][\w<>\[\]]*\s+([A-Za-z_]\w*)\s*\(", text):
        n = m.group(1)
        if n not in {"if", "for", "while", "switch", "catch", "return"}:
            names.add(n)
    return names


def extract_python_symbols(file_path: str, repo_key: str = "") -> dict:
    """Extract Python symbols using AST."""
    result = {"functions": set(), "classes": set(), "imports": set(), "calls": set(), "ast_skipped": False}
    max_ast_bytes = 500 * 1024
    mtime = None
    if repo_key:
        try:
            mtime = os.path.getmtime(file_path)
            cached = get_cached_ast_symbols(repo_key, file_path, mtime)
            if cached:
                return cached
        except Exception:
            mtime = None
    try:
        if os.path.getsize(file_path) > max_ast_bytes:
            result["ast_skipped"] = True
            return result
    except Exception:
        pass
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
        tree = ast.parse(src)
    except Exception:
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            result["functions"].add(node.name)
        elif isinstance(node, ast.ClassDef):
            result["classes"].add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    result["imports"].add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["imports"].add(node.module)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                result["calls"].add(fn.id)
            elif isinstance(fn, ast.Attribute):
                result["calls"].add(fn.attr)
    if repo_key and mtime is not None:
        set_cached_ast_symbols(repo_key, file_path, mtime, result)
    return result


def detect_structural_dependency(repo_path: str, file_a: str, file_b: str) -> dict:
    """Heuristic structural dependency signals between two files."""
    a = (file_a or "").replace("\\", "/")
    b = (file_b or "").replace("\\", "/")
    text_a = _read_text_safe(repo_path, a)
    text_b = _read_text_safe(repo_path, b)

    stem_b = os.path.splitext(os.path.basename(b))[0]
    module_b = os.path.splitext(b)[0].replace("/", ".")
    import_relation = False
    ast_call_relation = False
    ast_skipped_large = False
    symbol_overlap = 0

    if a.endswith(".py") and b.endswith(".py"):
        abs_a = a if os.path.isabs(a) else os.path.join(repo_path, a)
        abs_b = b if os.path.isabs(b) else os.path.join(repo_path, b)
        rkey = _repo_cache_key(repo_path)
        sym_a = extract_python_symbols(abs_a, repo_key=rkey)
        sym_b = extract_python_symbols(abs_b, repo_key=rkey)
        ast_skipped_large = bool(sym_a.get("ast_skipped")) or bool(sym_b.get("ast_skipped"))
        funcs_b = set(sym_b.get("functions", set()))
        calls_a = set(sym_a.get("calls", set()))
        imports_a = set(sym_a.get("imports", set()))

        called_from_b = calls_a.intersection(funcs_b)
        if called_from_b:
            ast_call_relation = True
            import_relation = True
        symbol_overlap += len(called_from_b)

        # semantic import match: module path or direct module stem mention
        if module_b in imports_a or any(im.endswith("." + stem_b) or im == stem_b for im in imports_a):
            import_relation = True

        shared = sorted(set(sym_a.get("functions", set())).intersection(funcs_b))
    else:
        shared = []

    if text_a and (a.endswith(".py") or b.endswith(".py")):
        import_patterns = [
            rf"(?m)^\s*import\s+.*\b{re.escape(stem_b)}\b",
            rf"(?m)^\s*from\s+.*\b{re.escape(stem_b)}\b\s+import\s+",
            rf"(?m)^\s*from\s+{re.escape(module_b)}\s+import\s+",
        ]
        import_relation = import_relation or any(re.search(p, text_a) for p in import_patterns)
        if a.endswith(".py") and b.endswith(".py"):
            import_relation = import_relation or ast_call_relation

    if not (a.endswith(".py") and b.endswith(".py")):
        funcs_a = _extract_function_names(text_a)
        funcs_b = _extract_function_names(text_b)
        shared = sorted(funcs_a.intersection(funcs_b))
        if text_a and funcs_b:
            for fn in funcs_b:
                if re.search(rf"\b{re.escape(fn)}\s*\(", text_a):
                    symbol_overlap += 1

    dir_a = os.path.dirname(a).strip("/")
    dir_b = os.path.dirname(b).strip("/")
    same_module_tree = (
        bool(dir_a and dir_b)
        and (dir_a == dir_b or dir_a.startswith(dir_b + "/") or dir_b.startswith(dir_a + "/"))
    )

    return {
        "import_relation": bool(import_relation),
        "ast_call_relation": bool(ast_call_relation),
        "symbol_overlap": int(symbol_overlap),
        "shared_functions": shared[:20],
        "same_module_tree": bool(same_module_tree),
        "ast_skipped_large": bool(ast_skipped_large),
    }


def build_import_graph(repo_path: str) -> dict:
    """Build lightweight static import graph for Python files."""
    ignore_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules", "build", "dist", "out", "target"}
    py_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for name in files:
            if name.endswith(".py"):
                abs_path = os.path.join(root, name)
                rel = os.path.relpath(abs_path, repo_path).replace("\\", "/")
                py_files.append(rel)

    module_to_file = {}
    for rel in py_files:
        no_ext = rel[:-3]
        module = no_ext.replace("/", ".")
        module_to_file[module] = rel
        if rel.endswith("/__init__.py"):
            pkg = rel[: -len("/__init__.py")].replace("/", ".")
            if pkg:
                module_to_file[pkg] = rel

    graph = {rel: {"imports": set(), "imported_by": set()} for rel in py_files}

    for rel in py_files:
        abs_path = os.path.join(repo_path, rel)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
            tree = ast.parse(src)
        except Exception:
            continue

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)

        resolved_files = set()
        for mod in imported_modules:
            if mod in module_to_file:
                resolved_files.add(module_to_file[mod])
                continue
            prefix = mod + "."
            for k, f in module_to_file.items():
                if k.startswith(prefix):
                    resolved_files.add(f)
                    break

        graph[rel]["imports"].update(resolved_files)

    for rel, data in graph.items():
        for dep in data["imports"]:
            if dep in graph:
                graph[dep]["imported_by"].add(rel)
    return graph


def compute_impact_propagation(repo_path: str, target_file: str, max_depth=2) -> dict:
    """Compute static dependent files up to depth 2 from import graph."""
    graph = build_import_graph(repo_path)
    t = (target_file or "").replace("\\", "/").strip()
    if not t:
        return {"level_1": [], "level_2": []}

    if t not in graph:
        # fallback by basename match
        bn = os.path.basename(t)
        matched = [k for k in graph if os.path.basename(k) == bn]
        if len(matched) == 1:
            t = matched[0]
        else:
            return {"level_1": [], "level_2": []}

    level_1 = sorted(graph[t]["imported_by"])
    if max_depth < 2:
        return {"level_1": level_1, "level_2": []}

    level_1_set = set(level_1)
    level_2_set = set()
    for f in level_1:
        level_2_set.update(graph.get(f, {}).get("imported_by", set()))
    level_2_set.discard(t)
    level_2_set -= level_1_set
    return {"level_1": level_1, "level_2": sorted(level_2_set)}


def _timeline_signals_from_changed_files(changed_files):
    out = []
    toks = ("tests/", "test", "spec", "ci", "github/workflows")
    joined = " ".join((changed_files or [])).lower()
    if any(t in joined for t in toks):
        out.append("test_or_ci_touched")
    return out


def _timeline_short_note(target_file, changed_files, structured_changes):
    if structured_changes:
        notes = []
        for ch in structured_changes[:2]:
            notes.append(
                f"{ch.get('key', '')}: {ch.get('from', '')} -> {ch.get('to', '')} "
                f"(file={ch.get('file', '')})"
            )
        if notes:
            return "; ".join(notes)
    parts = []
    if target_file:
        parts.append(f"Touched {target_file}.")
    if changed_files:
        parts.append(f"Changed {len(changed_files)} files.")
    if not parts:
        parts.append("Commit metadata captured from git history.")
    return " ".join(parts[:2])


def is_noise_commit(commit_msg: str, changed_files: list[str]) -> bool:
    """Return True if commit is likely noise for decision timeline."""
    msg = (commit_msg or "").lower()
    noisy_terms = ("merge", "delete", "rename", "format", "typo", "wip", "minor", "cleanup")
    if any(t in msg for t in noisy_terms):
        return True

    if changed_files:
        doc_asset_ext = {".md", ".pdf", ".png", ".jpg", ".ipynb", ".parquet", ".joblib"}
        lower_files = [f.lower() for f in changed_files]
        if all(os.path.splitext(f)[1] in doc_asset_ext for f in lower_files):
            return True
    return False


def _git_log_hashes(repo_path, n=50, window_days=None, no_merges=True):
    cmd = ["git", "-C", repo_path, "log", "--format=%H"]
    if no_merges:
        cmd.append("--no-merges")
    if window_days is not None:
        cmd.append(f"--since={int(window_days)} days ago")
    else:
        cmd.extend(["-n", str(max(1, int(n)))])
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def _extract_commits_optimized(repo_path, n=50, window_days=None):
    hashes = _git_log_hashes(repo_path, n=n, window_days=window_days, no_merges=True)
    if not hashes:
        return []
    out = []
    for h in hashes:
        proc = subprocess.run(
            ["git", "-C", repo_path, "show", "--date=short", "--format=%H%x1f%an%x1f%ad%x1f%s", "--unified=3", h],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode != 0:
            continue
        lines = (proc.stdout or "").splitlines()
        if not lines:
            continue
        meta = lines[0].split("\x1f")
        if len(meta) < 4:
            continue
        diff_text = "\n".join(lines[1:]).strip()
        out.append(
            {
                "hash": meta[0].strip(),
                "author": meta[1].strip(),
                "date": meta[2].strip(),
                "message": meta[3].strip(),
                "diff": diff_text,
            }
        )
    return out


def _files_cache_db(repo_key):
    root = os.path.join(".socratic_cache", "index", repo_key)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "files_cache.sqlite")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS files (commit_hash TEXT PRIMARY KEY, files_json TEXT, updated_at REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS patch (commit_hash TEXT PRIMARY KEY, patch_snippet TEXT, updated_at REAL)"
    )
    return conn


def _files_cache_get(repo_key, commit_hash):
    try:
        conn = _files_cache_db(repo_key)
        cur = conn.execute("SELECT files_json FROM files WHERE commit_hash = ?", (commit_hash,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def _files_cache_set(repo_key, commit_hash, files):
    try:
        conn = _files_cache_db(repo_key)
        conn.execute(
            "INSERT OR REPLACE INTO files(commit_hash, files_json, updated_at) VALUES(?,?,?)",
            (commit_hash, json.dumps(files or []), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _patch_cache_get(repo_key, commit_hash):
    try:
        conn = _files_cache_db(repo_key)
        cur = conn.execute("SELECT patch_snippet FROM patch WHERE commit_hash = ?", (commit_hash,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return False, ""
        return True, row[0] or ""
    except Exception:
        return False, ""


def _patch_cache_set(repo_key, commit_hash, patch_snippet):
    try:
        conn = _files_cache_db(repo_key)
        conn.execute(
            "INSERT OR REPLACE INTO patch(commit_hash, patch_snippet, updated_at) VALUES(?,?,?)",
            (commit_hash, patch_snippet or "", time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _name_only_files(repo_path, commit_hash, cache_ctx=None):
    if not commit_hash:
        return []
    repo_key = _repo_cache_key(repo_path)
    cached = _files_cache_get(repo_key, commit_hash)
    if cached is not None:
        if cache_ctx is not None:
            cache_ctx["files_hit"] = cache_ctx.get("files_hit", 0) + 1
        return cached
    if cache_ctx is not None:
        cache_ctx["files_miss"] = cache_ctx.get("files_miss", 0) + 1
    try:
        proc = subprocess.run(
            ["git", "-C", repo_path, "show", "--name-only", "--pretty=format:", commit_hash],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode != 0:
            return []
        files = []
        for ln in (proc.stdout or "").splitlines():
            ln = ln.strip()
            if ln and ln not in files:
                files.append(ln.replace("\\", "/"))
        _files_cache_set(repo_key, commit_hash, files)
        return files
    except Exception:
        return []


def _get_commit_patch_cached(repo_path, commit_hash, cache_ctx=None):
    repo_key = _repo_cache_key(repo_path)
    hit, patch = _patch_cache_get(repo_key, commit_hash)
    if hit:
        if cache_ctx is not None:
            cache_ctx["patch_hit"] = cache_ctx.get("patch_hit", 0) + 1
        return patch
    if cache_ctx is not None:
        cache_ctx["patch_miss"] = cache_ctx.get("patch_miss", 0) + 1
    try:
        proc = subprocess.run(
            ["git", "-C", repo_path, "show", "--unified=3", commit_hash],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        patch = proc.stdout or ""
    except Exception:
        patch = ""
    patch = (patch or "")[: 256 * 1024]
    _patch_cache_set(repo_key, commit_hash, patch)
    return patch


def _get_commit_meta(repo_path, commit_hash):
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                repo_path,
                "show",
                "--date=short",
                "--format=%H%x1f%an%x1f%ad%x1f%B",
                "--no-patch",
                commit_hash,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        out = (proc.stdout or "").splitlines()
        if not out:
            return {}
        head = out[0].split("\x1f")
        if len(head) < 4:
            return {}
        return {
            "hash": head[0].strip(),
            "author": head[1].strip(),
            "date": head[2].strip(),
            "message": "\n".join(out[1:]).strip() or head[3].strip(),
        }
    except Exception:
        return {}


def _get_introducing_commit_cached(repo_path, commit_hash, cache_ctx=None, need_patch=True):
    meta = _get_commit_meta(repo_path, commit_hash)
    if not meta:
        return {}
    changed_files = _name_only_files(repo_path, commit_hash, cache_ctx=cache_ctx)
    patch = ""
    if need_patch:
        patch = _get_commit_patch_cached(repo_path, commit_hash, cache_ctx=cache_ctx)
    return {
        "hash": meta.get("hash", commit_hash),
        "author": meta.get("author", ""),
        "date": meta.get("date", ""),
        "message": meta.get("message", ""),
        "diff": patch,
        "changed_files": changed_files,
    }


def _should_load_full_diff(target_file, symbol_hint, changed_files):
    if os.environ.get("SOCRATIC_PERF_LEGACY", "").strip() == "1":
        return True
    files = [f.replace("\\", "/").lower() for f in (changed_files or [])]
    if not files:
        return False
    if target_file:
        t = target_file.replace("\\", "/").lower()
        tbase = os.path.basename(t)
        if any(f == t or f.endswith("/" + tbase) for f in files):
            return True
    if symbol_hint:
        s = (symbol_hint or "").strip().lower()
        if s and any(s in os.path.basename(f) for f in files):
            return True
    return False


def build_decision_timeline(
    repo_path,
    target_file,
    introducing_commit,
    retrieved_commits,
    local_history_commits,
    max_items=8,
    symbol_hint="",
    cache_ctx=None,
):
    """Build evidence-only timeline sorted by date (old -> new)."""
    by_hash = {}
    introducing_hash = (introducing_commit or {}).get("hash", "")

    def push_commit_like(c):
        if not c:
            return
        h = c.get("hash", "")
        if not h:
            return
        if h not in by_hash:
            by_hash[h] = {
                "hash": h,
                "author": c.get("author", ""),
                "date": c.get("date", ""),
                "message": c.get("message", ""),
            }

    push_commit_like(introducing_commit or {})
    for c in (retrieved_commits or []):
        if isinstance(c, dict) and "commit" in c:
            push_commit_like(c.get("commit", {}))
        else:
            push_commit_like(c or {})
    for c in (local_history_commits or []):
        push_commit_like(c or {})

    if not by_hash:
        return []

    rows = []
    for h, base in by_hash.items():
        changed_files = _name_only_files(repo_path, h, cache_ctx=cache_ctx)
        full = {}
        structured = []
        need_full = h == introducing_hash or _should_load_full_diff(target_file, symbol_hint, changed_files)
        if need_full:
            full = _get_introducing_commit_cached(repo_path, h, cache_ctx=cache_ctx, need_patch=True) or {}
            if not changed_files:
                changed_files = full.get("changed_files", []) or extract_changed_files(full.get("diff", ""))
            structured = detect_structured_changes(full.get("diff", ""))
        msg = (full.get("message", "") or base.get("message", "") or "").splitlines()[0]
        date = (full.get("date", "") or base.get("date", "") or "")
        author = full.get("author", "") or base.get("author", "")
        signals = []
        decision_score = 0
        if h == introducing_hash:
            signals.append("introducing")
            decision_score += 2
        if structured:
            signals.append("structured_change")
            decision_score += 1
        file_classes = {classify_file(f) for f in (changed_files or [])}
        if "CODE" in file_classes:
            signals.append("code_change")
            decision_score += 1
        if "CONFIG" in file_classes:
            signals.append("config_change")
            decision_score += 1
        signals.extend(_timeline_signals_from_changed_files(changed_files))
        refs = extract_refs_from_message(full.get("message", "") or base.get("message", ""))
        for n in refs.get("numbers", [])[:1]:
            signals.append(f"github_ref:#{n}")
        if is_noise_commit(full.get("message", "") or base.get("message", ""), changed_files):
            signals.append("noise")
            decision_score -= 2
        if decision_score <= 0:
            continue
        rows.append(
            {
                "date": date,
                "commit_hash": h[:8],
                "author": author,
                "message_first_line": msg,
                "signals": signals,
                "short_note": _timeline_short_note(target_file, changed_files, structured),
                "decision_score": decision_score,
                "changed_files_count": len(changed_files or []),
            }
        )

    rows.sort(key=lambda x: x.get("date", ""))
    return rows[: max(1, int(max_items))]


def _keyword_index_path(repo_path):
    os.makedirs("data/keyword_index", exist_ok=True)
    return os.path.join("data", "keyword_index", f"{table_name_for_repo(repo_path)}.jsonl")


def has_index(repo_path):
    """Return whether vector/keyword index exists for the repo."""
    table = table_name_for_repo(repo_path)
    vector_path = os.path.join("data", "lancedb", f"{table}.lance")
    keyword_path = _keyword_index_path(repo_path)
    vector_exists = os.path.isdir(vector_path)
    keyword_exists = os.path.isfile(keyword_path)
    return vector_exists, keyword_exists


def _build_keyword_rows(commits):
    rows = []
    for c in commits:
        clean = clean_diff(c.get("diff", "") or "")
        changes = detect_structured_changes(c.get("diff", "") or "")
        change_summary = "\n".join(
            f"CHANGE {ch.get('key', '')} {ch.get('from', '')}->{ch.get('to', '')}" for ch in changes[:10]
        )
        text = ((c.get("message", "") or "") + "\n" + clean + ("\n" + change_summary if change_summary else "")).strip()
        rows.append(
            {
                "hash": c.get("hash", ""),
                "author": c.get("author", ""),
                "date": c.get("date", ""),
                "message": c.get("message", ""),
                "text": text,
                "files": extract_changed_files(c.get("diff", "")),
            }
        )
    return rows


def write_keyword_index(repo_path, commits):
    """Write keyword index JSONL for offline retrieval."""
    path = _keyword_index_path(repo_path)
    rows = _build_keyword_rows(commits)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path, len(rows)


def append_keyword_index(repo_path, commits):
    """Append new commits to keyword index JSONL with hash de-dup."""
    path = _keyword_index_path(repo_path)
    existing = _load_keyword_index(repo_path)
    seen = {r.get("hash", "") for r in existing}
    new_rows = [r for r in _build_keyword_rows(commits) if r.get("hash", "") not in seen]
    if not new_rows:
        return path, 0
    with open(path, "a", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path, len(new_rows)


def _load_keyword_index(repo_path):
    path = _keyword_index_path(repo_path)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def keyword_retrieve(repo_path, query, topk=3, n=200, window_days=None):
    """Simple offline retrieval using keyword overlap on indexed text."""
    rows = _load_keyword_index(repo_path)
    if not rows:
        commits = _extract_commits_optimized(repo_path, n=n, window_days=window_days)
        if not commits:
            commits = extract_commits(repo_path, n=n)
        if not commits:
            return []
        rows = _build_keyword_rows(commits)

    q_tokens = _tokenize_query(query)
    if not q_tokens:
        q_tokens = ["why"]

    scored = []
    for r in rows:
        text = (r.get("text", "") or "").lower()
        score = sum(1 for tok in q_tokens if tok in text)
        scored.append((score, r))

    # Prefer high score; keep recent order for ties.
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, topk)]
    results = []
    for _, r in top:
        results.append(
            {
                "commit": {
                    "hash": r.get("hash", ""),
                    "author": r.get("author", ""),
                    "date": r.get("date", ""),
                    "message": r.get("message", ""),
                    "diff": "",
                },
                "text": r.get("text", ""),
            }
        )
    return results


def resolve_contexts(repo_path, question, topk, retrieval_mode, command_label, verbose=True, window_days=None):
    """Retrieve contexts with vector mode + automatic keyword fallback."""
    topk = max(1, topk)
    mode = retrieval_mode or "vector"
    if mode == "keyword":
        contexts = keyword_retrieve(repo_path, question, topk=topk, window_days=window_days)
        return contexts, "keyword"

    if lancedb is None or SentenceTransformer is None:
        if verbose:
            print("Warning: Embedding model unavailable, falling back to keyword retrieval.")
        contexts = keyword_retrieve(repo_path, question, topk=topk, window_days=window_days)
        return contexts, "keyword"

    try:
        vector_exists, keyword_exists = has_index(repo_path)
        if vector_exists or keyword_exists:
            if vector_exists and keyword_exists:
                kind = "vector|keyword"
            elif vector_exists:
                kind = "vector"
            else:
                kind = "keyword"
            if verbose:
                print(f"[{command_label}] found existing index ({kind})")
        else:
            if verbose:
                print(f"[{command_label}] index not found, auto-indexing latest 50 commits...")
            t0 = time.perf_counter()
            commits = _extract_commits_optimized(repo_path, n=50, window_days=window_days)
            if not commits:
                commits = extract_commits(repo_path, n=50)
            if not commits:
                return [], "vector"
            index_commits(commits)
            t1 = time.perf_counter()
            if verbose:
                print(f"[{command_label}] auto-index done in {t1 - t0:.2f}s")

        contexts = retrieve(question, topk=topk)
        return contexts, "vector"
    except Exception:
        if verbose:
            print("Warning: Embedding model unavailable, falling back to keyword retrieval.")
        contexts = keyword_retrieve(repo_path, question, topk=topk, window_days=window_days)
        return contexts, "keyword"


def run_cmd_at_commit(repo, commit_hash, cmd):
    """Checkout commit in detached mode, run command, and restore original HEAD."""
    original_ref = None
    if repo.head.is_detached:
        original_ref = repo.head.commit.hexsha
    else:
        original_ref = repo.active_branch.name

    try:
        repo.git.checkout("--detach", commit_hash)
        proc = subprocess.run(
            cmd,
            cwd=repo.working_tree_dir,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out_lines = (proc.stdout or "").splitlines()[-200:]
        err_lines = (proc.stderr or "").splitlines()[-200:]
        stdout_tail = "\n".join(out_lines)[-8000:]
        stderr_tail = "\n".join(err_lines)[-8000:]
        return {
            "returncode": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    finally:
        try:
            repo.git.checkout(original_ref)
        except Exception:
            pass


def bisect_search(commits_range, is_bad_fn, max_steps=30):
    """Binary search first bad commit in ordered commits_range (oldest->newest)."""
    steps = []
    tested = {}
    left, right = 0, len(commits_range) - 1
    first_bad = None

    while left <= right and len(steps) < max_steps:
        mid = (left + right) // 2
        commit_hash = commits_range[mid]
        if commit_hash in tested:
            result = tested[commit_hash]
        else:
            result = is_bad_fn(commit_hash)
            tested[commit_hash] = result

        is_bad = bool(result.get("is_bad", False))
        steps.append(
            {
                "commit": commit_hash,
                "status": "bad" if is_bad else "good",
                "returncode": result.get("returncode", -1),
            }
        )
        if is_bad:
            first_bad = commit_hash
            right = mid - 1
        else:
            left = mid + 1

    return first_bad, steps, tested


def parse_github_remote(repo_path):
    """Parse GitHub owner/repo from origin url."""
    try:
        repo = Repo(repo_path)
        origin = next((r for r in repo.remotes if r.name == "origin"), None)
        if origin is None:
            return None
        url = origin.url.strip()
    except Exception:
        return None

    # https://github.com/<owner>/<repo>.git
    if url.startswith("http://") or url.startswith("https://"):
        try:
            parsed = urlparse(url)
            if (parsed.hostname or "").lower() != "github.com":
                return None
            path = (parsed.path or "").strip("/")
            if path.endswith(".git"):
                path = path[:-4]
            parts = path.split("/")
            if len(parts) >= 2:
                return {"owner": parts[0], "repo": parts[1]}
        except Exception:
            return None

    # git@github.com:<owner>/<repo>.git
    m = re.match(r"^git@github\.com:([^/]+)/(.+?)(?:\.git)?$", url)
    if m:
        return {"owner": m.group(1), "repo": m.group(2)}

    return None


def extract_pr_issue_refs(text):
    """Extract PR refs from text; issue refs kept optional/unknown."""
    src = text or ""
    prs = set()
    issues = set()
    for pat in [
        r"Merge pull request\s+#(\d+)",
        r"\bPR\s*#(\d+)\b",
        r"\(#(\d+)\)",
        r"\B#(\d+)\b",
    ]:
        for m in re.finditer(pat, src, flags=re.IGNORECASE):
            try:
                prs.add(int(m.group(1)))
            except Exception:
                pass
    return {"prs": sorted(prs), "issues": sorted(issues)}


def extract_refs_from_message(message: str) -> dict:
    """Extract issue/PR references like #123, Fixes #123, PR #123."""
    src = message or ""
    numbers = []
    raw_hits = []
    seen = set()

    patterns = [
        r"(?i)\b(Fixes|Closes|Resolves|Refs)\s+#(\d+)\b",
        r"(?i)\bPR\s+#(\d+)\b",
        r"\(#(\d+)\)",
        r"\B#(\d+)\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, src):
            if len(numbers) >= 10:
                break
            if m.lastindex is None:
                continue
            num_group = m.group(m.lastindex)
            try:
                n = int(num_group)
            except Exception:
                continue
            if n in seen:
                continue
            seen.add(n)
            numbers.append(n)
            raw_hits.append(m.group(0))
        if len(numbers) >= 10:
            break
    return {"numbers": numbers, "raw": raw_hits}


def _github_cache_path(owner: str, repo_name: str, number: int) -> str:
    cache_dir = os.path.join(".socratic_cache", "github")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{owner}_{repo_name}_{number}.json")


def _repo_cache_key(repo_path: str) -> str:
    return hashlib.sha1(os.path.realpath(repo_path).encode("utf-8")).hexdigest()[:16]


def _repo_cache_dir(repo_key: str) -> str:
    p = os.path.join(".socratic_cache", "index", repo_key)
    os.makedirs(p, exist_ok=True)
    return p


def load_index_state(repo_key) -> dict:
    path = os.path.join(_repo_cache_dir(repo_key), "index_state.json")
    state = {"indexed_commits": set(), "last_index_time": "", "retrieval_mode": "mixed"}
    if not os.path.exists(path):
        return state
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        state["indexed_commits"] = set(payload.get("indexed_commits", []) or [])
        state["last_index_time"] = payload.get("last_index_time", "")
        state["retrieval_mode"] = payload.get("retrieval_mode", "mixed")
    except Exception:
        pass
    return state


def save_index_state(repo_key, state) -> None:
    path = os.path.join(_repo_cache_dir(repo_key), "index_state.json")
    payload = {
        "indexed_commits": sorted(list(state.get("indexed_commits", set()))),
        "last_index_time": state.get("last_index_time", ""),
        "retrieval_mode": state.get("retrieval_mode", "mixed"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_embed_cache_meta(repo_key):
    path = os.path.join(_repo_cache_dir(repo_key), "embed_cache_meta.json")
    if not os.path.exists(path):
        return {"indexed_commits": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"indexed_commits": []}


def _save_embed_cache_meta(repo_key, data):
    path = os.path.join(_repo_cache_dir(repo_key), "embed_cache_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _ast_cache_db_path(repo_key):
    return os.path.join(_repo_cache_dir(repo_key), "ast_cache.sqlite")


def _ast_cache_conn(repo_key):
    conn = sqlite3.connect(_ast_cache_db_path(repo_key))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ast_cache (path TEXT PRIMARY KEY, mtime REAL, symbols_json TEXT)"
    )
    conn.commit()
    return conn


def get_cached_ast_symbols(repo_key, file_path, file_mtime) -> dict:
    try:
        conn = _ast_cache_conn(repo_key)
        cur = conn.execute("SELECT mtime, symbols_json FROM ast_cache WHERE path = ?", (file_path,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return {}
        mtime, symbols_json = row
        if float(mtime) != float(file_mtime):
            return {}
        payload = json.loads(symbols_json)
        return {
            "functions": set(payload.get("functions", [])),
            "classes": set(payload.get("classes", [])),
            "imports": set(payload.get("imports", [])),
            "calls": set(payload.get("calls", [])),
        }
    except Exception:
        return {}


def set_cached_ast_symbols(repo_key, file_path, file_mtime, symbols) -> None:
    try:
        payload = {
            "functions": sorted(list(symbols.get("functions", set()))),
            "classes": sorted(list(symbols.get("classes", set()))),
            "imports": sorted(list(symbols.get("imports", set()))),
            "calls": sorted(list(symbols.get("calls", set()))),
        }
        conn = _ast_cache_conn(repo_key)
        conn.execute(
            "INSERT OR REPLACE INTO ast_cache(path, mtime, symbols_json) VALUES(?,?,?)",
            (file_path, float(file_mtime), json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()
    except Exception:
        return


def _github_enabled(args):
    token = (getattr(args, "github_token", "") or "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    explicit = getattr(args, "github_evidence", None)
    if getattr(args, "no_github", False):
        return False, token
    if explicit == "on":
        return True, token
    if explicit == "off":
        return False, token
    # default: off, unless token provided
    return bool(token), token


def fetch_github_issue_optional(owner, repo_name, number, token="", timeout=10):
    """Fetch issue/PR info with local cache; never raises."""
    path = _github_cache_path(owner, repo_name, number)
    ttl = 7 * 24 * 3600
    now = time.time()

    if os.path.exists(path):
        try:
            st = os.stat(path)
            if now - st.st_mtime <= ttl:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_source"] = "cache"
                return data, None
        except Exception:
            pass

    url = f"https://api.github.com/repos/{owner}/{repo_name}/issues/{number}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "socratic-git"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib_request.Request(url, headers=headers, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib_error.HTTPError as e:
        return None, f"http_{e.code}"
    except urllib_error.URLError as e:
        return None, f"network_error:{e.reason}"
    except Exception as e:
        return None, f"error:{type(e).__name__}"

    body = re.sub(r"\s+", " ", (payload.get("body") or "")).strip()[:1200]
    data = {
        "number": number,
        "title": payload.get("title", ""),
        "body_snippet": body,
        "state": payload.get("state", ""),
        "html_url": payload.get("html_url", ""),
        "author": (payload.get("user") or {}).get("login", ""),
        "created_at": payload.get("created_at", ""),
        "is_pr": bool(payload.get("pull_request")),
        "_source": "network",
        "_url": url,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data, None


def collect_linked_github_evidence(repo_path, commits_like, args):
    """Collect linked GitHub issues/PRs from commit messages."""
    enabled, token = _github_enabled(args)
    timeout = int(getattr(args, "github_timeout", 10) or 10)
    remote = parse_github_remote(repo_path)
    result = {"enabled": enabled, "owner": "", "repo": "", "items": [], "errors": [], "numbers": [], "urls": []}

    if not enabled:
        result["errors"].append("GitHub evidence disabled.")
        return result
    if remote is None:
        result["errors"].append("GitHub remote not detected; skipping GitHub evidence.")
        return result

    owner = remote.get("owner", "")
    repo_name = remote.get("repo", "")
    result["owner"] = owner
    result["repo"] = repo_name

    seen = set()
    numbers = []
    for msg in commits_like:
        refs = extract_refs_from_message(msg)
        for n in refs.get("numbers", []):
            if n in seen:
                continue
            seen.add(n)
            numbers.append(n)
            if len(numbers) >= 20:
                break
        if len(numbers) >= 20:
            break
    result["numbers"] = numbers

    if not numbers:
        result["errors"].append("No issue/PR references found in commit messages.")
        return result

    for n in numbers:
        url = f"https://api.github.com/repos/{owner}/{repo_name}/issues/{n}"
        result["urls"].append(url)
        item, err = fetch_github_issue_optional(owner, repo_name, n, token=token, timeout=timeout)
        if item is not None:
            result["items"].append(item)
        elif err:
            result["errors"].append(f"#{n}: {err}")

    return result


def get_commit_message_body(repo_path, commit_hash):
    """Get full commit message body."""
    if not commit_hash:
        return ""
    try:
        repo = Repo(repo_path)
        return repo.git.show("-s", "--format=%B", commit_hash)
    except Exception:
        return ""


def fetch_github_pr(repo_path, owner, repo_name, pr_number, token=None):
    """Fetch PR details with 7-day local cache."""
    cache_dir = os.path.join(repo_path, ".socratic_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"github_pr_{owner}_{repo_name}_{pr_number}.json")
    now = time.time()
    ttl = 7 * 24 * 3600

    if os.path.exists(cache_path):
        try:
            st = os.stat(cache_path)
            if now - st.st_mtime <= ttl:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
        except Exception:
            pass

    if requests is None:
        return None

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except Exception:
        return None

    data = {
        "number": pr_number,
        "title": payload.get("title", ""),
        "body": (payload.get("body", "") or "")[:1200],
        "author": (payload.get("user") or {}).get("login", ""),
        "created_at": payload.get("created_at", ""),
        "merged_at": payload.get("merged_at", ""),
        "html_url": payload.get("html_url", ""),
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data


def build_github_pr_evidence(repo_path, commit_hash, disabled=False):
    """Build optional PR evidence block from commit refs."""
    if disabled or not commit_hash:
        return None
    # Always ensure cache directory exists when enhancement is enabled.
    os.makedirs(os.path.join(repo_path, ".socratic_cache"), exist_ok=True)
    gh = parse_github_remote(repo_path)
    if gh is None:
        return None

    full_msg = get_commit_message_body(repo_path, commit_hash)
    refs = extract_pr_issue_refs(full_msg)
    prs = refs.get("prs", [])
    if not prs:
        return None

    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    pr_no = prs[0]
    if requests is None:
        return {
            "available": False,
            "pr_number": pr_no,
            "text": "PR evidence unavailable (requests not installed; pip install requests)",
            "title": "",
            "body": "",
            "url": "",
        }
    pr = fetch_github_pr(repo_path, gh["owner"], gh["repo"], pr_no, token=token)
    if pr is None:
        return {
            "available": False,
            "pr_number": pr_no,
            "text": "PR evidence unavailable (rate limit or not found)",
            "title": "",
            "body": "",
            "url": "",
        }

    body_snippet = (pr.get("body", "") or "").strip()
    text = (
        "## GitHub PR Evidence (Optional)\n"
        f"- pr: #{pr_no}\n"
        f"- title: {pr.get('title', '')}\n"
        f"- author: {pr.get('author', '')}\n"
        f"- created/merged: {pr.get('created_at', '')} / {pr.get('merged_at', '')}\n"
        f"- url: {pr.get('html_url', '')}\n"
        f"- body_snippet:\n{body_snippet if body_snippet else '(empty)'}"
    )
    return {
        "available": True,
        "pr_number": pr_no,
        "text": text,
        "title": pr.get("title", ""),
        "body": pr.get("body", ""),
        "url": pr.get("html_url", ""),
    }


def _extract_motive_confidence(answer_text):
    m = re.search(r"Motive Confidence:\s*(High|Medium|Low)", answer_text or "")
    return m.group(1) if m else "Low"


def _html_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _decorate_report_text_html(text: str) -> str:
    escaped = _html_escape(text or "")
    escaped = re.sub(
        r"\b([0-9a-f]{7,40})\b",
        r'<span class="commit-hash">\1</span>',
        escaped,
    )
    escaped = re.sub(
        r"\b([A-Za-z0-9_.\-/]+/[A-Za-z0-9_.\-/]+\.[A-Za-z0-9_]+)\b",
        r'<span class="file-path">\1</span>',
        escaped,
    )
    return escaped


def render_html_report(markdown_text: str, structured_data: dict) -> str:
    """Render a self-contained HTML report from markdown + structured metadata."""
    lines = (markdown_text or "").splitlines()
    title = "Socratic Git Report"
    sections = []
    current_title = ""
    current_lines = []
    for ln in lines:
        if ln.startswith("# "):
            title = ln[2:].strip() or title
            continue
        if ln.startswith("## "):
            if current_title:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = ln[3:].strip()
            current_lines = []
        else:
            current_lines.append(ln)
    if current_title:
        sections.append((current_title, "\n".join(current_lines).strip()))

    repo_path = (structured_data or {}).get("repo_path", "")
    question = (structured_data or {}).get("question", "")
    motive_score = int((structured_data or {}).get("motive_score", 0) or 0)
    motive_conf = (structured_data or {}).get("motive_confidence", "Low")
    bar_width = max(0, min(100, motive_score * 10))

    details_html = []
    for i, (sec_title, sec_body) in enumerate(sections):
        body_html = _decorate_report_text_html(sec_body)
        details_html.append(
            (
                f"<details {'open' if i < 3 else ''}>"
                f"<summary>{_html_escape(sec_title)}</summary>"
                f"<pre>{body_html if body_html else '(empty)'}</pre>"
                f"</details>"
            )
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html_escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin: 0 0 12px 0; }}
    .meta {{ background: #f6f8fa; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 16px; }}
    .meta-line {{ margin: 4px 0; }}
    .score-wrap {{ margin-top: 8px; }}
    .score-bar {{ width: 260px; height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden; border: 1px solid #d1d5db; }}
    .score-fill {{ height: 100%; width: {bar_width}%; background: #2563eb; }}
    details {{ border: 1px solid #ddd; border-radius: 8px; margin: 10px 0; padding: 8px 10px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 10px 0 0 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .commit-hash {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #eef2ff; padding: 0 4px; border-radius: 4px; }}
    .file-path {{ color: #0b5; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>{_html_escape(title)}</h1>
  <div class="meta">
    <div class="meta-line"><strong>Project Path:</strong> {_html_escape(repo_path)}</div>
    <div class="meta-line"><strong>Question:</strong> {_html_escape(question)}</div>
    <div class="meta-line"><strong>Motive Confidence:</strong> {_html_escape(motive_conf)}</div>
    <div class="score-wrap">
      <div><strong>Motive Score:</strong> {motive_score}/10</div>
      <div class="score-bar"><div class="score-fill"></div></div>
    </div>
  </div>
  {''.join(details_html)}
</body>
</html>"""


def run_trace_structured(repo_path: str, question: str, args_like: dict) -> dict:
    """Run trace pipeline and return structured result + markdown report without writing files."""
    topk = max(1, int((args_like or {}).get("topk", 3)))
    retrieval = (args_like or {}).get("retrieval", "vector")
    no_github = bool((args_like or {}).get("no_github", False))
    github_evidence = (args_like or {}).get("github_evidence", None)
    github_timeout = int((args_like or {}).get("github_timeout", 10))
    cochange_mode = (args_like or {}).get("cochange_mode", "code")
    window_days = (args_like or {}).get("window_days", None)
    symbol_resolver = (args_like or {}).get("symbol_resolver", "heuristic")
    verbose = bool((args_like or {}).get("verbose", False))
    cache_ctx = {"files_hit": 0, "files_miss": 0, "patch_hit": 0, "patch_miss": 0}

    class _ArgsProxy:
        def __init__(self):
            self.no_github = no_github
            self.github_token = (args_like or {}).get("github_token", "")
            self.github_evidence = github_evidence
            self.github_timeout = github_timeout
            self.cochange_mode = cochange_mode

    args_proxy = _ArgsProxy()

    contexts, _used_retrieval = resolve_contexts(
        repo_path,
        question,
        topk=topk,
        retrieval_mode=retrieval,
        command_label="trace",
        verbose=verbose,
        window_days=window_days,
    )
    if not contexts:
        raise RuntimeError("no related commits retrieved")

    target_info = get_symbol_or_lines_from_query(question)
    detected_file = target_info.get("file")
    detected_line = target_info.get("line")
    detected_symbol = target_info.get("symbol")
    resolved_line = None
    symbol_resolution = {"mode": symbol_resolver, "status": "not_used", "details": ""}
    blame_info = ""
    introducing_commit = {}
    intro_files = []
    intro_msg_preview = ""
    intro_diff_preview = ""
    local_history_window = []
    github_pr_evidence = None

    if detected_symbol and not detected_line:
        # First try in-file fast path when file is provided (heuristic mode only).
        if detected_file and symbol_resolver != "lsp":
            resolved_line = find_symbol_definition(repo_path, detected_file, detected_symbol)
            if resolved_line:
                detected_line = resolved_line
                symbol_resolution = {
                    "mode": symbol_resolver,
                    "status": "ok",
                    "details": "symbol found in detected file",
                }

        # Fallback to repo-wide resolver (heuristic/lsp).
        if not detected_line:
            resolved_file, resolved_line2, meta = resolve_symbol_in_repo(
                repo_path=repo_path,
                symbol=detected_symbol,
                preferred_file=detected_file or "",
                mode=symbol_resolver,
            )
            symbol_resolution = meta
            if resolved_file and resolved_line2:
                detected_file = resolved_file
                resolved_line = resolved_line2
                detected_line = resolved_line2
            if symbol_resolver == "lsp" and verbose and meta.get("status") == "fallback":
                print(f"Warning: lsp symbol resolver fallback: {meta.get('details', '')}")

    if detected_file and detected_line:
        blame_dict = get_blame_for_line(repo_path, detected_file, detected_line)
        if blame_dict:
            blame_info = (
                f"file={blame_dict.get('file', '')}; "
                f"line={blame_dict.get('line', '')}; "
                f"commit={blame_dict.get('hash', '')}; "
                f"author={blame_dict.get('author', '')}; "
                f"date={blame_dict.get('date', '')}; "
                f"line_content={blame_dict.get('line_content', '')}"
            )
            introducing_commit = _get_introducing_commit_cached(
                repo_path, blame_dict.get("hash", ""), cache_ctx=cache_ctx, need_patch=True
            )
    elif detected_file:
        blame_info = get_file_blame(repo_path, detected_file)

    if introducing_commit:
        intro_msg_lines = introducing_commit.get("message", "").splitlines()[:10]
        intro_msg_preview = "\n".join(intro_msg_lines)
        intro_diff_clean = clean_diff(introducing_commit.get("diff", ""))
        intro_diff_lines = intro_diff_clean.splitlines()[:120]
        intro_diff_preview = "\n".join(intro_diff_lines)
        intro_files = introducing_commit.get("changed_files", []) or extract_changed_files(
            introducing_commit.get("diff", "")
        )
        github_pr_evidence = build_github_pr_evidence(
            repo_path,
            introducing_commit.get("hash", ""),
            disabled=no_github,
        )

    if detected_file:
        local_history_window = get_local_history_window(repo_path, detected_file, n=5)

    commit_msgs_for_refs = []
    if introducing_commit:
        commit_msgs_for_refs.append(introducing_commit.get("message", ""))
    for ctx in contexts:
        commit_msgs_for_refs.append(ctx.get("commit", {}).get("message", ""))
    for row in local_history_window:
        commit_msgs_for_refs.append(row.get("message", ""))
    linked_github = collect_linked_github_evidence(repo_path, commit_msgs_for_refs, args_proxy)

    used_hashes = []
    seen_hash = set()

    def add_hash(h):
        if h and h not in seen_hash:
            seen_hash.add(h)
            used_hashes.append(h)

    add_hash(introducing_commit.get("hash", ""))
    for row in local_history_window:
        add_hash(row.get("hash", ""))
    for ctx in contexts:
        add_hash(ctx.get("commit", {}).get("hash", ""))

    structured_changes = detect_structured_changes(introducing_commit.get("diff", "")) if introducing_commit else []
    selected_github = select_github_evidence(question, structured_changes, linked_github.get("items", []), topn=3)

    llm_contexts = contexts
    if introducing_commit:
        introducing_details_text = (
            "=== Introducing Commit Details ===\n"
            f"hash: {introducing_commit.get('hash', '')}\n"
            f"author/date: {introducing_commit.get('author', '')} / {introducing_commit.get('date', '')}\n"
            f"message:\n{intro_msg_preview}\n"
            f"changed_files: {', '.join(intro_files) if intro_files else '(unavailable)'}\n"
            f"diff_snippet: {intro_diff_preview if intro_diff_preview else '(unavailable)'}"
        )
        llm_contexts = [{"commit": introducing_commit, "text": introducing_details_text}] + llm_contexts
    if github_pr_evidence:
        llm_contexts = [
            {
                "commit": {
                    "hash": f"pr-{github_pr_evidence.get('pr_number', '')}",
                    "author": "github",
                    "date": "",
                    "message": "GitHub PR Evidence",
                    "diff": "",
                },
                "text": github_pr_evidence.get("text", ""),
            }
        ] + llm_contexts
    if selected_github:
        for item in selected_github:
            llm_contexts = [
                {
                    "commit": {
                        "hash": f"github-issue-{item.get('number')}",
                        "author": item.get("author", "github"),
                        "date": item.get("created_at", ""),
                        "message": f"GitHub Issue/PR #{item.get('number')}",
                        "diff": "",
                    },
                    "text": (
                        f"[GITHUB_ISSUE #{item.get('number')}] {item.get('title','')}\n"
                        f"{item.get('body_snippet','')}"
                    ),
                }
            ] + llm_contexts
    if local_history_window:
        history_lines = []
        for row in local_history_window:
            msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
            history_lines.append(f"{row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())
        history_text = "=== Local History Window (file) ===\n" + "\n".join(history_lines)
        llm_contexts = llm_contexts + [
            {
                "commit": {
                    "hash": "local-history",
                    "author": "git log",
                    "date": "",
                    "message": "Recent file-local history window",
                    "diff": "",
                },
                "text": history_text,
            }
        ]

    intent_info = detect_query_intent(question)
    intro_text_for_judgement = (
        ((introducing_commit.get("message", "") or "") + "\n" + (introducing_commit.get("diff", "") or ""))
        .lower()
    )
    if github_pr_evidence and github_pr_evidence.get("available"):
        intro_text_for_judgement += "\n" + (github_pr_evidence.get("title", "") or "").lower()
        intro_text_for_judgement += "\n" + (github_pr_evidence.get("body", "") or "").lower()
    for item in selected_github:
        intro_text_for_judgement += "\n" + (item.get("title", "") or "").lower()
        intro_text_for_judgement += "\n" + (item.get("body_snippet", "") or "").lower()
    motive_keywords = (
        "because",
        "so that",
        "in order to",
        "to improve",
        "motivation",
        "reason",
        "why",
        "fix timeout",
        "latency",
    )
    has_explicit_motive = any(k in intro_text_for_judgement for k in motive_keywords)
    why_chain = build_why_chain(resolved_line or detected_line, introducing_commit.get("hash", ""), selected_github)

    findings_text = (
        f"Question target: file={detected_file or '(not detected)'}, line={detected_line or '(not detected)'}, "
        f"symbol={detected_symbol or '(not detected)'}; "
    )
    if introducing_commit:
        findings_text += (
            f"introducing commit={introducing_commit.get('hash', '')} "
            f"({introducing_commit.get('date', '')}, {introducing_commit.get('author', '')}), "
            f"message='{(introducing_commit.get('message', '') or '').splitlines()[0]}'."
        )
    else:
        findings_text += "introducing commit not found; using retrieved commit evidence."

    motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
    if _llm_disabled():
        motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
    else:
        try:
            motive_prompt = (
                "You only write the Motive paragraph (max 3 sentences).\n"
                "Use provided commit/PR evidence only.\n"
                "If motive is not explicit, output exactly: Motive is not explicitly stated in commit/issue evidence."
            )
            motive_raw = _generate_answer(motive_prompt, llm_contexts)
            motive_text_out = _truncate_sentences(sanitize_llm_output(motive_raw), max_sentences=3)
            if not motive_text_out:
                motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
        except Exception:
            motive_text_out = "Motive is not explicitly stated in commit/issue evidence."

    limitations = []
    if not detected_file:
        limitations.append("- Query did not include a concrete file path; target detection may be weak.")
    if detected_symbol and not detected_line and not resolved_line:
        limitations.append("- Symbol was detected but could not be resolved to a line number.")
    if detected_file and not introducing_commit:
        limitations.append("- Introducing commit not found; report falls back to file-level evidence.")
    if introducing_commit and not introducing_commit.get("diff", "").strip():
        limitations.append("- Introducing commit diff is empty or unavailable.")
    generic_count = 0
    for ctx in contexts:
        first_line = (ctx.get("commit", {}).get("message", "") or "").splitlines()[0].lower()
        if first_line.startswith("update") or first_line.startswith("updating") or first_line.startswith("merge"):
            generic_count += 1
    if generic_count >= max(1, len(contexts) - 1):
        limitations.append("- Retrieved commit messages are generic, so why-level intent may be under-specified.")
    if "我不知道" in motive_text_out:
        limitations.append("- Evidence was insufficient for a confident why-explanation.")
    if not limitations:
        limitations.append("- No major limitations detected for this trace run.")

    cochange_targets = []
    if detected_file:
        cochange_targets = [detected_file]
    elif intro_files:
        cochange_targets = intro_files[:3]
    else:
        guessed = []
        for ctx in contexts:
            files = extract_changed_files(ctx.get("commit", {}).get("diff", ""))[:3]
            if not files:
                files = _changed_files_for_hash(repo_path, ctx.get("commit", {}).get("hash", ""))[:3]
            for f in files:
                if f not in guessed:
                    guessed.append(f)
            if len(guessed) >= 3:
                break
        cochange_targets = guessed[:3]
    cochange_top = []
    cochange_error = ""
    cochange_window = 0
    old_window_days_env = os.environ.get("SOCRATIC_WINDOW_DAYS")
    if window_days is not None:
        os.environ["SOCRATIC_WINDOW_DAYS"] = str(window_days)
    try:
        if cochange_targets:
            cochange_top = compute_cochange(
                repo_path, cochange_targets, n_commits=300, mode=cochange_mode
            )
            cochange_window = int(getattr(compute_cochange, "last_window_commits", 0) or 0)
        else:
            cochange_error = "no target files detected"
    except Exception as exc:
        cochange_error = str(exc)
    finally:
        if old_window_days_env is None:
            os.environ.pop("SOCRATIC_WINDOW_DAYS", None)
        else:
            os.environ["SOCRATIC_WINDOW_DAYS"] = old_window_days_env

    impact_target = ""
    impact_prop = {"level_1": [], "level_2": []}
    impact_error = ""
    if detected_file and classify_file(detected_file) == "CODE" and detected_file.endswith(".py"):
        impact_target = detected_file
        try:
            impact_prop = compute_impact_propagation(repo_path, impact_target, max_depth=2)
        except Exception as exc:
            impact_error = str(exc)

    timeline_items = build_decision_timeline(
        repo_path=repo_path,
        target_file=detected_file or impact_target or "",
        introducing_commit=introducing_commit if introducing_commit else {},
        retrieved_commits=contexts,
        local_history_commits=local_history_window,
        max_items=8,
        symbol_hint=detected_symbol or "",
        cache_ctx=cache_ctx,
    )
    motive_evidence = build_motive_evidence(
        repo_path=repo_path,
        query=question,
        evidence={
            "selected_github_evidence": selected_github,
            "introducing_commit": introducing_commit,
            "retrieved_commits": contexts,
            "structured_changes": structured_changes,
            "changed_files": intro_files,
            "impact_propagation": impact_prop,
            "cochange_top": cochange_top if isinstance(cochange_top, dict) else {},
            "timeline_items": timeline_items,
        },
    )
    answer = build_answer_template(
        findings=findings_text,
        motive=motive_text_out,
        has_explicit_motive=has_explicit_motive,
        structured_changes=structured_changes,
        evidence_text=intro_text_for_judgement,
        missing_items=[],
        intent_info=intent_info,
        changed_files=intro_files,
        selected_github_evidence=selected_github,
        why_chain=why_chain,
        motive_evidence=motive_evidence,
    )
    if verbose:
        saved = cache_ctx.get("files_hit", 0) + cache_ctx.get("patch_hit", 0)
        print(f"files_cache: hit={cache_ctx.get('files_hit', 0)} miss={cache_ctx.get('files_miss', 0)}")
        print(f"patch_cache: hit={cache_ctx.get('patch_hit', 0)} miss={cache_ctx.get('patch_miss', 0)}")
        print(f"git_show_calls_saved: {saved}")

    lines = [
        "# Socratic Git Trace Report",
        "",
        "## Question",
        question,
        "",
        "## Detected Target",
        f"- file: {detected_file or '(not detected)'}",
        f"- symbol: {detected_symbol or '(not detected)'}",
        f"- line: {detected_line if detected_line else '(not detected)'}",
        f"- resolved_line: {resolved_line if resolved_line else '(none)'}",
        (
            f"- Symbol Resolution: mode={symbol_resolution.get('mode','heuristic')}, "
            f"status={symbol_resolution.get('status','not_used')}, "
            f"details={symbol_resolution.get('details','')}"
            if detected_symbol
            else "- Symbol Resolution: (not used)"
        ),
        "",
        "## Introducing Commit",
    ]
    if introducing_commit:
        intro_msg = introducing_commit.get("message", "").splitlines()[0] if introducing_commit.get("message") else ""
        lines.extend(
            [
                f"- hash: {introducing_commit.get('hash', '')}",
                f"- author: {introducing_commit.get('author', '')}",
                f"- date: {introducing_commit.get('date', '')}",
                f"- message: {intro_msg}",
            ]
        )
    else:
        lines.append("- not found")
    lines.extend(["", "## Introducing Commit Details"])
    if introducing_commit:
        lines.extend(
            [
                f"- hash: {introducing_commit.get('hash', '')}",
                f"- author/date: {introducing_commit.get('author', '')} / {introducing_commit.get('date', '')}",
                "- message (first 10 lines):",
                "```text",
                intro_msg_preview or "(empty)",
                "```",
                f"- changed_files: {', '.join(intro_files) if intro_files else '(unavailable)'}",
                (
                    "- diff_snippet: (unavailable)"
                    if not intro_diff_preview
                    else "- diff_snippet (clean_diff + first 120 lines):"
                ),
            ]
        )
        if intro_diff_preview:
            lines.extend(["```diff", intro_diff_preview, "```"])
        else:
            lines.extend(["```diff", "(unavailable)", "```"])
    else:
        lines.append("- not available")
    lines.extend(["", "## Local History Window"])
    if local_history_window:
        for row in local_history_window:
            msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
            lines.append(f"- {row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())
    else:
        lines.append("- not available")
    lines.extend(["", "## Retrieved Commits (TopK)"])
    for i, ctx in enumerate(contexts[:topk], start=1):
        c = ctx.get("commit", {})
        first = (c.get("message", "") or "").splitlines()[0]
        files = extract_changed_files(c.get("diff", ""))
        lines.extend(
            [
                f"- [{i}] {c.get('hash', '')} {c.get('date', '')} {c.get('author', '')}",
                f"  - message: {first}",
                f"  - files: {', '.join(files) if files else '(none)'}",
            ]
        )
    lines.extend(["", "## Structured Changes"])
    if structured_changes:
        for ch in structured_changes[:10]:
            lb = (ch.get("line_before", "") or "")[:120]
            la = (ch.get("line_after", "") or "")[:120]
            lines.append(
                f"- {ch.get('key', '')}: {ch.get('from', '')} -> {ch.get('to', '')} "
                f"(file={ch.get('file', '')}, hunk={ch.get('hunk', '')})"
            )
            lines.append(f"  - line_before: {lb}")
            lines.append(f"  - line_after: {la}")
    else:
        lines.append("- (none detected)")

    lines.extend(
        [
            "",
            "## Cross-File Impact (Co-change)",
            f"- target_files: {', '.join(cochange_targets) if cochange_targets else '(none)'}",
            f"- window_commits: {cochange_window}",
            "",
            "### Top Related Code Files",
        ]
    )
    if cochange_top and cochange_top.get("top_code"):
        dep_target = (cochange_targets or [""])[0]
        for row in cochange_top.get("top_code", []):
            lines.append(
                f"- file={row.get('file')}, co_count={row.get('co_count')}, "
                f"lift={row.get('lift')}, p_other={row.get('p_other')}, "
                f"p_other_given_target={row.get('p_other_given_target')}"
            )
            dep = detect_structural_dependency(repo_path, dep_target, row.get("file", ""))
            try:
                lift_value = float(row.get("lift", 0) or 0)
            except Exception:
                lift_value = 0.0
            if lift_value > 2 and dep.get("ast_call_relation") is True:
                structural_confidence = "High"
            elif dep.get("import_relation") is True:
                structural_confidence = "Medium"
            else:
                structural_confidence = "Low"
            lines.append("  - Structural Dependency Signals")
            lines.append(f"  - structural_dependency: {dep}")
            lines.append(f"  - structural_confidence: {structural_confidence}")
            if dep.get("ast_skipped_large"):
                lines.append("  - AST skipped: file too large")
    else:
        lines.append("- (none detected)")
    lines.extend(["", "### Top Related Config Files"])
    if cochange_top and cochange_top.get("top_config"):
        for row in cochange_top.get("top_config", []):
            lines.append(
                f"- file={row.get('file')}, co_count={row.get('co_count')}, "
                f"lift={row.get('lift')}, p_other={row.get('p_other')}, "
                f"p_other_given_target={row.get('p_other_given_target')}"
            )
    else:
        lines.append("- (none detected)")
    lines.extend(["", "### Ignored (Artifacts / Data / Build)"])
    if cochange_top and cochange_top.get("ignored"):
        for row in cochange_top.get("ignored", []):
            lines.append(
                f"- file={row.get('file')}, class={row.get('class')}, co_count={row.get('co_count')}, "
                f"lift={row.get('lift')}"
            )
    else:
        lines.append("- (none detected)")
    if cochange_error:
        lines.append(f"- cochange unavailable: {cochange_error}")
    lines.append("- interpretation: This suggests structural coupling (statistical), not proven causality.")

    lines.extend(["", "### Coupling Summary"])
    summary_lines = []
    top_code_rows = (cochange_top or {}).get("top_code", [])
    top_config_rows = (cochange_top or {}).get("top_config", [])
    if top_code_rows:
        for row in top_code_rows[:3]:
            summary_lines.append(
                f"- {', '.join(cochange_targets[:1] or ['target files'])} and {row.get('file')} frequently co-change, "
                "suggesting they may be part of the same implementation flow."
            )
    if top_config_rows:
        for row in top_config_rows[:1]:
            summary_lines.append(
                f"- Config file {row.get('file')} likely changes with target logic, suggesting runtime/threshold alignment may be needed."
            )
    if not summary_lines:
        summary_lines.append("- (none detected)")
    lines.extend(summary_lines[:4])

    lines.extend(["", "### Co-change Evidence (Top 2)"])
    evidence_err = ""
    try:
        top2 = top_code_rows[:2]
        if not top2:
            lines.append("- (none detected)")
        for row in top2:
            lines.append(f"- file: {row.get('file')}")
            lines.append("  - supporting_commits:")
            ev = get_cochange_evidence(
                repo_path,
                cochange_targets,
                row.get("file", ""),
                n_commits=300,
                max_commits=3,
            )
            if ev:
                for c in ev:
                    lines.append(f"    - {c.get('hash')} {c.get('date')} {c.get('message')}")
            else:
                lines.append("    - (none detected)")
    except Exception as exc:
        evidence_err = str(exc)
    if evidence_err:
        lines.append(f"- cochange evidence unavailable: {evidence_err}")

    if impact_target:
        lines.extend(["", "## Impact Propagation (Static)", f"- Target: {impact_target}", "", "### Level 1: Direct Dependents"])
        if impact_prop.get("level_1"):
            for f in impact_prop.get("level_1", []):
                lines.append(f"- {f}")
        else:
            lines.append("- (none detected)")
        lines.extend(["", "### Level 2: Indirect Dependents"])
        if impact_prop.get("level_2"):
            for f in impact_prop.get("level_2", []):
                lines.append(f"- {f}")
        else:
            lines.append("- (none detected)")
        lines.append("")
        lines.append("This propagation is based on static import graph, not runtime behavior.")
        if impact_error:
            lines.append(f"- impact propagation unavailable: {impact_error}")

    lines.extend(["", "## Decision Timeline (Filtered Evidence)"])
    lines.append("Filtering rule: noise commits removed (merge/delete/docs-only)")
    if timeline_items:
        for it in timeline_items:
            sigs = ", ".join(it.get("signals", [])) if it.get("signals") else "(none)"
            lines.append(f"- {it.get('date','')} [{it.get('commit_hash','')}] {it.get('author','')}")
            lines.append(f"  score={it.get('decision_score', 0)}")
            lines.append(f"  signals: {sigs}")
            lines.append(f"  - note: {it.get('short_note','')}")
            lines.append(f"  changed_files: {it.get('changed_files_count', 0)}")
    else:
        lines.append("- (insufficient history for timeline)")

    lines.extend(["", "## Evidence Summary"])
    if used_hashes:
        for h in used_hashes:
            lines.append(f"- {h}")
    else:
        lines.append("- (none)")
    if selected_github:
        picked = ", ".join(
            f"#{x.get('number')} score={x.get('score', 0)} breakdown={x.get('score_breakdown', {})}"
            for x in selected_github
        )
        lines.append(f"- selected_github_evidence: [{picked}]")
    else:
        lines.append("- selected_github_evidence: (none)")
    if github_pr_evidence and github_pr_evidence.get("available") and github_pr_evidence.get("url"):
        lines.append(f"- pr_url: {github_pr_evidence.get('url')}")

    if github_pr_evidence:
        lines.extend(["", "## GitHub PR Evidence (Optional)"])
        if github_pr_evidence.get("available"):
            lines.extend(github_pr_evidence.get("text", "").splitlines()[1:])
        else:
            lines.append(f"- pr: #{github_pr_evidence.get('pr_number', '')}")
            lines.append("- PR evidence unavailable (rate limit or not found)")

    lines.extend(["", "## Linked GitHub Evidence (Optional)"])
    if not linked_github.get("enabled"):
        lines.append("- GitHub evidence disabled.")
    elif not linked_github.get("owner"):
        lines.append("- GitHub remote not detected; skipping GitHub evidence.")
    elif linked_github.get("items"):
        selected_map = {x.get("number"): x for x in selected_github}
        for item in linked_github.get("items", [])[:5]:
            chosen = selected_map.get(item.get("number"), item)
            lines.extend(
                [
                    f"- Issue/PR #{item.get('number')} (state={item.get('state','')}, author={item.get('author','')}, url={item.get('html_url','')})",
                    f"  - fetched_from: {item.get('_source', 'network')}",
                    f"  - title: {item.get('title','')}",
                    f"  - body_snippet: {(item.get('body_snippet','') or '(empty)')[:1200]}",
                    f"  - score_breakdown: {chosen.get('score_breakdown', {})}",
                ]
            )
        extra = linked_github.get("items", [])[5:]
        if extra:
            extra_nums = ", ".join(f"#{x.get('number')}" for x in extra)
            lines.append(f"- additional_refs: {extra_nums}")
    else:
        if linked_github.get("numbers"):
            lines.append(f"- refs_detected: {', '.join(f'#{n}' for n in linked_github.get('numbers', []))}")
        if linked_github.get("urls"):
            lines.append(f"- dry_run_urls: {', '.join(linked_github.get('urls', [])[:5])}")
        if linked_github.get("errors"):
            lines.append(f"- GitHub evidence unavailable: {'; '.join(linked_github.get('errors', [])[:3])}")
        else:
            lines.append("- GitHub evidence unavailable.")

    lines.extend([""] + render_motive_evidence_lines(motive_evidence))
    lines.extend(["", "## Answer (Evidence-Driven)", answer, "", "## Limitations"])
    lines.extend(limitations)
    lines.append("")

    markdown = "\n".join(lines)
    motive_evidence = dict(motive_evidence or {})
    motive_evidence["motive_confidence"] = _extract_motive_confidence(answer)

    return {
        "question": question,
        "introducing_commit": introducing_commit,
        "structured_changes": structured_changes,
        "motive_evidence": motive_evidence,
        "decision_timeline_items": timeline_items,
        "cochange_top": cochange_top if isinstance(cochange_top, dict) else {},
        "impact_propagation": impact_prop,
        "symbol_resolution": symbol_resolution,
        "report_markdown": markdown,
        "paths": {"repo": repo_path},
    }


def _section_lines(markdown_text, title):
    lines = (markdown_text or "").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == title:
            start = i + 1
            break
    if start is None:
        return []
    out = []
    for ln in lines[start:]:
        if ln.startswith("## "):
            break
        out.append(ln)
    return out


def _parse_trace_report_to_session(markdown_text):
    question_lines = _section_lines(markdown_text, "## Question")
    question = next((x.strip() for x in question_lines if x.strip()), "")

    intro_lines = _section_lines(markdown_text, "## Introducing Commit")
    introducing = {"hash": "", "author": "", "date": "", "message": ""}
    for ln in intro_lines:
        m = re.match(r"-\s*(hash|author|date|message):\s*(.*)$", ln.strip())
        if m:
            introducing[m.group(1)] = m.group(2).strip()

    structured_lines = _section_lines(markdown_text, "## Structured Changes")
    structured_changes = []
    for ln in structured_lines:
        if ln.strip().startswith("- ") and "->" in ln:
            structured_changes.append(ln.strip())

    motive_lines = _section_lines(markdown_text, "## Motive Evidence (Layered)")
    motive_score = 0
    score_breakdown = {}
    github_relevance = []
    for ln in motive_lines:
        if ln.strip().startswith("- motive_score:"):
            m = re.search(r"(\d+)\s*/\s*10", ln)
            if m:
                motive_score = int(m.group(1))
        elif ln.strip().startswith("- score_breakdown:"):
            raw = ln.split(":", 1)[1].strip()
            score_breakdown = raw
        elif ln.strip().startswith("- github_relevance:"):
            github_relevance.append(ln.strip())

    answer_lines = _section_lines(markdown_text, "## Answer (Evidence-Driven)")
    motive_conf = "Low"
    for ln in answer_lines:
        m = re.search(r"Motive Confidence:\s*(High|Medium|Low)", ln)
        if m:
            motive_conf = m.group(1)
            break

    timeline_lines = _section_lines(markdown_text, "## Decision Timeline (Filtered Evidence)")
    timeline_items = [ln.rstrip() for ln in timeline_lines if ln.strip()]

    cochange_lines = _section_lines(markdown_text, "## Cross-File Impact (Co-change)")
    top_code = []
    top_config = []
    part = ""
    for ln in cochange_lines:
        s = ln.strip()
        if s == "### Top Related Code Files":
            part = "code"
            continue
        if s == "### Top Related Config Files":
            part = "config"
            continue
        if s.startswith("### "):
            part = ""
            continue
        if s.startswith("- file="):
            if part == "code":
                top_code.append(s)
            elif part == "config":
                top_config.append(s)

    impact_lines = _section_lines(markdown_text, "## Impact Propagation (Static)")
    l1 = []
    l2 = []
    level = ""
    for ln in impact_lines:
        s = ln.strip()
        if s == "### Level 1: Direct Dependents":
            level = "l1"
            continue
        if s == "### Level 2: Indirect Dependents":
            level = "l2"
            continue
        if s.startswith("- ") and not s.startswith("- Target:"):
            val = s[2:].strip()
            if val and val != "(none detected)":
                if level == "l1":
                    l1.append(val)
                elif level == "l2":
                    l2.append(val)

    return {
        "question": question,
        "introducing_commit": introducing,
        "structured_changes": structured_changes[:10],
        "motive_evidence": {
            "motive_score": motive_score,
            "score_breakdown": score_breakdown,
            "github_relevance": github_relevance,
            "raw_lines": motive_lines,
            "motive_confidence": motive_conf,
        },
        "decision_timeline_items": timeline_items[:40],
        "cochange_top": {"top_code": top_code, "top_config": top_config},
        "impact_propagation": {"level_1": l1, "level_2": l2},
        "report_markdown": markdown_text,
    }


def _run_trace_for_chat(repo_path, question):
    out_path = os.path.join("/tmp", f"socratic_chat_trace_{int(time.time() * 1000)}.md")
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "run.py"),
        "trace",
        "--repo",
        repo_path,
        "--q",
        question,
        "--out",
        out_path,
    ]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc, out_path
    if not os.path.exists(out_path):
        return None, proc, out_path
    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        md = f.read()
    try:
        os.remove(out_path)
    except Exception:
        pass
    return _parse_trace_report_to_session(md), proc, out_path


def main():
    """Run CLI with index/ask subcommands."""
    if "--selftest" in sys.argv:
        from textwrap import dedent

        sample_diff = dedent(
            """
            diff --git a/src/config.py b/src/config.py
            index 1111111..2222222 100644
            --- a/src/config.py
            +++ b/src/config.py
            @@ -10,7 +10,7 @@
            -timeout = 5
            +timeout = 10
             logger.info("updated")
            -retries: 2
            +retries: 5
            """
        ).strip()
        changes = detect_structured_changes(sample_diff)
        print("Structured Changes Selftest:")
        for ch in changes:
            print(
                f"- {ch.get('key')}: {ch.get('from')} -> {ch.get('to')} "
                f"(file={ch.get('file')}, hunk={ch.get('hunk')})"
            )
        ok_timeout = any(ch.get("key") == "timeout" and ch.get("from") == "5" and ch.get("to") == "10" for ch in changes)
        ok_retries = any(ch.get("key") == "retries" and ch.get("from") == "2" and ch.get("to") == "5" for ch in changes)
        if ok_timeout and ok_retries:
            print("SELFTEST OK")
            return
        print("SELFTEST FAILED")
        raise SystemExit(1)

    if "--selftest-lsp" in sys.argv:
        rg_path = shutil.which("rg")
        pyright_ls = shutil.which("pyright-langserver")
        pyright_bin = shutil.which("pyright")
        ts_ls = shutil.which("typescript-language-server")
        gopls_bin = shutil.which("gopls")
        print("LSP Selftest: tool detection")
        print(f"- rg: {rg_path or '(not found)'}")
        print(f"- pyright-langserver: {pyright_ls or '(not found)'}")
        print(f"- pyright: {pyright_bin or '(not found)'}")
        print(f"- typescript-language-server: {ts_ls or '(not found)'}")
        print(f"- gopls: {gopls_bin or '(not found)'}")
        if rg_path is None:
            print("LSP selftest unavailable: rg is required.")
            print("Install hints:")
            print("- macOS: brew install ripgrep")
            print("- Linux: apt/yum install ripgrep")
            raise SystemExit(2)

        # Strict tool authenticity checks by version command.
        verified = []
        reasons = []
        if pyright_ls:
            ok, msg = _probe_tool_version(["pyright-langserver", "--version"])
            if ok:
                verified.append(("pyright-langserver", msg))
            elif pyright_bin:
                ok2, msg2 = _probe_tool_version(["pyright", "--version"])
                if ok2:
                    verified.append(("pyright-langserver", msg2))
                else:
                    reasons.append(f"pyright tool unusable: {msg}; pyright fallback: {msg2}")
            else:
                reasons.append(f"pyright-langserver unusable: {msg}")
        elif pyright_bin:
            ok, msg = _probe_tool_version(["pyright", "--version"])
            if ok:
                reasons.append("pyright exists but pyright-langserver missing; cannot run LSP server")
            else:
                reasons.append(f"pyright unusable: {msg}")

        if ts_ls:
            ok, msg = _probe_tool_version(["typescript-language-server", "--version"])
            if ok:
                verified.append(("typescript-language-server", msg))
            else:
                reasons.append(f"typescript-language-server unusable: {msg}")
        if gopls_bin:
            ok, msg = _probe_tool_version(["gopls", "version"])
            if ok:
                verified.append(("gopls", msg))
            else:
                reasons.append(f"gopls unusable: {msg}")

        if not verified:
            print("LSP selftest unavailable: missing required tools.")
            print("Install hints:")
            print("- macOS: brew install ripgrep && npm i -g pyright typescript-language-server typescript")
            print("- Linux: apt/yum install ripgrep && npm i -g pyright typescript-language-server typescript")
            for r in reasons:
                print(f"- reason: {r}")
            raise SystemExit(2)

        selected_tool, tool_version = verified[0]
        print(f"selected_tool={selected_tool}")
        print(f"tool_version={tool_version}")

        tmp_repo_py = "/tmp/phase242_repo_py"
        tmp_repo_ts = "/tmp/phase242_repo_ts"
        for p in [tmp_repo_py, tmp_repo_ts]:
            try:
                subprocess.run(["rm", "-rf", p], check=False, capture_output=True)
            except Exception:
                pass

        try:
            # Python repo
            os.makedirs(tmp_repo_py, exist_ok=True)
            subprocess.run(["git", "init"], cwd=tmp_repo_py, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "lsp-selftest"], cwd=tmp_repo_py, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "lsp-selftest@example.com"], cwd=tmp_repo_py, check=True, capture_output=True)
            with open(os.path.join(tmp_repo_py, "a.py"), "w", encoding="utf-8") as f:
                f.write("def compute_value():\n    return 1\n")
            with open(os.path.join(tmp_repo_py, "b.py"), "w", encoding="utf-8") as f:
                f.write("from a import compute_value\n\nprint(compute_value())\n")
            subprocess.run(["git", "add", "a.py", "b.py"], cwd=tmp_repo_py, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "add compute_value"], cwd=tmp_repo_py, check=True, capture_output=True)

            # TS repo
            os.makedirs(tmp_repo_ts, exist_ok=True)
            subprocess.run(["git", "init"], cwd=tmp_repo_ts, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "lsp-selftest"], cwd=tmp_repo_ts, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "lsp-selftest@example.com"], cwd=tmp_repo_ts, check=True, capture_output=True)
            with open(os.path.join(tmp_repo_ts, "a.ts"), "w", encoding="utf-8") as f:
                f.write("export function computeValue(): number { return 1; }\n")
            with open(os.path.join(tmp_repo_ts, "b.ts"), "w", encoding="utf-8") as f:
                f.write("import { computeValue } from './a';\nconsole.log(computeValue());\n")
            subprocess.run(["git", "add", "a.ts", "b.ts"], cwd=tmp_repo_ts, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "add computeValue"], cwd=tmp_repo_ts, check=True, capture_output=True)

            if selected_tool == "typescript-language-server":
                use_repo = tmp_repo_ts
                use_q = "In b.ts, why was computeValue added?"
            else:
                use_repo = tmp_repo_py
                use_q = "In b.py, why was compute_value added?"
            result = run_trace_structured(
                repo_path=use_repo,
                question=use_q,
                args_like={
                    "topk": 3,
                    "retrieval": "keyword",
                    "no_github": True,
                    "github_evidence": "off",
                    "github_timeout": 5,
                    "cochange_mode": "code",
                    "symbol_resolver": "lsp",
                    "verbose": True,
                },
            )
            sr = result.get("symbol_resolution", {}) or {}
            md_out = "/tmp/phase242.md"
            with open(md_out, "w", encoding="utf-8") as f:
                f.write(result.get("report_markdown", ""))
            resolved_file = ""
            resolved_line = ""
            for ln in (result.get("report_markdown", "") or "").splitlines():
                s = ln.strip()
                if s.startswith("- file:"):
                    resolved_file = s.split(":", 1)[1].strip()
                elif s.startswith("- line:"):
                    resolved_line = s.split(":", 1)[1].strip()
                if resolved_file and resolved_line:
                    break
            print(f"LSP selftest report saved: {md_out}")
            print(f"status={sr.get('status')} details={sr.get('details','')}")
            print(f"resolved_file={resolved_file}")
            print(f"resolved_line={resolved_line}")
            if sr.get("status") != "ok":
                print("LSP selftest could not verify real definition jump.")
                raise SystemExit(1)
            raise SystemExit(0)
        finally:
            try:
                subprocess.run(["rm", "-rf", tmp_repo_py], check=False, capture_output=True)
                subprocess.run(["rm", "-rf", tmp_repo_ts], check=False, capture_output=True)
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="Socratic Git CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Build/overwrite vector index")
    p_index.add_argument("--repo", required=True, help="Path to local git repository")
    p_index.add_argument("--n", type=int, default=200, help="Number of latest commits to index")
    p_index.add_argument("--window-days", type=int, default=None, help="Use git log --since '<N> days ago' instead of --n")
    p_index.add_argument("--mode", choices=["vector", "keyword"], default="vector", help="Index mode")
    p_index.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch instead of incremental")

    p_ask = subparsers.add_parser("ask", help="Ask question against indexed commits")
    p_ask.add_argument("--repo", required=True, help="Path to local git repository")
    p_ask.add_argument("--q", required=True, help="Question")
    p_ask.add_argument("--topk", type=int, default=3, help="Number of contexts to retrieve")
    p_ask.add_argument("--retrieval", choices=["vector", "keyword"], default="vector", help="Retrieval mode")

    p_trace = subparsers.add_parser("trace", help="Generate evidence-driven markdown trace report")
    p_trace.add_argument("--repo", required=True, help="Path to local git repository")
    p_trace.add_argument("--q", required=True, help="Question")
    p_trace.add_argument("--topk", type=int, default=3, help="Number of contexts to retrieve")
    p_trace.add_argument("--window-days", type=int, default=None, help="Use git log --since '<N> days ago' for history window")
    p_trace.add_argument("--retrieval", choices=["vector", "keyword"], default="vector", help="Retrieval mode")
    p_trace.add_argument("--out", default="", help="Output markdown path")
    p_trace.add_argument("--html-out", default="", help="Optional output HTML path")
    p_trace.add_argument("--no-github", action="store_true", help="Disable optional GitHub PR evidence fetch")
    p_trace.add_argument("--github-token", default="", help="GitHub token (optional; prefers env GITHUB_TOKEN)")
    p_trace.add_argument("--github-evidence", choices=["on", "off"], default=None, help="Enable/disable GitHub issue evidence")
    p_trace.add_argument("--github-timeout", type=int, default=10, help="GitHub API timeout seconds")
    p_trace.add_argument("--cochange-mode", choices=["code", "all"], default="code", help="Co-change filtering mode")
    p_trace.add_argument("--symbol-resolver", choices=["heuristic", "lsp"], default="heuristic", help="Symbol resolver strategy")
    p_trace.add_argument("--verbose", action="store_true", help="Verbose logs (cache stats, retrieval warnings)")

    p_regress = subparsers.add_parser("regress", help="Find introducing commit for regression/bug signal")
    p_regress.add_argument("--repo", required=True, help="Path to local git repository")
    p_regress.add_argument("--file", required=True, help="Target file path in repo")
    p_regress.add_argument("--pattern", default="", help="Text or regex pattern to match")
    p_regress.add_argument("--symbol", default="", help="Symbol/function/class name")
    p_regress.add_argument("--out", default="", help="Output markdown path")
    p_regress.add_argument("--html-out", default="", help="Optional output HTML path")
    p_regress.add_argument("--max", type=int, default=2000, help="Max commits to scan for the file")
    p_regress.add_argument("--window-days", type=int, default=None, help="Use git log --since '<N> days ago' for history window")
    p_regress.add_argument("--no-github", action="store_true", help="Disable optional GitHub PR evidence fetch")
    p_regress.add_argument("--github-token", default="", help="GitHub token (optional; prefers env GITHUB_TOKEN)")
    p_regress.add_argument("--github-evidence", choices=["on", "off"], default=None, help="Enable/disable GitHub issue evidence")
    p_regress.add_argument("--github-timeout", type=int, default=10, help="GitHub API timeout seconds")
    p_regress.add_argument("--cochange-mode", choices=["code", "all"], default="code", help="Co-change filtering mode")
    p_regress.add_argument("--symbol-resolver", choices=["heuristic", "lsp"], default="heuristic", help="Symbol resolver strategy")
    p_regress.add_argument("--verbose", action="store_true", help="Verbose logs (cache stats, retrieval warnings)")

    p_bisect = subparsers.add_parser("bisect", help="Command-driven first bad commit search")
    p_bisect.add_argument("--repo", required=True, help="Path to local git repository")
    p_bisect.add_argument("--good", required=True, help="Known good commit hash")
    p_bisect.add_argument("--bad", required=True, help="Known bad commit hash")
    p_bisect.add_argument("--cmd", required=True, help="Shell command to run at each commit")
    p_bisect.add_argument("--out", default="", help="Output markdown path")
    p_bisect.add_argument("--max-steps", type=int, default=30, help="Maximum bisect steps")
    p_bisect.add_argument("--no-github", action="store_true", help="Disable optional GitHub PR evidence fetch")
    p_bisect.add_argument(
        "--bisect-mode",
        choices=["clone", "worktree"],
        default="clone",
        help="Bisect mode: clone (default) or worktree",
    )
    p_bisect.add_argument("--verbose", action="store_true", help="Verbose logs (timing)")

    p_chat = subparsers.add_parser("chat", help="Interactive Socratic follow-up loop")
    p_chat.add_argument("--repo", required=True, help="Path to local git repository")

    args = parser.parse_args()
    repo_path = args.repo

    try:
        if not os.path.isdir(repo_path):
            print(f"Error: repo path does not exist: {repo_path}")
            return
        if Repo is None:
            print("Error: missing dependency 'gitpython'. Please install requirements.")
            return
        try:
            Repo(repo_path)
        except InvalidGitRepositoryError:
            print(f"Error: not a git repository: {repo_path}")
            return

        table_name = table_name_for_repo(repo_path)
        configure_table(table_name)

        if args.command == "index":
            print(f"[index] repo={repo_path}")
            t0 = time.perf_counter()
            repo_key = _repo_cache_key(repo_path)
            state = load_index_state(repo_key)
            indexed_commits = set(state.get("indexed_commits", set()))

            commits = _extract_commits_optimized(repo_path, n=args.n, window_days=getattr(args, "window_days", None))
            if not commits:
                commits = extract_commits(repo_path, n=args.n)
            print(f"[index] extracted commits: {len(commits)}")
            if not commits:
                print("Error: failed to read commits from repo.")
                return

            if args.rebuild:
                indexed_commits = set()
                state = {"indexed_commits": set(), "last_index_time": "", "retrieval_mode": args.mode}
                print("[index] rebuild enabled")

            new_commits = []
            skip_count = 0
            for c in commits:
                h = c.get("hash", "")
                if h and h in indexed_commits:
                    print(f"[skip] {h[:8]} already indexed")
                    skip_count += 1
                    continue
                new_commits.append(c)

            processed = 0
            used_mode = args.mode
            if args.mode == "keyword":
                if args.rebuild:
                    kpath, processed = write_keyword_index(repo_path, commits)
                    indexed_commits = {c.get("hash", "") for c in commits if c.get("hash")}
                else:
                    kpath, processed = append_keyword_index(repo_path, new_commits)
                    indexed_commits.update(c.get("hash", "") for c in new_commits if c.get("hash"))
                print(f"[index] keyword indexed commits: {processed}")
                print(f"[index] keyword index path: {kpath}")
            else:
                try:
                    if args.rebuild:
                        records = index_commits(commits)
                        processed = len(records)
                        indexed_commits = {c.get("hash", "") for c in commits if c.get("hash")}
                    else:
                        records = append_commits(new_commits)
                        processed = len(records)
                        indexed_commits.update(c.get("hash", "") for c in new_commits if c.get("hash"))
                    print(f"[index] indexed commits: {processed}")
                    meta = _load_embed_cache_meta(repo_key)
                    prev = set(meta.get("indexed_commits", []) or [])
                    prev.update(h for h in indexed_commits if h)
                    _save_embed_cache_meta(repo_key, {"indexed_commits": sorted(prev)})
                except Exception as exc:
                    used_mode = "mixed"
                    print(f"Warning: Embedding model unavailable, falling back to keyword index. ({exc})")
                    if args.rebuild:
                        kpath, processed = write_keyword_index(repo_path, commits)
                        indexed_commits = {c.get("hash", "") for c in commits if c.get("hash")}
                    else:
                        kpath, processed = append_keyword_index(repo_path, new_commits)
                        indexed_commits.update(c.get("hash", "") for c in new_commits if c.get("hash"))
                    print(f"[index] keyword indexed commits: {processed}")
                    print(f"[index] keyword index path: {kpath}")

            state["indexed_commits"] = {h for h in indexed_commits if h}
            state["last_index_time"] = datetime.now().isoformat(timespec="seconds")
            state["retrieval_mode"] = used_mode
            save_index_state(repo_key, state)
            _save_embed_cache_meta(
                repo_key,
                {
                    "indexed_commits": sorted(list(state["indexed_commits"])),
                    "status": used_mode,
                    "updated_at": state["last_index_time"],
                },
            )

            t1 = time.perf_counter()
            print(f"[index] skipped commits: {skip_count}")
            print(f"[index] indexed commits: {processed}")
            print(f"[index] done in {t1 - t0:.2f}s")
            return

        if args.command == "chat":
            print("Socratic Chat Loop")
            print("Commands: /trace <question>, /structured, /motive, /timeline, /export <path.md>, /exit")
            session = None

            while True:
                try:
                    raw = input("socratic> ").strip()
                except EOFError:
                    print("")
                    break
                except KeyboardInterrupt:
                    print("")
                    break
                if not raw:
                    continue
                if raw == "/exit":
                    break

                if raw.startswith("/trace "):
                    question = raw[len("/trace ") :].strip()
                    if not question:
                        print("Please provide a question after /trace")
                        continue
                    try:
                        parsed = run_trace_structured(
                            repo_path,
                            question,
                            {
                                "topk": 3,
                                "retrieval": "vector",
                                "no_github": False,
                                "github_evidence": None,
                                "github_timeout": 10,
                                "cochange_mode": "code",
                                "symbol_resolver": "heuristic",
                                "verbose": False,
                            },
                        )
                    except Exception as exc:
                        print("Trace failed.")
                        print(f"error: {exc}")
                        continue
                    session = parsed
                    intro_hash = session.get("introducing_commit", {}).get("hash", "") or "(not found)"
                    ms = session.get("motive_evidence", {}).get("motive_score", 0)
                    mc = session.get("motive_evidence", {}).get("motive_confidence", "Low")
                    print(f"Trace OK. introducing_commit={intro_hash} motive_score={ms}/10 motive_confidence={mc}")
                    continue

                if raw == "/structured":
                    if not session:
                        print("No session. Run /trace <question> first.")
                        continue
                    items = session.get("structured_changes", [])
                    if not items:
                        print("(none)")
                    else:
                        for ch in items[:10]:
                            print(
                                f"- {ch.get('key', '')}: {ch.get('from', '')} -> {ch.get('to', '')} "
                                f"(file={ch.get('file', '')}, hunk={ch.get('hunk', '')})"
                            )
                    continue

                if raw == "/motive":
                    if not session:
                        print("No session. Run /trace <question> first.")
                        continue
                    for ln in render_motive_evidence_lines(session.get("motive_evidence", {})):
                        print(ln)
                    continue

                if raw == "/timeline":
                    if not session:
                        print("No session. Run /trace <question> first.")
                        continue
                    items = session.get("decision_timeline_items", [])
                    if not items:
                        print("(none)")
                    else:
                        for it in items[:10]:
                            sigs = ", ".join(it.get("signals", [])) if it.get("signals") else "(none)"
                            print(f"- {it.get('date','')} [{it.get('commit_hash','')}] {it.get('author','')}")
                            print(f"  score={it.get('decision_score', 0)}")
                            print(f"  signals: {sigs}")
                            print(f"  - note: {it.get('short_note','')}")
                            print(f"  changed_files: {it.get('changed_files_count', 0)}")
                    continue

                if raw.startswith("/export "):
                    if not session:
                        print("No session. Run /trace <question> first.")
                        continue
                    out = raw[len("/export ") :].strip()
                    if not out:
                        print("Please provide output markdown path.")
                        continue
                    out_dir = os.path.dirname(out) or "."
                    os.makedirs(out_dir, exist_ok=True)
                    with open(out, "w", encoding="utf-8") as f:
                        f.write(session.get("report_markdown", ""))
                    print(f"exported: {out}")
                    continue

                # Bare text fallback: treat as /trace question
                question = raw
                try:
                    parsed = run_trace_structured(
                        repo_path,
                        question,
                        {
                            "topk": 3,
                            "retrieval": "vector",
                            "no_github": False,
                            "github_evidence": None,
                            "github_timeout": 10,
                            "cochange_mode": "code",
                            "symbol_resolver": "heuristic",
                            "verbose": False,
                        },
                    )
                except Exception as exc:
                    print("Trace failed.")
                    print(f"error: {exc}")
                    continue
                session = parsed
                intro_hash = session.get("introducing_commit", {}).get("hash", "") or "(not found)"
                ms = session.get("motive_evidence", {}).get("motive_score", 0)
                mc = session.get("motive_evidence", {}).get("motive_confidence", "Low")
                print(f"Trace OK. introducing_commit={intro_hash} motive_score={ms}/10 motive_confidence={mc}")
            return

        if args.command == "trace":
            question = args.q
            try:
                result = run_trace_structured(
                    repo_path=repo_path,
                    question=question,
                    args_like={
                        "topk": args.topk,
                        "retrieval": getattr(args, "retrieval", "vector"),
                        "no_github": getattr(args, "no_github", False),
                        "github_token": getattr(args, "github_token", ""),
                        "github_evidence": getattr(args, "github_evidence", None),
                        "github_timeout": getattr(args, "github_timeout", 10),
                        "cochange_mode": getattr(args, "cochange_mode", "code"),
                        "symbol_resolver": getattr(args, "symbol_resolver", "heuristic"),
                        "verbose": getattr(args, "verbose", False),
                        "window_days": getattr(args, "window_days", None),
                    },
                )
            except Exception as exc:
                print(f"Error: failed to run trace pipeline. Details: {exc}")
                return

            if args.out:
                out_path = args.out
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("outputs", f"socratic_report_{ts}.md")
            out_dir = os.path.dirname(out_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result.get("report_markdown", ""))
            print(f"report saved to {out_path}")
            if getattr(args, "html_out", ""):
                html_out = args.html_out
                html_dir = os.path.dirname(html_out) or "."
                os.makedirs(html_dir, exist_ok=True)
                html_text = render_html_report(
                    result.get("report_markdown", ""),
                    {
                        "repo_path": repo_path,
                        "question": question,
                        "motive_score": (result.get("motive_evidence", {}) or {}).get("motive_score", 0),
                        "motive_confidence": (result.get("motive_evidence", {}) or {}).get("motive_confidence", "Low"),
                    },
                )
                with open(html_out, "w", encoding="utf-8") as f:
                    f.write(html_text)
                print(f"html report saved to {html_out}")
            return

            target_info = get_symbol_or_lines_from_query(question)
            detected_file = target_info.get("file")
            detected_line = target_info.get("line")
            detected_symbol = target_info.get("symbol")
            resolved_line = None
            blame_info = ""
            introducing_commit = {}
            intro_files = []
            intro_msg_preview = ""
            intro_diff_preview = ""
            local_history_window = []
            github_pr_evidence = None

            if detected_file and not detected_line and detected_symbol:
                resolved_line = find_symbol_definition(repo_path, detected_file, detected_symbol)
                if resolved_line:
                    detected_line = resolved_line
                else:
                    fallback_file, fallback_line = find_symbol_in_repo(
                        repo_path, detected_symbol, preferred_file=detected_file
                    )
                    if fallback_file and fallback_line:
                        detected_file = fallback_file
                        resolved_line = fallback_line
                        detected_line = fallback_line

            if detected_file and detected_line:
                blame_dict = get_blame_for_line(repo_path, detected_file, detected_line)
                if blame_dict:
                    blame_info = (
                        f"file={blame_dict.get('file', '')}; "
                        f"line={blame_dict.get('line', '')}; "
                        f"commit={blame_dict.get('hash', '')}; "
                        f"author={blame_dict.get('author', '')}; "
                        f"date={blame_dict.get('date', '')}; "
                        f"line_content={blame_dict.get('line_content', '')}"
                    )
                    introducing_commit = _get_introducing_commit_cached(
                        repo_path, blame_dict.get("hash", ""), cache_ctx=None, need_patch=True
                    )
            elif detected_file:
                blame_info = get_file_blame(repo_path, detected_file)

            if introducing_commit:
                intro_msg_lines = introducing_commit.get("message", "").splitlines()[:10]
                intro_msg_preview = "\n".join(intro_msg_lines)
                intro_diff_clean = clean_diff(introducing_commit.get("diff", ""))
                intro_diff_lines = intro_diff_clean.splitlines()[:120]
                intro_diff_preview = "\n".join(intro_diff_lines)
                intro_files = introducing_commit.get("changed_files", []) or extract_changed_files(
                    introducing_commit.get("diff", "")
                )
                github_pr_evidence = build_github_pr_evidence(
                    repo_path,
                    introducing_commit.get("hash", ""),
                    disabled=getattr(args, "no_github", False),
                )

            if detected_file:
                local_history_window = get_local_history_window(repo_path, detected_file, n=5)

            # Optional GitHub linkage evidence from commit refs (#123, Fixes #123, etc.)
            commit_msgs_for_refs = []
            if introducing_commit:
                commit_msgs_for_refs.append(introducing_commit.get("message", ""))
            for ctx in contexts:
                commit_msgs_for_refs.append(ctx.get("commit", {}).get("message", ""))
            for row in local_history_window:
                commit_msgs_for_refs.append(row.get("message", ""))
            linked_github = collect_linked_github_evidence(repo_path, commit_msgs_for_refs, args)

            used_hashes = []
            seen_hash = set()

            def add_hash(h):
                if h and h not in seen_hash:
                    seen_hash.add(h)
                    used_hashes.append(h)

            add_hash(introducing_commit.get("hash", ""))
            for row in local_history_window:
                add_hash(row.get("hash", ""))
            for ctx in contexts:
                add_hash(ctx.get("commit", {}).get("hash", ""))

            structured_changes = detect_structured_changes(introducing_commit.get("diff", "")) if introducing_commit else []
            selected_github = select_github_evidence(question, structured_changes, linked_github.get("items", []), topn=3)

            llm_contexts = contexts
            if introducing_commit:
                introducing_details_text = (
                    "=== Introducing Commit Details ===\n"
                    f"hash: {introducing_commit.get('hash', '')}\n"
                    f"author/date: {introducing_commit.get('author', '')} / {introducing_commit.get('date', '')}\n"
                    f"message:\n{intro_msg_preview}\n"
                    f"changed_files: {', '.join(intro_files) if intro_files else '(unavailable)'}\n"
                    f"diff_snippet: {intro_diff_preview if intro_diff_preview else '(unavailable)'}"
                )
                llm_contexts = [
                    {
                        "commit": introducing_commit,
                        "text": introducing_details_text,
                    }
                ] + llm_contexts
            if github_pr_evidence:
                llm_contexts = [
                    {
                        "commit": {
                            "hash": f"pr-{github_pr_evidence.get('pr_number', '')}",
                            "author": "github",
                            "date": "",
                            "message": "GitHub PR Evidence",
                            "diff": "",
                        },
                        "text": github_pr_evidence.get("text", ""),
                    }
                ] + llm_contexts
            if selected_github:
                for item in selected_github:
                    llm_contexts = [
                        {
                            "commit": {
                                "hash": f"github-issue-{item.get('number')}",
                                "author": item.get("author", "github"),
                                "date": item.get("created_at", ""),
                                "message": f"GitHub Issue/PR #{item.get('number')}",
                                "diff": "",
                            },
                            "text": (
                                f"[GITHUB_ISSUE #{item.get('number')}] {item.get('title','')}\n"
                                f"{item.get('body_snippet','')}"
                            ),
                        }
                    ] + llm_contexts
            if local_history_window:
                history_lines = []
                for row in local_history_window:
                    msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
                    history_lines.append(f"{row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())
                history_text = "=== Local History Window (file) ===\n" + "\n".join(history_lines)
                llm_contexts = llm_contexts + [
                    {
                        "commit": {
                            "hash": "local-history",
                            "author": "git log",
                            "date": "",
                            "message": "Recent file-local history window",
                            "diff": "",
                        },
                        "text": history_text,
                    }
                ]

            intent_info = detect_query_intent(question)
            intro_text_for_judgement = (
                ((introducing_commit.get("message", "") or "") + "\n" + (introducing_commit.get("diff", "") or ""))
                .lower()
            )
            if github_pr_evidence and github_pr_evidence.get("available"):
                intro_text_for_judgement += "\n" + (github_pr_evidence.get("title", "") or "").lower()
                intro_text_for_judgement += "\n" + (github_pr_evidence.get("body", "") or "").lower()
            for item in selected_github:
                intro_text_for_judgement += "\n" + (item.get("title", "") or "").lower()
                intro_text_for_judgement += "\n" + (item.get("body_snippet", "") or "").lower()
            motive_keywords = ("because", "so that", "in order to", "to improve", "motivation", "reason", "why", "fix timeout", "latency")
            has_explicit_motive = any(k in intro_text_for_judgement for k in motive_keywords)

            why_chain = build_why_chain(resolved_line or detected_line, introducing_commit.get("hash", ""), selected_github)

            findings_text = (
                f"Question target: file={detected_file or '(not detected)'}, line={detected_line or '(not detected)'}, "
                f"symbol={detected_symbol or '(not detected)'}; "
            )
            if introducing_commit:
                findings_text += (
                    f"introducing commit={introducing_commit.get('hash', '')} "
                    f"({introducing_commit.get('date', '')}, {introducing_commit.get('author', '')}), "
                    f"message='{(introducing_commit.get('message', '') or '').splitlines()[0]}'."
                )
            else:
                findings_text += "introducing commit not found; using retrieved commit evidence."

            motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
            if _llm_disabled():
                motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
            else:
                try:
                    motive_prompt = (
                        "You only write the Motive paragraph (max 3 sentences).\n"
                        "Use provided commit/PR evidence only.\n"
                        "If motive is not explicit, output exactly: Motive is not explicitly stated in commit/issue evidence."
                    )
                    motive_raw = _generate_answer(motive_prompt, llm_contexts)
                    motive_text_out = _truncate_sentences(sanitize_llm_output(motive_raw), max_sentences=3)
                    if not motive_text_out:
                        motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
                except Exception as exc:
                    msg = str(exc)
                    if "No such file or directory" in msg or "not found" in msg.lower():
                        print(f"Error: model may be missing/not downloaded. Details: {exc}")
                    else:
                        print(f"Error: failed to run local model. Details: {exc}")
                    motive_text_out = "Motive is not explicitly stated in commit/issue evidence."

            answer = build_answer_template(
                findings=findings_text,
                motive=motive_text_out,
                has_explicit_motive=has_explicit_motive,
                structured_changes=structured_changes,
                evidence_text=intro_text_for_judgement,
                missing_items=[],
                intent_info=intent_info,
                changed_files=intro_files,
                selected_github_evidence=selected_github,
                why_chain=why_chain,
            )

            if args.out:
                out_path = args.out
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("outputs", f"socratic_report_{ts}.md")
            out_dir = os.path.dirname(out_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            limitations = []
            if not detected_file:
                limitations.append("- Query did not include a concrete file path; target detection may be weak.")
            if detected_symbol and not detected_line and not resolved_line:
                limitations.append("- Symbol was detected but could not be resolved to a line number.")
            if detected_file and not introducing_commit:
                limitations.append("- Introducing commit not found; report falls back to file-level evidence.")
            if introducing_commit and not introducing_commit.get("diff", "").strip():
                limitations.append("- Introducing commit diff is empty or unavailable.")
            generic_count = 0
            for ctx in contexts:
                first_line = (ctx.get("commit", {}).get("message", "") or "").splitlines()[0].lower()
                if first_line.startswith("update") or first_line.startswith("updating") or first_line.startswith("merge"):
                    generic_count += 1
            if generic_count >= max(1, len(contexts) - 1):
                limitations.append("- Retrieved commit messages are generic, so why-level intent may be under-specified.")
            if "我不知道" in answer:
                limitations.append("- Evidence was insufficient for a confident why-explanation.")
            if not limitations:
                limitations.append("- No major limitations detected for this trace run.")

            cochange_targets = []
            if detected_file:
                cochange_targets = [detected_file]
            elif intro_files:
                cochange_targets = intro_files[:3]
            else:
                guessed = []
                for ctx in contexts:
                    files = extract_changed_files(ctx.get("commit", {}).get("diff", ""))[:3]
                    if not files:
                        files = _changed_files_for_hash(repo_path, ctx.get("commit", {}).get("hash", ""))[:3]
                    for f in files:
                        if f not in guessed:
                            guessed.append(f)
                    if len(guessed) >= 3:
                        break
                cochange_targets = guessed[:3]
            cochange_top = []
            cochange_error = ""
            cochange_window = 0
            try:
                if cochange_targets:
                    cochange_top = compute_cochange(
                        repo_path, cochange_targets, n_commits=300, mode=getattr(args, "cochange_mode", "code")
                    )
                    cochange_window = int(getattr(compute_cochange, "last_window_commits", 0) or 0)
                else:
                    cochange_error = "no target files detected"
            except Exception as exc:
                cochange_error = str(exc)

            impact_target = ""
            impact_prop = {"level_1": [], "level_2": []}
            impact_error = ""
            if detected_file and classify_file(detected_file) == "CODE" and detected_file.endswith(".py"):
                impact_target = detected_file
                try:
                    impact_prop = compute_impact_propagation(repo_path, impact_target, max_depth=2)
                except Exception as exc:
                    impact_error = str(exc)

            lines = [
                "# Socratic Git Trace Report",
                "",
                "## Question",
                question,
                "",
                "## Detected Target",
                f"- file: {detected_file or '(not detected)'}",
                f"- symbol: {detected_symbol or '(not detected)'}",
                f"- line: {detected_line if detected_line else '(not detected)'}",
                f"- resolved_line: {resolved_line if resolved_line else '(none)'}",
                "",
                "## Introducing Commit",
            ]

            if introducing_commit:
                intro_msg = introducing_commit.get("message", "").splitlines()[0] if introducing_commit.get("message") else ""
                lines.extend(
                    [
                        f"- hash: {introducing_commit.get('hash', '')}",
                        f"- author: {introducing_commit.get('author', '')}",
                        f"- date: {introducing_commit.get('date', '')}",
                        f"- message: {intro_msg}",
                    ]
                )
            else:
                lines.append("- not found")

            lines.extend(
                [
                    "",
                    "## Introducing Commit Details",
                ]
            )

            if introducing_commit:
                lines.extend(
                    [
                        f"- hash: {introducing_commit.get('hash', '')}",
                        f"- author/date: {introducing_commit.get('author', '')} / {introducing_commit.get('date', '')}",
                        "- message (first 10 lines):",
                        "```text",
                        intro_msg_preview or "(empty)",
                        "```",
                        f"- changed_files: {', '.join(intro_files) if intro_files else '(unavailable)'}",
                        (
                            "- diff_snippet: (unavailable)"
                            if not intro_diff_preview
                            else "- diff_snippet (clean_diff + first 120 lines):"
                        ),
                    ]
                )
                if intro_diff_preview:
                    lines.extend(
                        [
                            "```diff",
                            intro_diff_preview,
                            "```",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "```diff",
                            "(unavailable)",
                            "```",
                        ]
                    )
            else:
                lines.append("- not available")

            lines.extend(
                [
                    "",
                    "## Local History Window",
                ]
            )
            if local_history_window:
                for row in local_history_window:
                    msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
                    lines.append(f"- {row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())
            else:
                lines.append("- not available")

            lines.extend(
                [
                    "",
                    "## Retrieved Commits (TopK)",
                ]
            )
            for i, ctx in enumerate(contexts[:topk], start=1):
                c = ctx.get("commit", {})
                first = (c.get("message", "") or "").splitlines()[0]
                files = extract_changed_files(c.get("diff", ""))
                lines.extend(
                    [
                        f"- [{i}] {c.get('hash', '')} {c.get('date', '')} {c.get('author', '')}",
                        f"  - message: {first}",
                        f"  - files: {', '.join(files) if files else '(none)'}",
                    ]
                )

            lines.extend(
                [
                    "",
                    "## Structured Changes",
                ]
            )
            if structured_changes:
                for ch in structured_changes[:10]:
                    lb = (ch.get("line_before", "") or "")[:120]
                    la = (ch.get("line_after", "") or "")[:120]
                    lines.append(
                        f"- {ch.get('key', '')}: {ch.get('from', '')} -> {ch.get('to', '')} "
                        f"(file={ch.get('file', '')}, hunk={ch.get('hunk', '')})"
                    )
                    lines.append(f"  - line_before: {lb}")
                    lines.append(f"  - line_after: {la}")
            else:
                lines.append("- (none detected)")

            lines.extend(
                [
                    "",
                    "## Cross-File Impact (Co-change)",
                    f"- target_files: {', '.join(cochange_targets) if cochange_targets else '(none)'}",
                    f"- window_commits: {cochange_window}",
                    "",
                    "### Top Related Code Files",
                ]
            )
            if cochange_top and cochange_top.get("top_code"):
                dep_target = (cochange_targets or [""])[0]
                for row in cochange_top.get("top_code", []):
                    lines.append(
                        f"- file={row.get('file')}, co_count={row.get('co_count')}, "
                        f"lift={row.get('lift')}, p_other={row.get('p_other')}, "
                        f"p_other_given_target={row.get('p_other_given_target')}"
                    )
                    dep = detect_structural_dependency(repo_path, dep_target, row.get("file", ""))
                    try:
                        lift_value = float(row.get("lift", 0) or 0)
                    except Exception:
                        lift_value = 0.0
                    if lift_value > 2 and dep.get("ast_call_relation") is True:
                        structural_confidence = "High"
                    elif dep.get("import_relation") is True:
                        structural_confidence = "Medium"
                    else:
                        structural_confidence = "Low"
                    lines.append("  - Structural Dependency Signals")
                    lines.append(f"  - structural_dependency: {dep}")
                    lines.append(f"  - structural_confidence: {structural_confidence}")
                    if dep.get("ast_skipped_large"):
                        lines.append("  - AST skipped: file too large")
            else:
                lines.append("- (none detected)")

            lines.extend(["", "### Top Related Config Files"])
            if cochange_top and cochange_top.get("top_config"):
                for row in cochange_top.get("top_config", []):
                    lines.append(
                        f"- file={row.get('file')}, co_count={row.get('co_count')}, "
                        f"lift={row.get('lift')}, p_other={row.get('p_other')}, "
                        f"p_other_given_target={row.get('p_other_given_target')}"
                    )
            else:
                lines.append("- (none detected)")

            lines.extend(["", "### Ignored (Artifacts / Data / Build)"])
            if cochange_top and cochange_top.get("ignored"):
                for row in cochange_top.get("ignored", []):
                    lines.append(
                        f"- file={row.get('file')}, class={row.get('class')}, co_count={row.get('co_count')}, "
                        f"lift={row.get('lift')}"
                    )
            else:
                lines.append("- (none detected)")

            if cochange_error:
                lines.append(f"- cochange unavailable: {cochange_error}")
            lines.append(
                "- interpretation: This suggests structural coupling (statistical), not proven causality."
            )

            lines.extend(["", "### Coupling Summary"])
            summary_lines = []
            top_code_rows = (cochange_top or {}).get("top_code", [])
            top_config_rows = (cochange_top or {}).get("top_config", [])
            if top_code_rows:
                for row in top_code_rows[:3]:
                    summary_lines.append(
                        f"- {', '.join(cochange_targets[:1] or ['target files'])} and {row.get('file')} frequently co-change, "
                        "suggesting they may be part of the same implementation flow."
                    )
            if top_config_rows:
                for row in top_config_rows[:1]:
                    summary_lines.append(
                        f"- Config file {row.get('file')} likely changes with target logic, suggesting runtime/threshold alignment may be needed."
                    )
            if not summary_lines:
                summary_lines.append("- (none detected)")
            lines.extend(summary_lines[:4])

            lines.extend(["", "### Co-change Evidence (Top 2)"])
            evidence_err = ""
            try:
                top2 = top_code_rows[:2]
                if not top2:
                    lines.append("- (none detected)")
                for row in top2:
                    lines.append(f"- file: {row.get('file')}")
                    lines.append("  - supporting_commits:")
                    ev = get_cochange_evidence(
                        repo_path,
                        cochange_targets,
                        row.get("file", ""),
                        n_commits=300,
                        max_commits=3,
                    )
                    if ev:
                        for c in ev:
                            lines.append(f"    - {c.get('hash')} {c.get('date')} {c.get('message')}")
                    else:
                        lines.append("    - (none detected)")
            except Exception as exc:
                evidence_err = str(exc)
            if evidence_err:
                lines.append(f"- cochange evidence unavailable: {evidence_err}")

            if impact_target:
                lines.extend(
                    [
                        "",
                        "## Impact Propagation (Static)",
                        f"- Target: {impact_target}",
                        "",
                        "### Level 1: Direct Dependents",
                    ]
                )
                if impact_prop.get("level_1"):
                    for f in impact_prop.get("level_1", []):
                        lines.append(f"- {f}")
                else:
                    lines.append("- (none detected)")
                lines.extend(["", "### Level 2: Indirect Dependents"])
                if impact_prop.get("level_2"):
                    for f in impact_prop.get("level_2", []):
                        lines.append(f"- {f}")
                else:
                    lines.append("- (none detected)")
                lines.append("")
                lines.append("This propagation is based on static import graph, not runtime behavior.")
                if impact_error:
                    lines.append(f"- impact propagation unavailable: {impact_error}")

            timeline_items = build_decision_timeline(
                repo_path=repo_path,
                target_file=detected_file or impact_target or "",
                introducing_commit=introducing_commit if introducing_commit else {},
                retrieved_commits=contexts,
                local_history_commits=local_history_window,
                max_items=8,
                symbol_hint=detected_symbol or "",
            )
            motive_evidence = build_motive_evidence(
                repo_path=repo_path,
                query=question,
                evidence={
                    "selected_github_evidence": selected_github,
                    "introducing_commit": introducing_commit,
                    "retrieved_commits": contexts,
                    "structured_changes": structured_changes,
                    "changed_files": intro_files,
                    "impact_propagation": impact_prop,
                    "cochange_top": cochange_top if isinstance(cochange_top, dict) else {},
                    "timeline_items": timeline_items,
                },
            )
            answer = build_answer_template(
                findings=findings_text,
                motive=motive_text_out,
                has_explicit_motive=has_explicit_motive,
                structured_changes=structured_changes,
                evidence_text=intro_text_for_judgement,
                missing_items=[],
                intent_info=intent_info,
                changed_files=intro_files,
                selected_github_evidence=selected_github,
                why_chain=why_chain,
                motive_evidence=motive_evidence,
            )
            lines.extend(["", "## Decision Timeline (Filtered Evidence)"])
            lines.append("Filtering rule: noise commits removed (merge/delete/docs-only)")
            if timeline_items:
                for it in timeline_items:
                    sigs = ", ".join(it.get("signals", [])) if it.get("signals") else "(none)"
                    lines.append(
                        f"- {it.get('date','')} [{it.get('commit_hash','')}] {it.get('author','')}"
                    )
                    lines.append(f"  score={it.get('decision_score', 0)}")
                    lines.append(f"  signals: {sigs}")
                    lines.append(f"  - note: {it.get('short_note','')}")
                    lines.append(f"  changed_files: {it.get('changed_files_count', 0)}")
            else:
                lines.append("- (insufficient history for timeline)")

            lines.extend(
                [
                    "",
                    "## Evidence Summary",
                ]
            )
            if used_hashes:
                for h in used_hashes:
                    lines.append(f"- {h}")
            else:
                lines.append("- (none)")
            if selected_github:
                picked = ", ".join(
                    f"#{x.get('number')} score={x.get('score', 0)} breakdown={x.get('score_breakdown', {})}"
                    for x in selected_github
                )
                lines.append(f"- selected_github_evidence: [{picked}]")
            else:
                lines.append("- selected_github_evidence: (none)")
            if github_pr_evidence and github_pr_evidence.get("available") and github_pr_evidence.get("url"):
                lines.append(f"- pr_url: {github_pr_evidence.get('url')}")

            if github_pr_evidence:
                lines.extend(
                    [
                        "",
                        "## GitHub PR Evidence (Optional)",
                    ]
                )
                if github_pr_evidence.get("available"):
                    lines.extend(github_pr_evidence.get("text", "").splitlines()[1:])
                else:
                    lines.append(f"- pr: #{github_pr_evidence.get('pr_number', '')}")
                    lines.append("- PR evidence unavailable (rate limit or not found)")

            lines.extend(
                [
                    "",
                    "## Linked GitHub Evidence (Optional)",
                ]
            )
            if not linked_github.get("enabled"):
                lines.append("- GitHub evidence disabled.")
            elif not linked_github.get("owner"):
                lines.append("- GitHub remote not detected; skipping GitHub evidence.")
            elif linked_github.get("items"):
                selected_map = {x.get("number"): x for x in selected_github}
                for item in linked_github.get("items", [])[:5]:
                    chosen = selected_map.get(item.get("number"), item)
                    lines.extend(
                        [
                            f"- Issue/PR #{item.get('number')} (state={item.get('state','')}, author={item.get('author','')}, url={item.get('html_url','')})",
                            f"  - fetched_from: {item.get('_source', 'network')}",
                            f"  - title: {item.get('title','')}",
                            f"  - body_snippet: {(item.get('body_snippet','') or '(empty)')[:1200]}",
                            f"  - score_breakdown: {chosen.get('score_breakdown', {})}",
                        ]
                    )
                extra = linked_github.get("items", [])[5:]
                if extra:
                    extra_nums = ", ".join(f"#{x.get('number')}" for x in extra)
                    lines.append(f"- additional_refs: {extra_nums}")
            else:
                if linked_github.get("numbers"):
                    lines.append(f"- refs_detected: {', '.join(f'#{n}' for n in linked_github.get('numbers', []))}")
                if linked_github.get("urls"):
                    lines.append(f"- dry_run_urls: {', '.join(linked_github.get('urls', [])[:5])}")
                if linked_github.get("errors"):
                    lines.append(f"- GitHub evidence unavailable: {'; '.join(linked_github.get('errors', [])[:3])}")
                else:
                    lines.append("- GitHub evidence unavailable.")

            lines.extend([""] + render_motive_evidence_lines(motive_evidence))
            lines.extend(
                [
                    "",
                    "## Answer (Evidence-Driven)",
                    answer,
                    "",
                    "## Limitations",
                ]
            )
            lines.extend(limitations)
            lines.append("")

            markdown_report = "\n".join(lines)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(markdown_report)

            if getattr(args, "html_out", ""):
                html_out = args.html_out
                html_dir = os.path.dirname(html_out) or "."
                os.makedirs(html_dir, exist_ok=True)
                html_text = render_html_report(
                    markdown_report,
                    {
                        "repo_path": repo_path,
                        "question": f"When was {target_desc} introduced in {file_path}?",
                        "motive_score": (motive_evidence or {}).get("motive_score", 0),
                        "motive_confidence": _extract_motive_confidence(answer_block),
                    },
                )
                with open(html_out, "w", encoding="utf-8") as f:
                    f.write(html_text)
                print(f"html report saved to {html_out}")

            print(f"report saved to {out_path}")
            return

        if args.command == "regress":
            file_path = args.file
            pattern = (args.pattern or "").strip()
            symbol = (args.symbol or "").strip()
            max_commits = max(1, args.max)
            cache_ctx = {"files_hit": 0, "files_miss": 0, "patch_hit": 0, "patch_miss": 0}
            if not pattern and not symbol:
                print("Error: provide --pattern or --symbol (or both).")
                return

            repo = Repo(repo_path)
            file_commits = list(repo.iter_commits(paths=file_path, max_count=max_commits))
            commit_hashes = [c.hexsha for c in file_commits]
            if not commit_hashes:
                print(f"Error: no commits found for file: {file_path}")
                return

            pattern_re = None
            if pattern:
                try:
                    pattern_re = re.compile(pattern)
                except Exception:
                    pattern_re = None

            def predicate_fn(commit_hash):
                text = get_file_content_at_commit(repo_path, commit_hash, file_path)
                if not text:
                    return False
                ok_pattern = True
                ok_symbol = True
                if pattern:
                    if pattern_re is not None:
                        ok_pattern = bool(pattern_re.search(text))
                    else:
                        ok_pattern = pattern in text
                if symbol:
                    ok_symbol = find_symbol_definition_in_text(text, symbol) is not None
                return ok_pattern and ok_symbol

            introducing_hash = find_introducing_commit_by_predicate(commit_hashes, predicate_fn)
            if not introducing_hash:
                print("Error: no introducing commit matched the predicate.")
                return

            introducing_commit = _get_introducing_commit_cached(
                repo_path, introducing_hash, cache_ctx=cache_ctx, need_patch=True
            )
            intro_msg_lines = introducing_commit.get("message", "").splitlines()[:10]
            intro_msg_preview = "\n".join(intro_msg_lines)
            intro_diff = clean_diff(introducing_commit.get("diff", ""))
            intro_diff_preview = "\n".join(intro_diff.splitlines()[:120])
            intro_files = introducing_commit.get("changed_files", []) or extract_changed_files(
                introducing_commit.get("diff", "")
            )
            structured_changes = detect_structured_changes(introducing_commit.get("diff", ""))
            github_pr_evidence = build_github_pr_evidence(
                repo_path, introducing_hash, disabled=getattr(args, "no_github", False)
            )

            content_at_intro = get_file_content_at_commit(repo_path, introducing_hash, file_path)
            snippet = "(unavailable)"
            if content_at_intro:
                lines = content_at_intro.splitlines()
                match_line = None
                if pattern:
                    for i, line in enumerate(lines, start=1):
                        matched = bool(pattern_re.search(line)) if pattern_re is not None else (pattern in line)
                        if matched:
                            match_line = i
                            break
                if match_line is None and symbol:
                    match_line = find_symbol_definition_in_text(content_at_intro, symbol)
                if match_line is not None:
                    start = max(1, match_line - 5)
                    end = min(len(lines), match_line + 5)
                    chunk = []
                    for ln in range(start, end + 1):
                        chunk.append(f"{ln:>4}: {lines[ln - 1]}")
                    snippet = "\n".join(chunk)

            idx = commit_hashes.index(introducing_hash)
            left = max(0, idx - 2)
            right = min(len(file_commits), idx + 3)
            local_window = file_commits[left:right]
            commit_msgs_for_refs = [introducing_commit.get("message", "")]
            for c in local_window:
                commit_msgs_for_refs.append(c.message.strip() if c.message else "")
            linked_github = collect_linked_github_evidence(repo_path, commit_msgs_for_refs, args)
            selected_github = select_github_evidence(
                f"When was {pattern or symbol} introduced in {file_path}?",
                structured_changes,
                linked_github.get("items", []),
                topn=3,
            )

            used_hashes = []
            seen_hash = set()
            for h in [introducing_hash] + [c.hexsha for c in local_window]:
                if h and h not in seen_hash:
                    seen_hash.add(h)
                    used_hashes.append(h)

            motive_text = ((introducing_commit.get("message", "") or "") + "\n" + (introducing_commit.get("diff", "") or "")).lower()
            if github_pr_evidence and github_pr_evidence.get("available"):
                motive_text += "\n" + (github_pr_evidence.get("title", "") or "").lower()
                motive_text += "\n" + (github_pr_evidence.get("body", "") or "").lower()
            for item in selected_github:
                motive_text += "\n" + (item.get("title", "") or "").lower()
                motive_text += "\n" + (item.get("body_snippet", "") or "").lower()
            motive_keywords = ("because", "so that", "in order to", "to improve", "motivation", "reason", "why")
            has_explicit_motive = any(k in motive_text for k in motive_keywords)

            llm_contexts = [
                {
                    "commit": introducing_commit,
                    "text": (
                        "INTRODUCING COMMIT DETAILS\n"
                        f"hash={introducing_commit.get('hash', '')}\n"
                        f"message={introducing_commit.get('message', '')}\n"
                        f"changed_files={', '.join(intro_files) if intro_files else '(unavailable)'}\n"
                        f"diff_snippet={intro_diff_preview if intro_diff_preview else '(unavailable)'}\n"
                        f"matched_snippet=\n{snippet}"
                    ),
                }
            ]
            if local_window:
                hist_lines = []
                for c in local_window:
                    msg = c.message.strip().splitlines()[0] if c.message else ""
                    hist_lines.append(f"{c.hexsha} {c.committed_datetime.date().isoformat()} {msg}")
                llm_contexts.append(
                    {
                        "commit": {
                            "hash": "local-window",
                            "author": "git log",
                            "date": "",
                            "message": "Local history around introducing commit",
                            "diff": "",
                        },
                        "text": "\n".join(hist_lines),
                    }
                )
            if github_pr_evidence:
                llm_contexts.append(
                    {
                        "commit": {
                            "hash": f"pr-{github_pr_evidence.get('pr_number', '')}",
                            "author": "github",
                            "date": "",
                            "message": "GitHub PR Evidence",
                            "diff": "",
                        },
                        "text": github_pr_evidence.get("text", ""),
                    }
                )
            if selected_github:
                for item in selected_github:
                    llm_contexts.append(
                        {
                            "commit": {
                                "hash": f"github-issue-{item.get('number')}",
                                "author": item.get("author", "github"),
                                "date": item.get("created_at", ""),
                                "message": f"GitHub Issue/PR #{item.get('number')}",
                                "diff": "",
                            },
                            "text": (
                                f"[GITHUB_ISSUE #{item.get('number')}] {item.get('title','')}\n"
                                f"{item.get('body_snippet','')}"
                            ),
                        }
                    )

            snippet_summary = ""
            if snippet and snippet != "(unavailable)":
                for raw in snippet.splitlines():
                    if ":" in raw:
                        snippet_summary = raw.split(":", 1)[1].strip()
                        if snippet_summary:
                            break
            if not snippet_summary:
                snippet_summary = "(snippet unavailable)"

            findings_text = (
                f"{pattern or symbol} was introduced in commit {introducing_commit.get('hash', '')} "
                f"on {introducing_commit.get('date', '')} by {introducing_commit.get('author', '')} "
                f"in {file_path}. Key snippet: {snippet_summary}"
            )
            why_chain = build_why_chain(None, introducing_commit.get("hash", ""), selected_github)
            regress_query_text = f"When was {pattern or symbol} changed in {file_path}?"
            intent_info = detect_query_intent(regress_query_text)

            motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
            if _llm_disabled():
                motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
            else:
                try:
                    motive_question = (
                        f"Why was {pattern or symbol} introduced in {file_path}?\n"
                        f"Evidence hashes you may use: {', '.join(used_hashes)}\n"
                        "Write only the Motive paragraph in at most 3 sentences.\n"
                        "If motive is not explicit, output exactly: Motive is not explicitly stated in commit/issue evidence."
                    )
                    motive_raw = _generate_answer(motive_question, llm_contexts)
                    motive_text_out = _truncate_sentences(sanitize_llm_output(motive_raw), max_sentences=3)
                    if not motive_text_out:
                        motive_text_out = "Motive is not explicitly stated in commit/issue evidence."
                except Exception:
                    motive_text_out = "Motive is not explicitly stated in commit/issue evidence."

            answer_block = build_answer_template(
                findings=findings_text,
                motive=motive_text_out,
                has_explicit_motive=has_explicit_motive,
                structured_changes=structured_changes,
                evidence_text=motive_text,
                missing_items=[],
                intent_info=intent_info,
                changed_files=intro_files,
                selected_github_evidence=selected_github,
                why_chain=why_chain,
            )

            if args.out:
                out_path = args.out
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("outputs", f"regression_report_{ts}.md")
            out_dir = os.path.dirname(out_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            limitations = []
            if not intro_files:
                limitations.append("- changed_files unavailable from introducing commit.")
            if not intro_diff_preview:
                limitations.append("- diff_snippet unavailable for introducing commit.")
            if snippet == "(unavailable)":
                limitations.append("- Could not extract matched evidence snippet from file content at introducing commit.")
            if not has_explicit_motive:
                limitations.append("- Commit message/diff do not explicitly explain motive.")
            if not limitations:
                limitations.append("- No major limitations detected for this regression run.")

            local_history_for_timeline = []
            for c in local_window:
                local_history_for_timeline.append(
                    {
                        "hash": c.hexsha,
                    "author": c.author.name if c.author else "",
                    "date": c.committed_datetime.date().isoformat(),
                    "message": c.message.strip() if c.message else "",
                }
            )
            timeline_items = build_decision_timeline(
                repo_path=repo_path,
                target_file=file_path,
                introducing_commit=introducing_commit,
                retrieved_commits=[],
                local_history_commits=local_history_for_timeline,
                max_items=8,
                symbol_hint=symbol or "",
                cache_ctx=cache_ctx,
            )
            motive_evidence = build_motive_evidence(
                repo_path=repo_path,
                query=regress_query_text,
                evidence={
                    "selected_github_evidence": selected_github,
                    "introducing_commit": introducing_commit,
                    "retrieved_commits": [],
                    "structured_changes": structured_changes,
                    "changed_files": intro_files,
                    "impact_propagation": {},
                    "cochange_top": {},
                    "timeline_items": timeline_items,
                },
            )
            answer_block = build_answer_template(
                findings=findings_text,
                motive=motive_text_out,
                has_explicit_motive=has_explicit_motive,
                structured_changes=structured_changes,
                evidence_text=motive_text,
                missing_items=[],
                intent_info=intent_info,
                changed_files=intro_files,
                selected_github_evidence=selected_github,
                why_chain=why_chain,
                motive_evidence=motive_evidence,
            )

            cochange_targets = [file_path] if file_path else (intro_files[:3] if intro_files else [])
            cochange_top = []
            cochange_error = ""
            cochange_window = 0
            old_window_days_env = os.environ.get("SOCRATIC_WINDOW_DAYS")
            if getattr(args, "window_days", None) is not None:
                os.environ["SOCRATIC_WINDOW_DAYS"] = str(getattr(args, "window_days"))
            try:
                if cochange_targets:
                    cochange_top = compute_cochange(
                        repo_path, cochange_targets, n_commits=300, mode=getattr(args, "cochange_mode", "code")
                    )
                    cochange_window = int(getattr(compute_cochange, "last_window_commits", 0) or 0)
                else:
                    cochange_error = "no target files detected"
            except Exception as exc:
                cochange_error = str(exc)
            finally:
                if old_window_days_env is None:
                    os.environ.pop("SOCRATIC_WINDOW_DAYS", None)
                else:
                    os.environ["SOCRATIC_WINDOW_DAYS"] = old_window_days_env

            target_desc = pattern if pattern else symbol
            lines = [
                "# Socratic Git Regression Report",
                "",
                "## Question",
                f"When was {target_desc} introduced in {file_path}?",
                "",
                "## Introducing Commit",
                f"- hash: {introducing_commit.get('hash', '')}",
                f"- author: {introducing_commit.get('author', '')}",
                f"- date: {introducing_commit.get('date', '')}",
                f"- message: {(introducing_commit.get('message', '') or '').splitlines()[0] if introducing_commit.get('message') else ''}",
                "",
                "## Evidence Snippets",
                f"- changed_files: {', '.join(intro_files) if intro_files else '(unavailable)'}",
                "- matched snippet (+/- 5 lines):",
                "```text",
                snippet,
                "```",
                "- diff_snippet:",
                "```diff",
                intro_diff_preview if intro_diff_preview else "(unavailable)",
                "```",
                "",
                "## Structured Changes",
            ]

            if structured_changes:
                for ch in structured_changes[:10]:
                    lb = (ch.get("line_before", "") or "")[:120]
                    la = (ch.get("line_after", "") or "")[:120]
                    lines.append(
                        f"- {ch.get('key', '')}: {ch.get('from', '')} -> {ch.get('to', '')} "
                        f"(file={ch.get('file', '')}, hunk={ch.get('hunk', '')})"
                    )
                    lines.append(f"  - line_before: {lb}")
                    lines.append(f"  - line_after: {la}")
            else:
                lines.append("- (none detected)")

            lines.extend(
                [
                    "",
                    "## Cross-File Impact (Co-change)",
                    f"- target_files: {', '.join(cochange_targets) if cochange_targets else '(none)'}",
                    f"- window_commits: {cochange_window}",
                    "",
                    "### Top Related Code Files",
                ]
            )
            if cochange_top and cochange_top.get("top_code"):
                dep_target = (cochange_targets or [""])[0]
                for row in cochange_top.get("top_code", []):
                    lines.append(
                        f"- file={row.get('file')}, co_count={row.get('co_count')}, "
                        f"lift={row.get('lift')}, p_other={row.get('p_other')}, "
                        f"p_other_given_target={row.get('p_other_given_target')}"
                    )
                    dep = detect_structural_dependency(repo_path, dep_target, row.get("file", ""))
                    try:
                        lift_value = float(row.get("lift", 0) or 0)
                    except Exception:
                        lift_value = 0.0
                    if lift_value > 2 and dep.get("ast_call_relation") is True:
                        structural_confidence = "High"
                    elif dep.get("import_relation") is True:
                        structural_confidence = "Medium"
                    else:
                        structural_confidence = "Low"
                    lines.append("  - Structural Dependency Signals")
                    lines.append(f"  - structural_dependency: {dep}")
                    lines.append(f"  - structural_confidence: {structural_confidence}")
                    if dep.get("ast_skipped_large"):
                        lines.append("  - AST skipped: file too large")
            else:
                lines.append("- (none detected)")

            lines.extend(["", "### Top Related Config Files"])
            if cochange_top and cochange_top.get("top_config"):
                for row in cochange_top.get("top_config", []):
                    lines.append(
                        f"- file={row.get('file')}, co_count={row.get('co_count')}, "
                        f"lift={row.get('lift')}, p_other={row.get('p_other')}, "
                        f"p_other_given_target={row.get('p_other_given_target')}"
                    )
            else:
                lines.append("- (none detected)")

            lines.extend(["", "### Ignored (Artifacts / Data / Build)"])
            if cochange_top and cochange_top.get("ignored"):
                for row in cochange_top.get("ignored", []):
                    lines.append(
                        f"- file={row.get('file')}, class={row.get('class')}, co_count={row.get('co_count')}, "
                        f"lift={row.get('lift')}"
                    )
            else:
                lines.append("- (none detected)")

            if cochange_error:
                lines.append(f"- cochange unavailable: {cochange_error}")
            lines.append(
                "- interpretation: This suggests structural coupling (statistical), not proven causality."
            )

            lines.extend(["", "### Coupling Summary"])
            summary_lines = []
            top_code_rows = (cochange_top or {}).get("top_code", [])
            top_config_rows = (cochange_top or {}).get("top_config", [])
            if top_code_rows:
                for row in top_code_rows[:3]:
                    summary_lines.append(
                        f"- {', '.join(cochange_targets[:1] or ['target files'])} and {row.get('file')} frequently co-change, "
                        "suggesting they may be part of the same implementation flow."
                    )
            if top_config_rows:
                for row in top_config_rows[:1]:
                    summary_lines.append(
                        f"- Config file {row.get('file')} likely changes with target logic, suggesting runtime/threshold alignment may be needed."
                    )
            if not summary_lines:
                summary_lines.append("- (none detected)")
            lines.extend(summary_lines[:4])

            lines.extend(["", "### Co-change Evidence (Top 2)"])
            evidence_err = ""
            try:
                top2 = top_code_rows[:2]
                if not top2:
                    lines.append("- (none detected)")
                for row in top2:
                    lines.append(f"- file: {row.get('file')}")
                    lines.append("  - supporting_commits:")
                    ev = get_cochange_evidence(
                        repo_path,
                        cochange_targets,
                        row.get("file", ""),
                        n_commits=300,
                        max_commits=3,
                    )
                    if ev:
                        for c in ev:
                            lines.append(f"    - {c.get('hash')} {c.get('date')} {c.get('message')}")
                    else:
                        lines.append("    - (none detected)")
            except Exception as exc:
                evidence_err = str(exc)
            if evidence_err:
                lines.append(f"- cochange evidence unavailable: {evidence_err}")

            lines.extend(
                [
                "",
                "## Local History Window",
            ]
            )

            for c in local_window:
                msg = c.message.strip().splitlines()[0] if c.message else ""
                lines.append(f"- {c.hexsha} {c.committed_datetime.date().isoformat()} {msg}")

            lines.extend(["", "## Decision Timeline (Filtered Evidence)"])
            lines.append("Filtering rule: noise commits removed (merge/delete/docs-only)")
            if timeline_items:
                for it in timeline_items:
                    sigs = ", ".join(it.get("signals", [])) if it.get("signals") else "(none)"
                    lines.append(
                        f"- {it.get('date','')} [{it.get('commit_hash','')}] {it.get('author','')}"
                    )
                    lines.append(f"  score={it.get('decision_score', 0)}")
                    lines.append(f"  signals: {sigs}")
                    lines.append(f"  - note: {it.get('short_note','')}")
                    lines.append(f"  changed_files: {it.get('changed_files_count', 0)}")
            else:
                lines.append("- (insufficient history for timeline)")

            lines.extend([""] + render_motive_evidence_lines(motive_evidence))
            lines.extend(
                [
                    "",
                "## Answer (Evidence-Driven)",
                answer_block,
                "",
                "## Limitations",
            ]
            )
            if github_pr_evidence:
                lines.extend(
                    [
                        "",
                        "## GitHub PR Evidence (Optional)",
                    ]
                )
                if github_pr_evidence.get("available"):
                    lines.extend(github_pr_evidence.get("text", "").splitlines()[1:])
                else:
                    lines.append(f"- pr: #{github_pr_evidence.get('pr_number', '')}")
                    lines.append("- PR evidence unavailable (rate limit or not found)")

            lines.extend(
                [
                    "",
                    "## Linked GitHub Evidence (Optional)",
                ]
            )
            if not linked_github.get("enabled"):
                lines.append("- GitHub evidence disabled.")
            elif not linked_github.get("owner"):
                lines.append("- GitHub remote not detected; skipping GitHub evidence.")
            elif linked_github.get("items"):
                selected_map = {x.get("number"): x for x in selected_github}
                for item in linked_github.get("items", [])[:5]:
                    chosen = selected_map.get(item.get("number"), item)
                    lines.extend(
                        [
                            f"- Issue/PR #{item.get('number')} (state={item.get('state','')}, author={item.get('author','')}, url={item.get('html_url','')})",
                            f"  - fetched_from: {item.get('_source', 'network')}",
                            f"  - title: {item.get('title','')}",
                            f"  - body_snippet: {(item.get('body_snippet','') or '(empty)')[:1200]}",
                            f"  - score_breakdown: {chosen.get('score_breakdown', {})}",
                        ]
                    )
                extra = linked_github.get("items", [])[5:]
                if extra:
                    extra_nums = ", ".join(f"#{x.get('number')}" for x in extra)
                    lines.append(f"- additional_refs: {extra_nums}")
            else:
                if linked_github.get("numbers"):
                    lines.append(f"- refs_detected: {', '.join(f'#{n}' for n in linked_github.get('numbers', []))}")
                if linked_github.get("urls"):
                    lines.append(f"- dry_run_urls: {', '.join(linked_github.get('urls', [])[:5])}")
                if linked_github.get("errors"):
                    lines.append(f"- GitHub evidence unavailable: {'; '.join(linked_github.get('errors', [])[:3])}")
                else:
                    lines.append("- GitHub evidence unavailable.")
            if selected_github:
                picked = ", ".join(
                    f"#{x.get('number')} score={x.get('score', 0)} breakdown={x.get('score_breakdown', {})}"
                    for x in selected_github
                )
                lines.append(f"- selected_github_evidence: [{picked}]")
            else:
                lines.append("- selected_github_evidence: (none)")
            lines.extend(limitations)
            lines.append("")

            markdown_report = "\n".join(lines)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(markdown_report)

            if getattr(args, "html_out", ""):
                html_out = args.html_out
                html_dir = os.path.dirname(html_out) or "."
                os.makedirs(html_dir, exist_ok=True)
                html_text = render_html_report(
                    markdown_report,
                    {
                        "repo_path": repo_path,
                        "question": f"When was {target_desc} introduced in {file_path}?",
                        "motive_score": (motive_evidence or {}).get("motive_score", 0),
                        "motive_confidence": _extract_motive_confidence(answer_block),
                    },
                )
                with open(html_out, "w", encoding="utf-8") as f:
                    f.write(html_text)
                print(f"html report saved to {html_out}")

            print(f"report saved to {out_path}")
            return

        if args.command == "bisect":
            good_hash = args.good
            bad_hash = args.bad
            cmd = args.cmd
            max_steps = max(1, args.max_steps)
            bisect_mode = getattr(args, "bisect_mode", "clone")

            source_repo = Repo(repo_path)
            head_before = None
            head_ref_before = "DETACHED"
            try:
                head_before = source_repo.git.rev_parse("HEAD").strip()
                ref = source_repo.git.rev_parse("--abbrev-ref", "HEAD").strip()
                head_ref_before = "DETACHED" if ref == "HEAD" else ref
            except Exception:
                head_before = None

            if bisect_mode == "worktree" and source_repo.is_dirty(untracked_files=True):
                print("Error: repository has uncommitted changes. Please commit/stash before bisect.")
                return

            worktree_path = ""
            cleanup_status = "ok"
            head_restored = "unknown"
            bisect_repo_path = repo_path
            setup_seconds = 0.0
            bisect_seconds = 0.0
            cleanup_seconds = 0.0
            t_setup = time.perf_counter()
            try:
                if bisect_mode == "clone":
                    worktree_path = os.path.join("/tmp", f"socratic_bisect_clone_{int(time.time() * 1000)}")
                    subprocess.run(
                        ["git", "clone", repo_path, worktree_path],
                        check=True,
                        capture_output=True,
                    )
                    bisect_repo_path = worktree_path
                else:
                    worktree_path = os.path.join("/tmp", f"socratic_bisect_worktree_{int(time.time() * 1000)}")
                    subprocess.run(
                        ["git", "-C", repo_path, "worktree", "add", "--detach", worktree_path, bad_hash],
                        check=True,
                        capture_output=True,
                    )
                    bisect_repo_path = worktree_path
            except Exception as exc:
                print(f"Error: failed to prepare bisect {bisect_mode} workspace. Details: {exc}")
                return
            setup_seconds = time.perf_counter() - t_setup

            repo = Repo(bisect_repo_path)

            try:
                range_hashes = repo.git.rev_list("--ancestry-path", f"{good_hash}..{bad_hash}").splitlines()
            except Exception as exc:
                print(f"Error: failed to build commit range from good/bad. Details: {exc}")
                cleanup_status = "fail"
                range_hashes = []
            if not range_hashes:
                if cleanup_status == "fail":
                    pass
                else:
                    print("Error: no commits found between --good and --bad.")
                # fallthrough to cleanup/report below
                first_bad = None
                steps = []
                tested = {}
                good_run = {"returncode": -1}
                bad_run = {"returncode": -1}
                commits_range = []
                goto_report = True
            else:
                goto_report = False

            if not goto_report:
                t_bisect = time.perf_counter()
                commits_range = list(reversed(range_hashes))  # oldest -> newest, excludes good, includes bad

                # Run boundary checks and enforce bisect preconditions.
                good_run = run_cmd_at_commit(repo, good_hash, cmd)
                bad_run = run_cmd_at_commit(repo, bad_hash, cmd)
                if good_run.get("returncode", 1) != 0:
                    print("Provided good commit does not pass; bisect precondition violated.")
                    goto_report = True
                    first_bad = None
                    steps = []
                    tested = {}
                elif bad_run.get("returncode", 0) == 0:
                    print("Provided bad commit does not fail; bisect precondition violated.")
                    goto_report = True
                    first_bad = None
                    steps = []
                    tested = {}

            if not goto_report:
                def is_bad_fn(commit_hash):
                    run = run_cmd_at_commit(repo, commit_hash, cmd)
                    return {
                        "is_bad": run.get("returncode", 1) != 0,
                        "returncode": run.get("returncode", -1),
                        "stdout_tail": run.get("stdout_tail", ""),
                        "stderr_tail": run.get("stderr_tail", ""),
                    }

                first_bad, steps, tested = bisect_search(commits_range, is_bad_fn, max_steps=max_steps)
                bisect_seconds = time.perf_counter() - t_bisect
            boundary_steps = [
                {"commit": good_hash, "status": "good", "returncode": good_run.get("returncode", -1), "note": "boundary=good"},
                {"commit": bad_hash, "status": "bad", "returncode": bad_run.get("returncode", -1), "note": "boundary=bad"},
            ]
            steps_all = boundary_steps + [
                {
                    "commit": s.get("commit", ""),
                    "status": s.get("status", ""),
                    "returncode": s.get("returncode", -1),
                    "note": "mid",
                }
                for s in steps
            ]
            first_bad_commit = {}
            changed_files = []
            diff_preview = ""
            stdout_tail = "(empty)"
            stderr_tail = "(empty)"
            motive = "Motive is not explicitly stated in commit message/diff."
            if first_bad:
                first_bad_commit = _get_introducing_commit_cached(
                    repo_path, first_bad, cache_ctx=None, need_patch=True
                )
                changed_files = first_bad_commit.get("changed_files", []) or extract_changed_files(first_bad_commit.get("diff", ""))
                diff_preview = "\n".join((first_bad_commit.get("diff", "") or "").splitlines()[:120])
                fail_log = tested.get(first_bad, {})
                stdout_tail = fail_log.get("stdout_tail", "") or "(empty)"
                stderr_tail = fail_log.get("stderr_tail", "") or "(empty)"
                github_pr_evidence = build_github_pr_evidence(
                    repo_path, first_bad, disabled=getattr(args, "no_github", False)
                )

                motive_text = ((first_bad_commit.get("message", "") or "") + "\n" + (first_bad_commit.get("diff", "") or "")).lower()
                if github_pr_evidence and github_pr_evidence.get("available"):
                    motive_text += "\n" + (github_pr_evidence.get("title", "") or "").lower()
                    motive_text += "\n" + (github_pr_evidence.get("body", "") or "").lower()
                motive_keywords = ("because", "so that", "in order to", "to improve", "motivation", "reason", "why")
                has_explicit_motive = any(k in motive_text for k in motive_keywords)
                if has_explicit_motive:
                    if github_pr_evidence and github_pr_evidence.get("available"):
                        motive = "Possible motivation cues found in commit and/or linked PR evidence."
                    else:
                        motive = "Commit message/diff contains possible motivation cues; review message and diff context."

                findings = (
                    f"First bad commit is {first_bad} (date={first_bad_commit.get('date', '')}, "
                    f"author={first_bad_commit.get('author', '')}) under command: {cmd}"
                )
                if len(commits_range) <= 1:
                    findings += " Only 1 candidate commit in range; no mid-point tests were necessary."
            else:
                findings = f"No failing commit detected in range ({good_hash}..{bad_hash}) under command: {cmd}"
                github_pr_evidence = None

            if args.out:
                out_path = args.out
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("outputs", f"bisect_report_{ts}.md")
            out_dir = os.path.dirname(out_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            # Cleanup for worktree/clone
            t_cleanup = time.perf_counter()
            try:
                subprocess.run(
                    ["git", "-C", bisect_repo_path, "bisect", "reset"],
                    check=False,
                    capture_output=True,
                )
            except Exception:
                cleanup_status = "fail"
            try:
                if bisect_mode == "worktree" and worktree_path:
                    subprocess.run(
                        ["git", "-C", repo_path, "worktree", "remove", "--force", worktree_path],
                        check=False,
                        capture_output=True,
                    )
                    subprocess.run(
                        ["git", "-C", repo_path, "worktree", "prune"],
                        check=False,
                        capture_output=True,
                    )
                elif bisect_mode == "clone" and worktree_path:
                    subprocess.run(["rm", "-rf", worktree_path], check=False, capture_output=True)
            except Exception:
                cleanup_status = "fail"
            cleanup_seconds = time.perf_counter() - t_cleanup
            try:
                head_after = source_repo.git.rev_parse("HEAD").strip()
                head_ref_after_raw = source_repo.git.rev_parse("--abbrev-ref", "HEAD").strip()
                head_ref_after = "DETACHED" if head_ref_after_raw == "HEAD" else head_ref_after_raw
                head_restored = (
                    "yes" if head_before and head_after == head_before and head_ref_after == head_ref_before else "no"
                )
            except Exception:
                head_restored = "no"
                head_after = ""
                head_ref_after = "UNKNOWN"

            total_seconds = setup_seconds + bisect_seconds + cleanup_seconds
            if getattr(args, "verbose", False):
                print(f"setup_seconds={setup_seconds:.2f}")
                print(f"bisect_seconds={bisect_seconds:.2f}")
                print(f"cleanup_seconds={cleanup_seconds:.2f}")
                print(f"total_seconds={total_seconds:.2f}")

            lines = [
                "# Socratic Git Bisect Report",
                "",
                "## Inputs",
                f"- repo: {repo_path}",
                f"- good: {good_hash}",
                f"- bad: {bad_hash}",
                f"- cmd: `{cmd}`",
                f"- bisect_mode: {bisect_mode}",
                f"- worktree_path: {worktree_path or '(none)'}",
                f"- cleanup: {cleanup_status}",
                f"- head_restored: {head_restored}",
                f"- source_head_ref_before: {head_ref_before}",
                f"- source_head_sha_before: {head_before or ''}",
                f"- source_head_ref_after: {head_ref_after}",
                f"- source_head_sha_after: {head_after}",
                f"- setup_seconds: {setup_seconds:.2f}",
                f"- bisect_seconds: {bisect_seconds:.2f}",
                f"- cleanup_seconds: {cleanup_seconds:.2f}",
                f"- total_seconds: {total_seconds:.2f}",
                f"- max_steps: {max_steps}",
                "- safety_note: `--cmd` executes arbitrary shell commands; use at your own risk.",
                "",
                "## Reproducibility",
                f"- good_returncode: {good_run.get('returncode', -1)}",
                f"- bad_returncode: {bad_run.get('returncode', -1)}",
                "",
                "## Steps",
                "| commit | pass_fail | returncode | note |",
                "|---|---|---|---|",
            ]
            for s in steps_all:
                status = "fail" if s["status"] == "bad" else "pass"
                lines.append(f"| {s['commit']} | {status} | {s['returncode']} | {s.get('note', '')} |")

            lines.extend(
                [
                    "",
                    "## First Bad Commit",
                    f"- hash: {first_bad if first_bad else '(none)'}",
                    f"- author: {first_bad_commit.get('author', '') if first_bad else ''}",
                    f"- date: {first_bad_commit.get('date', '') if first_bad else ''}",
                    (
                        f"- message: {(first_bad_commit.get('message', '') or '').splitlines()[0]}"
                        if first_bad and first_bad_commit.get("message")
                        else "- message: "
                    ),
                    f"- changed_files: {', '.join(changed_files) if changed_files else '(unavailable)'}",
                ]
            )
            if first_bad:
                lines.extend(
                    [
                        "- diff_snippet:",
                        "```diff",
                        diff_preview if diff_preview else "(unavailable)",
                        "```",
                    ]
                )
            else:
                lines.append("- diff_snippet: (unavailable)")

            lines.extend(
                [
                    "",
                    "## Failure Log Excerpt",
                    "### stdout_tail",
                    "```text",
                    stdout_tail,
                    "```",
                    "### stderr_tail",
                    "```text",
                    stderr_tail,
                    "```",
                    "",
                    "## Findings",
                    findings,
                    "",
                    "## Motive",
                    motive,
                ]
            )
            if github_pr_evidence:
                lines.extend(
                    [
                        "",
                        "## GitHub PR Evidence (Optional)",
                    ]
                )
                if github_pr_evidence.get("available"):
                    lines.extend(github_pr_evidence.get("text", "").splitlines()[1:])
                else:
                    lines.append(f"- pr: #{github_pr_evidence.get('pr_number', '')}")
                    lines.append("- PR evidence unavailable (rate limit or not found)")

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(f"report saved to {out_path}")
            return

        question = args.q
        topk = max(1, args.topk)
        contexts, _used_retrieval = resolve_contexts(
            repo_path,
            question,
            topk=topk,
            retrieval_mode=getattr(args, "retrieval", "vector"),
            command_label="ask",
        )
        if not contexts:
            print("Error: no related commits retrieved.")
            return

        target_info = get_symbol_or_lines_from_query(question)
        detected_file = target_info.get("file")
        detected_line = target_info.get("line")
        detected_symbol = target_info.get("symbol")
        resolved_line = None
        blame_info = ""
        introducing_commit = {}
        introducing_details_text = ""
        local_history_window = []

        # Phase 5 example:
        # python socratic_mvp.py ask --repo <repo> --q "In src/main.py, why was hybrid_search added?" --topk 3
        if detected_file and not detected_line and detected_symbol:
            resolved_line = find_symbol_definition(repo_path, detected_file, detected_symbol)
            if resolved_line:
                detected_line = resolved_line
            else:
                fallback_file, fallback_line = find_symbol_in_repo(
                    repo_path, detected_symbol, preferred_file=detected_file
                )
                if fallback_file and fallback_line:
                    detected_file = fallback_file
                    resolved_line = fallback_line
                    detected_line = fallback_line

        if detected_file and detected_line:
            blame_dict = get_blame_for_line(repo_path, detected_file, detected_line)
            if blame_dict:
                blame_info = (
                    f"file={blame_dict.get('file', '')}; "
                    f"line={blame_dict.get('line', '')}; "
                    f"commit={blame_dict.get('hash', '')}; "
                    f"author={blame_dict.get('author', '')}; "
                    f"date={blame_dict.get('date', '')}; "
                    f"line_content={blame_dict.get('line_content', '')}"
                )
                introducing_commit = _get_introducing_commit_cached(
                    repo_path, blame_dict.get("hash", ""), cache_ctx=None, need_patch=True
                )
        elif detected_file:
            blame_info = get_file_blame(repo_path, detected_file)

        if introducing_commit:
            intro_msg_lines = introducing_commit.get("message", "").splitlines()[:10]
            intro_msg_preview = "\n".join(intro_msg_lines)
            intro_diff_clean = clean_diff(introducing_commit.get("diff", ""))
            intro_diff_lines = intro_diff_clean.splitlines()[:120]
            intro_diff_preview = "\n".join(intro_diff_lines)
            intro_files = extract_changed_files(introducing_commit.get("diff", ""))
            introducing_details_text = (
                "=== Introducing Commit Details ===\n"
                f"hash: {introducing_commit.get('hash', '')}\n"
                f"author/date: {introducing_commit.get('author', '')} / {introducing_commit.get('date', '')}\n"
                f"message:\n{intro_msg_preview}\n"
                f"files: {', '.join(intro_files) if intro_files else '(none)'}\n"
                f"diff_snippet:\n{intro_diff_preview}"
            )

        if detected_file:
            local_history_window = get_local_history_window(repo_path, detected_file, n=3)
        if local_history_window:
            print("=== Local History Window (file) ===")
            for row in local_history_window:
                msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
                print(f"{row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())

        print_evidence(contexts, topk=topk)
        if detected_file:
            if detected_line:
                print(f"Detected target: file={detected_file}, line={detected_line}")
            else:
                print(f"Detected target: file={detected_file}")
        if detected_symbol:
            print(f"Detected symbol: {detected_symbol}")
        if resolved_line:
            print(f"Resolved line: {resolved_line}")
        if introducing_commit:
            intro_msg = introducing_commit.get("message", "").splitlines()[0] if introducing_commit.get("message") else ""
            print(f"Introducing commit: {introducing_commit.get('hash', '')} {intro_msg}".rstrip())
        if introducing_details_text:
            print(introducing_details_text)
        if blame_info:
            print(f"Blame Info: {blame_info}")

        print("=== Answer ===")
        try:
            llm_contexts = contexts
            if introducing_details_text:
                llm_contexts = [
                    {
                        "commit": introducing_commit,
                        "text": introducing_details_text,
                    }
                ] + llm_contexts
            if blame_info:
                llm_contexts = [
                    {
                        "commit": {
                            "hash": "blame",
                            "author": "git blame",
                            "date": "",
                            "message": "Blame summary from referenced file",
                            "diff": "",
                        },
                        "text": blame_info,
                    }
                ] + llm_contexts
            if local_history_window:
                history_lines = []
                for row in local_history_window:
                    msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
                    history_lines.append(f"{row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())
                history_text = "=== Local History Window (file) ===\n" + "\n".join(history_lines)
                llm_contexts = llm_contexts + [
                    {
                        "commit": {
                            "hash": "local-history",
                            "author": "git log",
                            "date": "",
                            "message": "Recent file-local history window",
                            "diff": "",
                        },
                        "text": history_text,
                    }
                ]
            if _llm_disabled():
                answer = "I don't know (insufficient evidence in commit message/diff)."
            else:
                answer = _generate_answer(question, llm_contexts)
        except Exception as exc:
            msg = str(exc)
            if "No such file or directory" in msg or "not found" in msg.lower():
                print(f"Error: model may be missing/not downloaded. Details: {exc}")
            else:
                print(f"Error: failed to run local model. Details: {exc}")
            return
        print(answer)

    except RuntimeError as exc:
        print(f"Error: {exc}")
    except Exception as exc:
        print(f"Error: {exc}")
        return


if __name__ == "__main__":
    main()
