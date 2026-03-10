"""
nns_engine/article_analyser.py
════════════════════════════════
LLM-powered Negative News Screening article analyser.

What this solves that rules cannot:
  ─────────────────────────────────────────────────────────────────────
  1. ENTITY DISAMBIGUATION
     "JOHN SMITH arrested for money laundering" — is this your customer
     or a different John Smith?  Fuzzy name scoring cannot distinguish.
     The LLM reads the article and checks whether context (company,
     country, role, age) corroborates or contradicts the match.

  2. ACCURATE SEVERITY RECLASSIFICATION
     The vendor's pre-assigned category_severity is often too coarse.
     "Fined £500 for minor admin breach" and "Convicted of $50M wire
     fraud" both arrive as FRAUD/0.70.  The LLM reads the text and
     returns a calibrated severity_score replacing the vendor figure.

  3. RELATIONSHIP EXTRACTION FOR GRAPH ENGINE
     "HASSAN AL-RASHID, brother of sanctioned MOHAMMED AL-RASHID" —
     the LLM extracts the relationship type and linked name so the
     graph_engine can create a new evidence-based edge.
  ─────────────────────────────────────────────────────────────────────

Architecture:
  • Only fires when name_score >= ACTIVATION_THRESHOLD (default 0.40)
    and a headline_snippet is present — not on every record.
  • Calls Anthropic claude-sonnet-4-20250514 via /v1/messages.
  • Returns structured ArticleAnalysis dataclass (strict JSON prompt).
  • Graceful fallback: if API call fails for any reason, returns an
    ArticleAnalysis with llm_used=False and original rule scores intact.
    Screening is never blocked by an LLM failure.
  • Thread-safe: no shared mutable state.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
ACTIVATION_THRESHOLD  = 0.40   # min name_score required to invoke LLM
MODEL_ID              = "claude-sonnet-4-20250514"
MAX_TOKENS            = 512
API_TIMEOUT_SECS      = 8.0    # wall-clock timeout for the HTTP call
API_URL               = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION     = "2023-06-01"

# Confidence floor: LLM entity_confirmed=False below this => drop hit
ENTITY_CONFIRMED_MIN  = 0.30

# How much weight the LLM severity gets vs the vendor's original value
LLM_SEVERITY_WEIGHT   = 0.70   # 70 % LLM, 30 % vendor
VENDOR_SEVERITY_WEIGHT = 0.30


# ══════════════════════════════════════════════════════════════════════════════
#  Output dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArticleAnalysis:
    """
    Result of one LLM article analysis call.

    When llm_used=False (fallback mode), all LLM-derived fields carry
    neutral / pass-through values so downstream code is unaffected.
    """
    # ── Provenance ──────────────────────────────────────────────────────────
    llm_used:               bool  = False
    fallback_reason:        str   = ""       # populated when llm_used=False
    latency_ms:             float = 0.0

    # ── Entity disambiguation ────────────────────────────────────────────────
    entity_confirmed:       bool  = True     # True = likely same person/org
    entity_confidence:      float = 0.50     # 0–1 confidence in confirmation
    disambiguation_note:    str   = ""       # analyst-readable explanation

    # ── Severity reclassification ────────────────────────────────────────────
    llm_severity_score:     float = 0.50     # 0–1 calibrated by LLM
    blended_severity_score: float = 0.50     # weighted blend with vendor
    severity_rationale:     str   = ""       # analyst-readable explanation

    # ── Relationship extraction ──────────────────────────────────────────────
    relationships:          list  = field(default_factory=list)
    # Each item: {"linked_name": str, "relationship_type": str, "confidence": float}

    # ── Adjusted rule score ──────────────────────────────────────────────────
    adjusted_rule_score:    float = 0.0      # replaces NNSArticleRule.rule_score
    should_suppress:        bool  = False    # True = entity confirmed different person


# ══════════════════════════════════════════════════════════════════════════════
#  System prompt
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an AML (Anti-Money Laundering) compliance analyst assistant.
You analyse news article snippets to help determine whether a named person or organisation
in an article is the same entity as a screening subject.

You must respond ONLY with a valid JSON object. No preamble, no markdown, no explanation.
No trailing commas. The JSON must match this exact schema:

{
  "entity_confirmed": true or false,
  "entity_confidence": float between 0.0 and 1.0,
  "disambiguation_note": "one sentence explaining your reasoning",
  "severity_score": float between 0.0 and 1.0,
  "severity_rationale": "one sentence justifying the severity",
  "relationships": [
    {
      "linked_name": "full name of related person or org",
      "relationship_type": "one of: FAMILY, BUSINESS_ASSOCIATE, EMPLOYER, SUBSIDIARY, BENEFICIAL_OWNER, NOMINEE, UNKNOWN",
      "confidence": float between 0.0 and 1.0
    }
  ]
}

Severity scoring guide:
  1.0  — Terrorism / WMD / proliferation financing
  0.95 — Sanctions evasion / arms trafficking
  0.90 — Drug trafficking / money laundering conviction
  0.80 — Bribery / corruption / organised crime
  0.70 — Fraud conviction / human trafficking
  0.60 — Regulatory fine / insider trading allegation
  0.50 — Minor regulatory breach / allegation only
  0.30 — Historical minor incident / dropped charges

Entity confirmation guide:
  Set entity_confirmed=true and confidence 0.7–1.0 when the article
  mentions context that MATCHES the subject (same country, role, company,
  approximate age, or other corroborating detail).
  Set entity_confirmed=false and confidence 0.7–1.0 when context clearly
  CONTRADICTS (different country, different company, article refers to a
  well-known public figure who is clearly not the customer).
  Set confidence 0.3–0.6 when context is ambiguous or absent.

Extract ALL named third parties (persons or organisations) mentioned in
the article who have a direct relationship to the subject."""


def _build_user_prompt(
    subject_name: str,
    subject_context: dict,
    headline: str,
    category: str,
    source: str,
    pub_date: str,
) -> str:
    """Build the per-article user message."""
    ctx_parts = []
    if subject_context.get("entity_type"):
        ctx_parts.append(f"Type: {subject_context['entity_type']}")
    if subject_context.get("country"):
        ctx_parts.append(f"Country: {subject_context['country']}")
    if subject_context.get("role"):
        ctx_parts.append(f"Role/occupation: {subject_context['role']}")
    if subject_context.get("dob"):
        ctx_parts.append(f"DOB: {subject_context['dob']}")
    context_str = " | ".join(ctx_parts) if ctx_parts else "No additional context"

    return f"""SCREENING SUBJECT: {subject_name}
SUBJECT CONTEXT: {context_str}

ARTICLE:
  Headline: {headline}
  Category: {category}
  Source: {source}
  Date: {pub_date}

Analyse whether the subject in this article is the same entity as the screening subject.
Return JSON only."""


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP call
# ══════════════════════════════════════════════════════════════════════════════

def _call_anthropic_api(user_message: str) -> dict:
    """
    Call the Anthropic /v1/messages endpoint.
    Returns the parsed JSON response dict.
    Raises RuntimeError on any failure (caller handles gracefully).
    """
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model":      MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "system":     _SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": user_message}],
    }).encode("utf-8")

    req = urllib.request.Request(
        url=API_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type":      "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECS) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Anthropic API HTTP {e.code}: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Anthropic API call failed: {e}") from e

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse API response: {e}") from e

    content_blocks = body.get("content", [])
    text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

    if not text.strip():
        raise RuntimeError("Anthropic API returned empty content")

    # Strip any accidental markdown fences
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}\nRaw: {text[:300]}") from e


# ══════════════════════════════════════════════════════════════════════════════
#  Parse and validate LLM response
# ══════════════════════════════════════════════════════════════════════════════

def _parse_llm_response(
    raw: dict,
    original_name_score: float,
    vendor_severity: float,
) -> ArticleAnalysis:
    """Convert raw LLM JSON dict into a validated ArticleAnalysis."""

    def _clamp(v, lo=0.0, hi=1.0) -> float:
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return (lo + hi) / 2

    entity_confirmed  = bool(raw.get("entity_confirmed", True))
    entity_confidence = _clamp(raw.get("entity_confidence", 0.5))

    llm_severity    = _clamp(raw.get("severity_score", vendor_severity))
    blended_sev     = round(
        LLM_SEVERITY_WEIGHT * llm_severity +
        VENDOR_SEVERITY_WEIGHT * vendor_severity,
        4
    )

    # Parse relationships
    rels = []
    for r in raw.get("relationships", []):
        if not isinstance(r, dict):
            continue
        linked = str(r.get("linked_name", "")).strip()
        if not linked:
            continue
        rels.append({
            "linked_name":       linked,
            "relationship_type": str(r.get("relationship_type", "UNKNOWN")).upper(),
            "confidence":        _clamp(r.get("confidence", 0.5)),
        })

    # Suppress if LLM is confident this is a different entity
    should_suppress = (
        not entity_confirmed and
        entity_confidence >= ENTITY_CONFIRMED_MIN
    )

    # Adjusted rule score:
    #   original_name_score × blended_severity
    #   If suppressed → 0.0
    if should_suppress:
        adjusted = 0.0
    else:
        adjusted = round(original_name_score * blended_sev, 4)

    return ArticleAnalysis(
        llm_used               = True,
        fallback_reason        = "",
        entity_confirmed       = entity_confirmed,
        entity_confidence      = entity_confidence,
        disambiguation_note    = str(raw.get("disambiguation_note", ""))[:300],
        llm_severity_score     = llm_severity,
        blended_severity_score = blended_sev,
        severity_rationale     = str(raw.get("severity_rationale", ""))[:300],
        relationships          = rels,
        adjusted_rule_score    = adjusted,
        should_suppress        = should_suppress,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Fallback builder
# ══════════════════════════════════════════════════════════════════════════════

def _fallback(
    reason: str,
    original_rule_score: float,
    vendor_severity: float,
) -> ArticleAnalysis:
    """Return a pass-through ArticleAnalysis leaving scores unchanged."""
    logger.debug("[ArticleAnalyser] Fallback — %s", reason)
    return ArticleAnalysis(
        llm_used               = False,
        fallback_reason        = reason,
        entity_confirmed       = True,
        entity_confidence      = 0.50,
        llm_severity_score     = vendor_severity,
        blended_severity_score = vendor_severity,
        adjusted_rule_score    = original_rule_score,
        should_suppress        = False,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def analyse_article(
    inp: dict,
    row: pd.Series,
    name_score: float,
    original_rule_score: float,
) -> ArticleAnalysis:
    """
    Analyse one NNS article match with the LLM.

    Parameters
    ----------
    inp                  : Screening input dict (name, entity_type, country, dob …)
    row                  : NNS_ARTICLES candidate row (pd.Series)
    name_score           : Raw name match score from RuleChain [0–1]
    original_rule_score  : Existing rule score (name_score × recency × vendor_sev)

    Returns
    -------
    ArticleAnalysis
      • If name_score < ACTIVATION_THRESHOLD → fallback (no LLM call)
      • If no headline_snippet               → fallback (no LLM call)
      • If API fails for any reason          → fallback (score unchanged)
      • Otherwise                            → full LLM analysis result
    """
    headline = str(row.get("headline_snippet", "")).strip()
    category = str(row.get("category", "UNKNOWN"))
    source   = str(row.get("source_publication", "UNKNOWN"))
    pub_date = str(row.get("publication_date", ""))
    vendor_sev = float(row.get("category_severity", 0.50))

    # ── Gate 1: name score too low → not worth calling LLM ────────────────
    if name_score < ACTIVATION_THRESHOLD:
        return _fallback(
            f"name_score {name_score:.3f} < threshold {ACTIVATION_THRESHOLD}",
            original_rule_score,
            vendor_sev,
        )

    # ── Gate 2: no article content to analyse ─────────────────────────────
    if not headline:
        return _fallback("no headline_snippet", original_rule_score, vendor_sev)

    # ── Build subject context from input record ────────────────────────────
    subject_context = {
        "entity_type": str(inp.get("entity_type", "")),
        "country":     str(inp.get("country", inp.get("nationality", ""))),
        "dob":         str(inp.get("dob", "")),
        "role":        str(inp.get("role", inp.get("political_role", ""))),
    }

    subject_name = str(inp.get("name", ""))
    user_msg = _build_user_prompt(
        subject_name    = subject_name,
        subject_context = subject_context,
        headline        = headline,
        category        = category,
        source          = source,
        pub_date        = pub_date,
    )

    # ── Call LLM with graceful fallback on any failure ─────────────────────
    t0 = time.perf_counter()
    try:
        raw_response = _call_anthropic_api(user_msg)
        latency_ms   = round((time.perf_counter() - t0) * 1000, 1)
        analysis     = _parse_llm_response(raw_response, name_score, vendor_sev)
        analysis.latency_ms = latency_ms
        logger.info(
            "[ArticleAnalyser] %s → confirmed=%s conf=%.2f sev=%.2f→%.2f "
            "suppress=%s rels=%d latency=%.0fms",
            subject_name, analysis.entity_confirmed, analysis.entity_confidence,
            vendor_sev, analysis.blended_severity_score,
            analysis.should_suppress, len(analysis.relationships), latency_ms,
        )
        return analysis

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        reason = f"API error after {latency_ms:.0f}ms: {exc}"
        logger.warning("[ArticleAnalyser] Fallback — %s", reason)
        result = _fallback(reason, original_rule_score, vendor_sev)
        result.latency_ms = latency_ms
        return result


def analyse_batch(
    inp: dict,
    candidates: pd.DataFrame,
    name_scores: list[float],
    original_rule_scores: list[float],
    max_calls: int = 10,
) -> list[ArticleAnalysis]:
    """
    Analyse multiple article candidates for one input record.

    Only the top `max_calls` by name_score above ACTIVATION_THRESHOLD
    are sent to the LLM. The rest receive fallback results.

    Parameters
    ----------
    inp                   : Screening input dict
    candidates            : DataFrame of NNS_ARTICLES rows
    name_scores           : Parallel list of name_score per row
    original_rule_scores  : Parallel list of rule_score per row
    max_calls             : Maximum LLM calls per batch (cost control)

    Returns
    -------
    List[ArticleAnalysis] in the same order as candidates rows.
    """
    n = len(candidates)
    results: list[ArticleAnalysis | None] = [None] * n

    # Rank rows by name_score descending; only call LLM on top max_calls
    ranked = sorted(
        range(n),
        key=lambda i: name_scores[i],
        reverse=True,
    )

    calls_made = 0
    for idx in ranked:
        row  = candidates.iloc[idx]
        ns   = name_scores[idx]
        ors  = original_rule_scores[idx]

        if calls_made < max_calls and ns >= ACTIVATION_THRESHOLD:
            results[idx] = analyse_article(inp, row, ns, ors)
            calls_made += 1
        else:
            vendor_sev = float(row.get("category_severity", 0.50))
            results[idx] = _fallback(
                "below threshold or max_calls reached", ors, vendor_sev
            )

    return results  # type: ignore[return-value]
