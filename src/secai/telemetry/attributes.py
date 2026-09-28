"""Canonical span names, attribute keys and score names.

Keeping these as constants means a typo is an import error rather than a
silently-missing attribute in Langfuse.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Root trace
# --------------------------------------------------------------------------
JOB_SPAN_PREFIX: Final = "job."


def job_span_name(job: str) -> str:
    """The root span name for a job run: ``job.<job_name>``."""
    return f"{JOB_SPAN_PREFIX}{job}"


# --------------------------------------------------------------------------
# secai.* span attributes (CLAUDE.md 6.4)
# --------------------------------------------------------------------------
ATTR_JOB: Final = "secai.job"
ATTR_DEAL_ID: Final = "secai.deal_id"
ATTR_ASSET_CLASS: Final = "secai.asset_class"
ATTR_DOC_TYPE: Final = "secai.doc_type"
ATTR_PROMPT_VERSION: Final = "secai.prompt_version"
ATTR_MODEL: Final = "secai.model"

# Langfuse's OTel ingestion stores only attributes it recognises and silently
# drops the rest, so every secai.* attribute is also written under this prefix,
# which Langfuse keeps as observation metadata ("secai.job" → metadata).
LANGFUSE_METADATA_PREFIX: Final = "langfuse.observation.metadata."

JOB_ATTRIBUTES: Final = (
    ATTR_JOB,
    ATTR_DEAL_ID,
    ATTR_ASSET_CLASS,
    ATTR_DOC_TYPE,
    ATTR_PROMPT_VERSION,
    ATTR_MODEL,
)

# Document parsing (JOB-03), set on the ``parse`` and ``chunk`` spans.
ATTR_DOC_SHA256: Final = "secai.doc.sha256"
ATTR_PARSE_PARSER: Final = "secai.parse.parser"
ATTR_PARSE_PAGE_COUNT: Final = "secai.parse.page_count"
ATTR_PARSE_TABLE_COUNT: Final = "secai.parse.table_count"
ATTR_PARSE_FALLBACK_TABLES: Final = "secai.parse.fallback_table_count"
ATTR_PARSE_BATCH_PAGES: Final = "secai.parse.batch_pages"
ATTR_PARSE_CACHED: Final = "secai.parse.cached"
ATTR_PARSE_DURATION_S: Final = "secai.parse.duration_s"
ATTR_CHUNK_SECTION_COUNT: Final = "secai.chunk.section_count"
ATTR_CHUNK_WATERFALL_COUNT: Final = "secai.chunk.waterfall_count"

# --------------------------------------------------------------------------
# Child span names (CLAUDE.md 6.4)
# --------------------------------------------------------------------------
SPAN_PARSE: Final = "parse"
SPAN_CHUNK: Final = "chunk"
SPAN_RETRIEVE: Final = "retrieve"
SPAN_LLM_EXTRACT: Final = "llm.extract"
SPAN_VALIDATE: Final = "validate"
SPAN_PERSIST: Final = "persist"

CHILD_SPANS: Final = (
    SPAN_PARSE,
    SPAN_CHUNK,
    SPAN_RETRIEVE,
    SPAN_LLM_EXTRACT,
    SPAN_VALIDATE,
    SPAN_PERSIST,
)

# --------------------------------------------------------------------------
# Langfuse score names (CLAUDE.md 6.4)
# --------------------------------------------------------------------------
SCORE_VALIDATION_PASS: Final = "validation_pass"
SCORE_FIELD_F1: Final = "field_f1"
SCORE_CRITICAL_FIELD_F1: Final = "critical_field_f1"
SCORE_FIELD_ACCURACY: Final = "field_accuracy"
SCORE_TOUCHLESS: Final = "touchless"
SCORE_RECON_BREAKS: Final = "recon_breaks"
SCORE_NEEDS_REVIEW: Final = "needs_review"
SCORE_LATENCY_S: Final = "latency_s"
SCORE_COST_TOKENS: Final = "cost_tokens"

BOOLEAN_SCORES: Final = (
    SCORE_VALIDATION_PASS,
    SCORE_TOUCHLESS,
    SCORE_NEEDS_REVIEW,
)
