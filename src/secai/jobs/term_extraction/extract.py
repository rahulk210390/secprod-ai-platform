"""The LLM calls: one per context, each constrained to a raw schema.

Every call goes through :class:`LLMClient.generate_structured`, so it is a
Langfuse generation with the model, token usage and prompt version, and its
output is a validated Pydantic object, never free text.

A schema failure on one context is recorded and the run continues: one
unreadable excerpt should cost that excerpt's facts, not the whole deal. A
transport failure (vLLM down) is not recoverable here and propagates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from secai.config import Settings, get_settings
from secai.jobs.term_extraction.assemble import Extraction
from secai.jobs.term_extraction.context import Context, Target
from secai.llm import LLMClient, LLMResponseError, Message, Prompt, get_prompt
from secai.schemas.deal_terms import RawCoverageTests, RawDealFacts, RawTranches, RawTriggers
from secai.validation.deal_terms import Issue

SYSTEM_PROMPT: Final = (
    "You extract facts from securitisation prospectuses. Copy every value exactly as it "
    "is printed in the excerpt. Never calculate, convert, round or infer a value. If "
    "something is not in the excerpt, use null or an empty list."
)
PROMPT_NAMES: Final = {
    Target.FACTS: "term_extraction_facts",
    Target.TRANCHES: "term_extraction_tranches",
    Target.TRIGGERS: "term_extraction_triggers",
    Target.COVERAGE: "term_extraction_coverage",
}
SCHEMAS: Final[dict[Target, type[BaseModel]]] = {
    Target.FACTS: RawDealFacts,
    Target.TRANCHES: RawTranches,
    Target.TRIGGERS: RawTriggers,
    Target.COVERAGE: RawCoverageTests,
}


@dataclass
class TermExtractor:
    client: LLMClient
    prompts: dict[Target, Prompt]
    model: str

    @classmethod
    def create(cls, client: LLMClient, settings: Settings | None = None) -> TermExtractor:
        resolved = settings or get_settings()
        prompts = {target: get_prompt(name) for target, name in PROMPT_NAMES.items()}
        return cls(client=client, prompts=prompts, model=resolved.vllm.model)

    @property
    def prompt_version(self) -> str:
        """One version for the prompt set: any edit to any prompt changes it."""
        joined = "|".join(f"{t.value}:{self.prompts[t].version}" for t in Target)
        return "set-" + hashlib.sha256(joined.encode()).hexdigest()[:12]

    def prompt_chars(self, target: Target) -> int:
        """Characters the prompt adds around a context, for sizing contexts."""
        return len(SYSTEM_PROMPT) + len(self.prompts[target].template)

    def _call(self, target: Target, context: Context, doc_name: str) -> BaseModel:
        prompt = self.prompts[target]
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.render(doc_name=doc_name, context=context.text)},
        ]
        return self.client.generate_structured(
            messages,
            SCHEMAS[target],
            prompt_version=prompt.version,
        )

    def run(
        self, contexts: dict[Target, list[Context]], doc_name: str
    ) -> tuple[Extraction, list[Issue]]:
        extraction = Extraction()
        issues: list[Issue] = []
        for target in Target:
            for context in contexts.get(target, []):
                try:
                    result = self._call(target, context, doc_name)
                except LLMResponseError as exc:
                    issues.append(
                        Issue(code="llm_schema_error", field=target.value, message=str(exc)[:300])
                    )
                    continue
                if isinstance(result, RawDealFacts):
                    extraction.facts.append((result, context.block_indices))
                elif isinstance(result, RawTranches):
                    extraction.tranches, extraction.tranches_blocks = result, context.block_indices
                elif isinstance(result, RawTriggers):
                    extraction.triggers.append((result, context.block_indices))
                elif isinstance(result, RawCoverageTests):
                    extraction.coverage.append((result, context.block_indices))
        return extraction, issues


# --------------------------------------------------------------------------
# extraction cache: same document, prompts and model → same raw output
# --------------------------------------------------------------------------
def cache_key(doc_sha256: str, prompt_version: str, model: str) -> str:
    return hashlib.sha256(f"{doc_sha256}|{prompt_version}|{model}".encode()).hexdigest()[:24]


def save_extraction(path: Path, extraction: Extraction, issues: list[Issue]) -> None:
    payload = {
        "facts": [[r.model_dump(), b] for r, b in extraction.facts],
        "tranches": extraction.tranches.model_dump() if extraction.tranches else None,
        "tranches_blocks": extraction.tranches_blocks,
        "triggers": [[r.model_dump(), b] for r, b in extraction.triggers],
        "coverage": [[r.model_dump(), b] for r, b in extraction.coverage],
        "issues": [i.model_dump(mode="json") for i in issues],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_extraction(path: Path) -> tuple[Extraction, list[Issue]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    extraction = Extraction(
        facts=[(RawDealFacts.model_validate(r), b) for r, b in raw["facts"]],
        tranches=RawTranches.model_validate(raw["tranches"]) if raw["tranches"] else None,
        tranches_blocks=raw["tranches_blocks"],
        triggers=[(RawTriggers.model_validate(r), b) for r, b in raw["triggers"]],
        coverage=[(RawCoverageTests.model_validate(r), b) for r, b in raw["coverage"]],
    )
    return extraction, [Issue.model_validate(i) for i in raw["issues"]]
