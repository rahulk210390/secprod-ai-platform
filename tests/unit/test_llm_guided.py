"""Guided decoding and prompt resolution (JOB-02)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from secai.llm.errors import LLMEmptyResponseError, LLMSchemaError
from secai.llm.guided import json_schema_for, parse_structured, response_format_for
from secai.llm.prompts import Prompt, get_prompt, load_local, local_prompt_path


class Trigger(BaseModel):
    trigger_type: str
    metric: str
    threshold: float
    consequence: str | None = None


class DealTerms(BaseModel):
    deal_name: str
    closing_date: str
    triggers: list[Trigger] = Field(default_factory=list)


class TestSchemaGeneration:
    def test_schema_describes_the_model(self) -> None:
        schema = json_schema_for(DealTerms)
        assert schema["properties"]["deal_name"]["type"] == "string"
        assert "triggers" in schema["properties"]

    def test_required_fields_are_marked(self) -> None:
        schema = json_schema_for(DealTerms)
        assert set(schema["required"]) == {"deal_name", "closing_date"}

    def test_response_format_carries_the_schema(self) -> None:
        fmt = response_format_for(DealTerms)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "DealTerms"
        assert fmt["json_schema"]["schema"] == json_schema_for(DealTerms)
        assert fmt["json_schema"]["strict"] is True


class TestParsing:
    def test_valid_payload_returns_the_model(self) -> None:
        result = parse_structured(
            '{"deal_name": "SYNTH 2024-1", "closing_date": "2024-03-15"}', DealTerms
        )
        assert result.deal_name == "SYNTH 2024-1"
        assert result.triggers == []

    def test_nested_objects_parse(self) -> None:
        payload = (
            '{"deal_name": "SYNTH 2024-1", "closing_date": "2024-03-15",'
            ' "triggers": [{"trigger_type": "OC", "metric": "oc_ratio",'
            ' "threshold": 115.5, "consequence": "divert"}]}'
        )
        result = parse_structured(payload, DealTerms)
        assert result.triggers[0].threshold == 115.5

    @pytest.mark.parametrize("content", ["", "   ", None])
    def test_empty_content_raises(self, content: str | None) -> None:
        with pytest.raises(LLMEmptyResponseError):
            parse_structured(content, DealTerms)

    def test_non_json_raises_with_the_raw_output(self) -> None:
        with pytest.raises(LLMSchemaError) as caught:
            parse_structured("the deal is called SYNTH", DealTerms)
        assert caught.value.raw_output == "the deal is called SYNTH"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(LLMSchemaError, match="did not satisfy DealTerms"):
            parse_structured('{"deal_name": "SYNTH 2024-1"}', DealTerms)

    def test_wrong_type_raises(self) -> None:
        # A threshold arriving as prose is exactly what guided decoding prevents.
        payload = (
            '{"deal_name": "S", "closing_date": "2024-03-15",'
            ' "triggers": [{"trigger_type": "OC", "metric": "oc",'
            ' "threshold": "one hundred", "consequence": null}]}'
        )
        with pytest.raises(LLMSchemaError):
            parse_structured(payload, DealTerms)


class TestPrompts:
    def test_local_prompts_exist_for_the_shipped_templates(self) -> None:
        for name in ("deal_terms_extraction", "deal_qa"):
            assert local_prompt_path(name).is_file()

    def test_load_local_returns_a_versioned_prompt(self) -> None:
        prompt = load_local("deal_terms_extraction")
        assert prompt.source == "local"
        assert prompt.version.startswith("local-")
        assert "Never compute" in prompt.template

    def test_version_changes_with_content(self) -> None:
        a = Prompt(name="p", template="one", version="", source="local")
        first = load_local("deal_terms_extraction").version
        second = load_local("deal_qa").version
        assert first != second
        assert a.template == "one"

    def test_missing_local_prompt_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="no local fallback"):
            load_local("does_not_exist")

    def test_render_fills_placeholders(self) -> None:
        prompt = Prompt(name="p", template="deal {name} page {page}", version="v1", source="local")
        assert prompt.render(name="SYNTH", page=12) == "deal SYNTH page 12"

    def test_render_fails_loudly_on_a_missing_value(self) -> None:
        # A half-rendered prompt reaching the model is worse than an exception.
        prompt = Prompt(name="p", template="deal {name}", version="v1", source="local")
        with pytest.raises(KeyError, match="needs a value"):
            prompt.render(wrong="x")

    def test_falls_back_to_local_when_langfuse_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Broken:
            def get_prompt(self, *args: object, **kwargs: object) -> object:
                raise RuntimeError("langfuse down")

        monkeypatch.setattr("secai.llm.prompts.get_client", Broken)
        prompt = get_prompt("deal_terms_extraction")
        # A Langfuse outage must not take extraction down with it.
        assert prompt.source == "local"

    def test_langfuse_prompt_wins_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Fetched:
            prompt = "remote template"
            version = 7

        class Working:
            def get_prompt(self, *args: object, **kwargs: object) -> Fetched:
                return Fetched()

        monkeypatch.setattr("secai.llm.prompts.get_client", Working)
        prompt = get_prompt("deal_terms_extraction")
        assert prompt.source == "langfuse"
        assert prompt.version == "7"

    def test_use_langfuse_false_skips_the_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode() -> object:
            raise AssertionError("Langfuse must not be consulted")

        monkeypatch.setattr("secai.llm.prompts.get_client", explode)
        assert get_prompt("deal_qa", use_langfuse=False).source == "local"
