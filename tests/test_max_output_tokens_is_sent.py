"""
The registry's output ceiling reaches the provider request (issue #114).

`max_output_tokens` was plumbed through every provider and declared per model in
`conf/*_models.json`, but no tool ever passed it — `grep -rn max_output_tokens
tools/` returned nothing — so each provider's own default governed reply length
instead of the ceiling our own registry advertises. The four call sites now pass
it, sourced from the already-resolved ModelContext rather than a fresh lookup.

Two things need pinning, and the second is the sharper one:

* the ceiling is present in the outgoing request for a model that declares one;
* a model whose registry entry declares 0 sends no ceiling at all.
  `ModelCapabilities.max_output_tokens` defaults to 0 and many entries never set
  it, so a naive read would put `max_tokens: 0` on the wire and ask the provider
  for an empty completion. That is the failure mode the issue names.

Wiring the callers also exposed a latent bug in the provider itself: the o3
Responses-API branch had been building `max_completion_tokens`, which is the
Chat Completions name and is rejected by `client.responses.create`. The branch
had never executed, because nothing had ever supplied a ceiling to reach it.
`test_responses_endpoint_uses_the_responses_api_parameter_name` pins the fix.
"""

import pytest

from providers.gemini import GeminiModelProvider
from providers.openai import OpenAIModelProvider
from providers.shared.model_capabilities import ModelCapabilities
from providers.shared.provider_type import ProviderType
from tools.shared.exceptions import ToolExecutionError


def _stub_capabilities(max_output_tokens: int) -> ModelCapabilities:
    return ModelCapabilities(
        provider=ProviderType.OPENAI,
        model_name="stub",
        friendly_name="Stub",
        max_output_tokens=max_output_tokens,
    )


class TestZeroMeansUnset:
    """A declared 0 must never reach the wire as a ceiling."""

    def test_zero_capability_reports_no_ceiling(self):
        caps = _stub_capabilities(0)
        assert caps.get_effective_max_output_tokens() is None

    def test_negative_capability_reports_no_ceiling(self):
        caps = _stub_capabilities(-1)
        assert caps.get_effective_max_output_tokens() is None

    def test_declared_capability_is_returned(self):
        caps = _stub_capabilities(4096)
        assert caps.get_effective_max_output_tokens() == 4096


class TestOpenAIChatCompletions:
    def _build(self, model_name, max_output_tokens):
        provider = OpenAIModelProvider(api_key="dummy-key-for-tests")
        return provider._build_request(
            prompt="hello",
            model_name=model_name,
            system_prompt=None,
            temperature=0.2,
            max_output_tokens=max_output_tokens,
        )

    def test_declared_ceiling_is_sent_as_max_tokens(self):
        request = self._build("gpt-5", 128_000)
        assert request["endpoint"] == "chat"
        assert request["params"]["max_tokens"] == 128_000

    def test_no_ceiling_omits_the_field_entirely(self):
        """Not `max_tokens: 0`, and not `max_tokens: None` — absent."""
        request = self._build("gpt-5", None)
        assert "max_tokens" not in request["params"]

    def test_zero_is_treated_as_unset_end_to_end(self):
        request = self._build("gpt-5", 0)
        assert "max_tokens" not in request["params"]


class TestOpenAIResponsesEndpoint:
    """o3 models go to client.responses.create, which names the parameter differently."""

    def _build(self, max_output_tokens):
        provider = OpenAIModelProvider(api_key="dummy-key-for-tests")
        return provider._build_request(
            prompt="hello",
            model_name="o3-pro",
            system_prompt=None,
            temperature=1.0,
            max_output_tokens=max_output_tokens,
        )

    def test_responses_endpoint_uses_the_responses_api_parameter_name(self):
        """max_completion_tokens is a Chat Completions name; responses.create rejects it."""
        request = self._build(100_000)
        assert request["endpoint"] == "responses"
        assert request["params"]["max_output_tokens"] == 100_000
        assert "max_completion_tokens" not in request["params"]

    def test_responses_parameter_is_accepted_by_the_installed_sdk(self):
        """Guards against the rename drifting back: ask the SDK, do not assume."""
        import inspect

        from openai.resources.responses import AsyncResponses

        accepted = set(inspect.signature(AsyncResponses.create).parameters)
        assert "max_output_tokens" in accepted
        assert "max_completion_tokens" not in accepted

    def test_no_ceiling_omits_the_field(self):
        request = self._build(None)
        assert "max_output_tokens" not in request["params"]


class TestGemini:
    def _build(self, max_output_tokens):
        provider = GeminiModelProvider(api_key="dummy-key-for-tests")
        return provider._build_request(
            prompt="hello",
            model_name="gemini-2.5-flash",
            system_prompt=None,
            temperature=0.2,
            max_output_tokens=max_output_tokens,
        )

    def test_declared_ceiling_reaches_generation_config(self):
        request = self._build(65_536)
        assert request["config"].max_output_tokens == 65_536

    def test_no_ceiling_leaves_generation_config_unset(self):
        request = self._build(None)
        assert request["config"].max_output_tokens is None


class TestToolsActuallyPassIt:
    """The provider plumbing was always there; the callers were what never used it.

    This is the half issue #114 is really about — `grep -rn max_output_tokens
    tools/` returned nothing before this change — so it is asserted at the tool
    boundary rather than inferred from the provider tests above.
    """

    @pytest.mark.asyncio
    async def test_chat_tool_passes_the_registry_ceiling_to_the_provider(self, monkeypatch, tmp_path):
        from tools.chat import ChatTool

        captured = {}

        async def _capture(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after capturing the outgoing request")

        monkeypatch.setattr("providers.gemini.GeminiModelProvider.generate_content", _capture, raising=True)

        # The stub raises to stop the call once the outgoing kwargs are captured;
        # SimpleTool.execute wraps any provider failure in ToolExecutionError.
        with pytest.raises(ToolExecutionError):
            await ChatTool().execute(
                {
                    "prompt": "hello",
                    "model": "gemini-2.5-flash",
                    "working_directory_absolute_path": str(tmp_path),
                }
            )

        assert "max_output_tokens" in captured, "the chat tool sent no output ceiling at all"
        expected = GeminiModelProvider.MODEL_CAPABILITIES["gemini-2.5-flash"].get_effective_max_output_tokens()
        assert captured["max_output_tokens"] == expected


class TestRegistryValuesAreActuallyDeclared:
    """The wiring is only useful if the registry states ceilings worth sending."""

    @pytest.mark.parametrize(
        ("provider_cls", "model_name"),
        [
            (OpenAIModelProvider, "gpt-5"),
            (OpenAIModelProvider, "o3-pro"),
            (GeminiModelProvider, "gemini-2.5-flash"),
            (GeminiModelProvider, "gemini-2.5-pro"),
        ],
    )
    def test_known_models_declare_a_positive_ceiling(self, provider_cls, model_name):
        capabilities = provider_cls.MODEL_CAPABILITIES[model_name]
        ceiling = capabilities.get_effective_max_output_tokens()
        assert ceiling is not None and ceiling > 0, (
            f"{model_name} declares max_output_tokens={capabilities.max_output_tokens}; "
            "the tools would send no ceiling for it."
        )
