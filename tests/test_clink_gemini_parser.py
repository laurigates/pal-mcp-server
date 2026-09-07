"""Tests for the Gemini CLI JSON parser."""

import pytest

from clink.parsers.gemini import GeminiJSONParser, GeminiUnsupportedClientError, ParserError


def _build_rate_limit_stdout() -> str:
    return (
        "{\n"
        '  "response": "",\n'
        '  "stats": {\n'
        '    "models": {\n'
        '      "gemini-2.5-pro": {\n'
        '        "api": {\n'
        '          "totalRequests": 5,\n'
        '          "totalErrors": 5,\n'
        '          "totalLatencyMs": 13319\n'
        "        },\n"
        '        "tokens": {"prompt": 0, "candidates": 0, "total": 0, "cached": 0, "thoughts": 0, "tool": 0}\n'
        "      }\n"
        "    },\n"
        '    "tools": {"totalCalls": 0},\n'
        '    "files": {"totalLinesAdded": 0, "totalLinesRemoved": 0}\n'
        "  }\n"
        "}"
    )


def test_gemini_parser_handles_rate_limit_empty_response():
    parser = GeminiJSONParser()
    stdout = _build_rate_limit_stdout()
    stderr = "Attempt 1 failed with status 429. Retrying with backoff... ApiError: quota exceeded"

    parsed = parser.parse(stdout, stderr)

    assert "429" in parsed.content
    assert parsed.metadata.get("rate_limit_status") == 429
    assert parsed.metadata.get("empty_response") is True
    assert "Attempt 1 failed" in parsed.metadata.get("stderr", "")


def test_gemini_parser_still_errors_when_no_fallback_available():
    parser = GeminiJSONParser()
    stdout = '{"response": "", "stats": {}}'

    with pytest.raises(ParserError):
        parser.parse(stdout, stderr="")


def _build_unsupported_client_stderr() -> str:
    """Approximate the Gemini CLI's rejection of an unsupported account tier."""

    return (
        "ApiError: Request failed with status 403. "
        '{"error":{"code":403,"status":"PERMISSION_DENIED","message":'
        '"UNSUPPORTED_CLIENT: This client is no longer supported for your account tier. '
        'Please migrate to the Antigravity suite: https://antigravity.google"}}'
    )


def test_gemini_parser_reports_unsupported_client_from_stderr():
    parser = GeminiJSONParser()

    with pytest.raises(GeminiUnsupportedClientError) as excinfo:
        parser.parse(stdout="", stderr=_build_unsupported_client_stderr())

    message = str(excinfo.value)
    assert "UNSUPPORTED_CLIENT" in message
    assert "antigravity.google" in message
    assert "GEMINI_API_KEY" in message
    assert "Gemini Code Assist" in message


def test_gemini_parser_reports_unsupported_client_from_non_json_stdout():
    parser = GeminiJSONParser()

    with pytest.raises(GeminiUnsupportedClientError):
        parser.parse(stdout=_build_unsupported_client_stderr(), stderr="")


def test_gemini_parser_reports_unsupported_client_from_json_error_payload():
    parser = GeminiJSONParser()
    stdout = '{"error": {"code": "UNSUPPORTED_CLIENT", "message": "migrate to the Antigravity suite"}}'

    with pytest.raises(GeminiUnsupportedClientError):
        parser.parse(stdout=stdout, stderr="")


def test_gemini_parser_still_parses_a_valid_response():
    parser = GeminiJSONParser()
    stdout = '{"response": "All good.", "stats": {"models": {"gemini-3-pro": {"tokens": {"total": 12}}}}}'

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "All good."
    assert parsed.metadata["model_used"] == "gemini-3-pro"


def test_gemini_parser_does_not_flag_a_response_that_merely_mentions_the_code():
    parser = GeminiJSONParser()
    stdout = '{"response": "The CLI fails with UNSUPPORTED_CLIENT on free-tier accounts."}'

    parsed = parser.parse(stdout=stdout, stderr="")

    assert "UNSUPPORTED_CLIENT" in parsed.content
