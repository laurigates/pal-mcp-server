"""jules tool - drive Google Jules, an asynchronous AI coding agent.

Jules (https://developers.google.com/jules/api) is an async coding agent: you
create a *session* against a connected GitHub repository, it plans and writes
code in the background, and you *poll activities* for progress until it opens a
pull request. This does not fit PAL's synchronous ``ModelProvider`` chat
contract, so Jules lives as a self-executing tool modelled on ``clink``:
``requires_model()`` is ``False`` and ``execute()`` is overridden to talk to the
Jules REST API directly.

The tool is action-based and non-blocking. Each call performs a single operation
(``list_sources``, ``create``, ``status``, ``message``, ``approve``) and returns
immediately; the calling CLI drives the poll loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, NoReturn

import httpx
from mcp.types import TextContent
from pydantic import BaseModel, Field

from tools.models import ToolModelCategory, ToolOutput
from tools.shared.exceptions import ToolExecutionError
from tools.simple.base import SimpleTool
from utils.env import get_env
from utils.progress import get_progress_reporter

logger = logging.getLogger(__name__)

JULES_API_BASE = "https://jules.googleapis.com/v1alpha"
JULES_API_KEY_ENV = "JULES_API_KEY"
JULES_API_KEY_PLACEHOLDER = "your_jules_api_key_here"
JULES_REQUEST_TIMEOUT = 60.0

VALID_ACTIONS = ("list_sources", "create", "status", "message", "approve")
AUTOMATION_MODES = ("AUTO_CREATE_PR", "AUTOMATION_MODE_UNSPECIFIED")
DEFAULT_AUTOMATION_MODE = "AUTO_CREATE_PR"

# Activity union keys (Jules REST) mapped to a short label + the field holding text.
_ACTIVITY_KINDS = {
    "agentMessaged": ("agent_message", "agentMessage"),
    "userMessaged": ("user_message", "userMessage"),
    "planGenerated": ("plan_generated", None),
    "planApproved": ("plan_approved", None),
    "progressUpdated": ("progress", "description"),
    "sessionCompleted": ("session_completed", None),
    "sessionFailed": ("session_failed", "reason"),
}
MAX_ACTIVITIES_RETURNED = 30


class JulesRequest(BaseModel):
    """Request model for the jules tool."""

    action: str = Field(..., description=f"Operation to perform. One of: {', '.join(VALID_ACTIONS)}.")
    prompt: str | None = Field(
        default=None,
        description="Task for Jules (action=create) or steering feedback (action=message).",
    )
    source: str | None = Field(
        default=None,
        description="Source resource name for action=create, e.g. 'sources/github/{owner}/{repo}' (from list_sources).",
    )
    starting_branch: str | None = Field(
        default=None,
        description="Base branch for the session (action=create). Defaults to the repository default branch.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session id for status/message/approve. Accepts either the raw id or 'sessions/{id}'.",
    )
    title: str | None = Field(
        default=None,
        description="Optional human-readable session title (action=create).",
    )
    require_plan_approval: bool = Field(
        default=False,
        description="If true, the session pauses at AWAITING_PLAN_APPROVAL until action=approve (action=create).",
    )
    automation_mode: str = Field(
        default=DEFAULT_AUTOMATION_MODE,
        description="Automation mode for action=create. AUTO_CREATE_PR lets Jules open a pull request automatically.",
    )
    page_size: int = Field(
        default=MAX_ACTIVITIES_RETURNED,
        description="Maximum number of recent activities to return for action=status (1-100).",
    )


class JulesTool(SimpleTool):
    """Drive the Google Jules async coding agent over its REST API."""

    def get_name(self) -> str:
        return "jules"

    def get_description(self) -> str:
        return (
            "Drive Google Jules, an asynchronous AI coding agent that works on a connected GitHub repo. "
            "Action-based and non-blocking: list_sources, create (start a session), status (poll progress + "
            "get the PR), message (steer), approve (approve a plan). Requires JULES_API_KEY."
        )

    def get_annotations(self) -> dict[str, Any] | None:
        # Not read-only: creating a session mutates a repository and can open a pull request.
        return {"readOnlyHint": False}

    def requires_model(self) -> bool:
        return False

    def get_model_category(self) -> ToolModelCategory:
        return ToolModelCategory.BALANCED

    def get_system_prompt(self) -> str:
        from systemprompts import JULES_PROMPT

        return JULES_PROMPT

    def get_request_model(self):
        return JulesRequest

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        """Unused: jules builds its schema end-to-end in get_input_schema()."""
        return {}

    def get_input_schema(self) -> dict[str, Any]:
        properties = {
            "action": {
                "type": "string",
                "enum": list(VALID_ACTIONS),
                "description": (
                    "Operation to perform: 'list_sources' (discover repos), 'create' (start a session), "
                    "'status' (poll state + activities + PR output), 'message' (steer a running session), "
                    "'approve' (approve a pending plan)."
                ),
            },
            "prompt": {
                "type": "string",
                "description": "Task for Jules (create) or steering feedback (message).",
            },
            "source": {
                "type": "string",
                "description": "Source resource name for create, e.g. 'sources/github/{owner}/{repo}'.",
            },
            "starting_branch": {
                "type": "string",
                "description": "Base branch for the session (create). Defaults to the repository default branch.",
            },
            "session_id": {
                "type": "string",
                "description": "Session id for status/message/approve (raw id or 'sessions/{id}').",
            },
            "title": {
                "type": "string",
                "description": "Optional session title (create).",
            },
            "require_plan_approval": {
                "type": "boolean",
                "description": "If true, the session waits for action=approve before executing (create).",
            },
            "automation_mode": {
                "type": "string",
                "enum": list(AUTOMATION_MODES),
                "description": "Automation mode for create. AUTO_CREATE_PR opens a pull request automatically.",
            },
            "page_size": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Max recent activities to return for status.",
            },
        }
        return {
            "type": "object",
            "properties": properties,
            "required": ["action"],
            "additionalProperties": False,
        }

    # ------------------------------------------------------------------ helpers

    def _raise_tool_error(self, message: str, metadata: dict[str, Any] | None = None) -> NoReturn:
        error_output = ToolOutput(status="error", content=message, content_type="text", metadata=metadata)
        raise ToolExecutionError(error_output.model_dump_json())

    def _get_api_key(self) -> str:
        api_key = get_env(JULES_API_KEY_ENV)
        if not api_key or api_key == JULES_API_KEY_PLACEHOLDER:
            self._raise_tool_error(
                f"{JULES_API_KEY_ENV} is not configured. Create a key at "
                "https://jules.google.com/settings#api and set it in your environment."
            )
        return api_key

    def _build_client(self, api_key: str) -> httpx.AsyncClient:
        """Construct the Jules HTTP client. Overridable in tests."""
        return httpx.AsyncClient(
            base_url=JULES_API_BASE,
            headers={"X-Goog-Api-Key": api_key, "Content-Type": "application/json"},
            timeout=JULES_REQUEST_TIMEOUT,
        )

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        return session_id.split("/", 1)[1] if session_id.startswith("sessions/") else session_id

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await client.request(method, path, json=json_body, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            self._raise_tool_error(
                f"Jules API request failed ({exc.response.status_code}) for {method} {path}: {body}",
                metadata={"status_code": exc.response.status_code},
            )
        except httpx.HTTPError as exc:
            self._raise_tool_error(f"Jules API request error for {method} {path}: {exc}")
        if not response.content:
            return {}
        return response.json()

    @staticmethod
    def _summarize_activity(activity: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "originator": activity.get("originator"),
            "createTime": activity.get("createTime"),
        }
        kind = "unknown"
        text: str | None = activity.get("description")
        for key, (label, text_field) in _ACTIVITY_KINDS.items():
            if key in activity:
                kind = label
                payload = activity.get(key)
                if text_field and isinstance(payload, dict):
                    text = payload.get(text_field) or text
                elif key == "planGenerated" and isinstance(payload, dict):
                    steps = payload.get("plan", {}).get("steps") or payload.get("steps")
                    if steps:
                        text = f"{len(steps)} plan step(s)"
                break
        summary["kind"] = kind
        if text:
            summary["text"] = text
        return summary

    # ------------------------------------------------------------------ actions

    async def _do_list_sources(self, client: httpx.AsyncClient) -> ToolOutput:
        data = await self._request(client, "GET", "/sources")
        sources = [
            {
                "name": src.get("name"),
                "id": src.get("id"),
                "githubRepo": src.get("githubRepo"),
            }
            for src in data.get("sources", [])
        ]
        return ToolOutput(
            status="success",
            content_type="json",
            content=self._json(sources),
            metadata={"action": "list_sources", "count": len(sources)},
        )

    async def _do_create(self, client: httpx.AsyncClient, request: JulesRequest) -> ToolOutput:
        if not request.prompt:
            self._raise_tool_error("action=create requires 'prompt'.")
        if not request.source:
            self._raise_tool_error("action=create requires 'source' (see action=list_sources).")

        github_context: dict[str, Any] = {}
        if request.starting_branch:
            github_context["startingBranch"] = request.starting_branch
        source_context: dict[str, Any] = {"source": request.source}
        if github_context:
            source_context["githubRepoContext"] = github_context

        body: dict[str, Any] = {
            "prompt": request.prompt,
            "sourceContext": source_context,
            "automationMode": request.automation_mode,
        }
        if request.require_plan_approval:
            body["requirePlanApproval"] = True
        if request.title:
            body["title"] = request.title

        data = await self._request(client, "POST", "/sessions", json_body=body)
        summary = {
            "session_id": data.get("id"),
            "name": data.get("name"),
            "state": data.get("state"),
            "url": data.get("url"),
            "title": data.get("title"),
        }
        return ToolOutput(
            status="success",
            content_type="json",
            content=self._json(summary),
            metadata={"action": "create"},
        )

    async def _do_status(self, client: httpx.AsyncClient, request: JulesRequest) -> ToolOutput:
        if not request.session_id:
            self._raise_tool_error("action=status requires 'session_id'.")
        sid = self._normalize_session_id(request.session_id)
        page_size = max(1, min(request.page_size or MAX_ACTIVITIES_RETURNED, 100))

        session = await self._request(client, "GET", f"/sessions/{sid}")
        activities_data = await self._request(
            client, "GET", f"/sessions/{sid}/activities", params={"pageSize": page_size}
        )
        activities = [self._summarize_activity(a) for a in activities_data.get("activities", [])]

        result = {
            "session_id": session.get("id"),
            "state": session.get("state"),
            "url": session.get("url"),
            "outputs": session.get("outputs", []),
            "activities": activities,
        }
        return ToolOutput(
            status="success",
            content_type="json",
            content=self._json(result),
            metadata={"action": "status", "state": session.get("state")},
        )

    async def _do_message(self, client: httpx.AsyncClient, request: JulesRequest) -> ToolOutput:
        if not request.session_id:
            self._raise_tool_error("action=message requires 'session_id'.")
        if not request.prompt:
            self._raise_tool_error("action=message requires 'prompt'.")
        sid = self._normalize_session_id(request.session_id)
        await self._request(client, "POST", f"/sessions/{sid}:sendMessage", json_body={"prompt": request.prompt})
        return ToolOutput(
            status="success",
            content="Message sent to Jules session.",
            metadata={"action": "message", "session_id": sid},
        )

    async def _do_approve(self, client: httpx.AsyncClient, request: JulesRequest) -> ToolOutput:
        if not request.session_id:
            self._raise_tool_error("action=approve requires 'session_id'.")
        sid = self._normalize_session_id(request.session_id)
        await self._request(client, "POST", f"/sessions/{sid}:approvePlan")
        return ToolOutput(
            status="success",
            content="Plan approved for Jules session.",
            metadata={"action": "approve", "session_id": sid},
        )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ execute

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            request = self.get_request_model()(**arguments)
        except Exception as exc:
            self._raise_tool_error(f"Invalid jules arguments: {exc}")

        if request.action not in VALID_ACTIONS:
            self._raise_tool_error(f"Unknown action '{request.action}'. Valid actions: {', '.join(VALID_ACTIONS)}.")

        api_key = self._get_api_key()
        dispatch = {
            "list_sources": lambda c: self._do_list_sources(c),
            "create": lambda c: self._do_create(c, request),
            "status": lambda c: self._do_status(c, request),
            "message": lambda c: self._do_message(c, request),
            "approve": lambda c: self._do_approve(c, request),
        }

        reporter = get_progress_reporter()
        client = self._build_client(api_key)
        try:
            async with reporter.heartbeat(f"jules · {request.action} · running"):
                tool_output = await dispatch[request.action](client)
        finally:
            await client.aclose()

        return [TextContent(type="text", text=tool_output.model_dump_json())]

    async def prepare_prompt(self, request) -> str:  # pragma: no cover - execute() is overridden
        return ""
