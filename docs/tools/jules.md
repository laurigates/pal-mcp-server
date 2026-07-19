# Jules Tool - Async Coding Agent

**Delegate a coding task to Google Jules and poll for the resulting pull request**

The `jules` tool drives [Google Jules](https://developers.google.com/jules/api), an
asynchronous AI coding agent. You point Jules at a connected GitHub repository with a
prompt; it plans, writes code, and (by default) opens a pull request on its own in the
background. Unlike a chat model, Jules is not a request/response completion — so it
lives as a dedicated tool rather than a model provider.

The tool is **action-based and non-blocking**: each call performs one operation and
returns immediately. The calling CLI drives the poll loop.

> **CAUTION**: With the default `automation_mode="AUTO_CREATE_PR"`, Jules writes code
> and opens pull requests in your repository autonomously. Point it only at repos you
> intend it to modify. Set `require_plan_approval=true` to gate execution behind an
> explicit `approve` step.

## Setup

1. Install the Jules GitHub app and connect your repositories at
   [jules.google.com](https://jules.google.com).
2. Create an API key at
   [jules.google.com/settings#api](https://jules.google.com/settings#api) (max 3 keys).
3. Set `JULES_API_KEY` in your `.env` / MCP environment.

Note: the Jules API is in **alpha** (`v1alpha`), and API sessions draw from your
account's task quota (e.g. 15 tasks/day on the free tier).

## Actions

| `action` | Required params | Purpose |
|---|---|---|
| `list_sources` | — | Discover connected repos (`sources/github/{owner}/{repo}`). |
| `create` | `prompt`, `source` | Start a session. Optional: `starting_branch`, `title`, `require_plan_approval`, `automation_mode`. Returns `session_id`, `state`, `url`. |
| `status` | `session_id` | Poll `state`, recent `activities`, and `outputs` (the PR URL). Optional `page_size`. |
| `message` | `session_id`, `prompt` | Send steering feedback to a running session. |
| `approve` | `session_id` | Approve a pending plan (when created with `require_plan_approval=true`). |

## Typical workflow

```
jules action=list_sources
jules action=create source="sources/github/me/app" prompt="Add a /health endpoint"
jules action=status session_id="<id>"    # repeat until state is COMPLETED or FAILED
```

Poll `status` periodically (don't busy-loop). When `state` reaches `COMPLETED`, the
pull request URL appears in `outputs[].pullRequest.url`.
