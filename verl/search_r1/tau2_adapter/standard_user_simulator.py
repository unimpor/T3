from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, AzureOpenAI, InternalServerError, PermissionDeniedError, RateLimitError

from search_r1.tau2_adapter.action_parser import parse_action_string
from search_r1.tau2_adapter.core.environment import Environment
from search_r1.tau2_adapter.core.message import AssistantMessage, ToolMessage, UserMessage
from search_r1.tau2_adapter.core.tasks import Task


STOP_TOKEN = "###STOP###"
TRANSFER_TOKEN = "###TRANSFER###"
OUT_OF_SCOPE_TOKEN = "###OUT-OF-SCOPE###"
DEFAULT_FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"
DEFAULT_USER_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_USER_SEED = "300"
DEFAULT_USER_MAX_RETRIES = 3
DEFAULT_DISCONNECT_MESSAGE = "Sorry, I got disconnected for a moment. Could you repeat that?"

_OUTPUT_BLOCK_RE = re.compile(r"<(message|tool)>(.*?)</\1>", re.DOTALL)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tau2_guidelines_root() -> Path:
    override = os.getenv("TAU2_STANDARD_USER_GUIDELINES_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    bundled = Path(__file__).resolve().parent / "data" / "user_simulator"
    if bundled.exists():
        return bundled
    return _repo_root() / "tau2-bench" / "data" / "tau2" / "user_simulator"


def _load_guidelines(use_tools: bool) -> str:
    root = _tau2_guidelines_root()
    path = root / ("simulation_guidelines_tools.md" if use_tools else "simulation_guidelines.md")
    return path.read_text(encoding="utf-8")


def _format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    args = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in arguments.items())
    return f"{name}({args})"


def _default_logid() -> str:
    host = os.getenv("HOSTNAME", "local").strip() or "local"
    return f"tau2-standard-{host}-{os.getpid()}"


def _is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)):
        return True
    if isinstance(exc, PermissionDeniedError):
        text = str(exc)
        text_lower = text.lower()
        if "compliance gateway http" in text_lower and (
            "closed connection before returning the first response byte" in text_lower
            or "forward to backend error" in text_lower
            or "dial tcp" in text_lower
            or "lookup" in text_lower
            or "i/o timeout" in text_lower
            or "timeout" in text_lower
        ):
            return True
    return False


class Tau2StandardUserSimulator:
    def __init__(self, task: Task, env: Environment, controller: dict[str, Any]):
        self.task = task
        self.env = env
        self.controller = controller
        self.history: list[AssistantMessage | UserMessage | ToolMessage] = []
        self.max_tool_hops = int(controller.get("user_max_tool_hops", os.getenv("TAU2_STANDARD_USER_MAX_TOOL_HOPS", "8")))
        self.temperature = float(controller.get("user_temperature", os.getenv("TAU2_STANDARD_USER_TEMPERATURE", "0.0")))
        self.max_tokens = int(controller.get("user_max_tokens", os.getenv("TAU2_STANDARD_USER_MAX_TOKENS", "256")))
        self.max_retries = int(controller.get("user_max_retries", os.getenv("TAU2_STANDARD_USER_MAX_RETRIES", str(DEFAULT_USER_MAX_RETRIES))))
        self.retry_base_seconds = float(controller.get("user_retry_base_seconds", os.getenv("TAU2_STANDARD_USER_RETRY_BASE_SECONDS", "1.0")))
        self.disconnect_fallback = str(
            controller.get(
                "user_disconnect_fallback",
                os.getenv("TAU2_STANDARD_USER_DISCONNECT_FALLBACK", "true"),
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.disconnect_message = str(
            controller.get(
                "user_disconnect_message",
                os.getenv("TAU2_STANDARD_USER_DISCONNECT_MESSAGE", DEFAULT_DISCONNECT_MESSAGE),
            )
        ).strip() or DEFAULT_DISCONNECT_MESSAGE
        seed_raw = controller.get("user_seed", os.getenv("TAU2_STANDARD_USER_SEED", DEFAULT_USER_SEED))
        seed_str = str(seed_raw).strip()
        self.seed = int(seed_str) if seed_str else None
        self._client: AzureOpenAI | None = None

        try:
            user_tools = env.get_user_tools(include=task.user_tools) or []
        except Exception:
            user_tools = []
        self.user_tools = user_tools
        self.system_prompt = self._build_system_prompt()

    @staticmethod
    def stop_kind(message: UserMessage) -> str | None:
        if message.is_tool_call() or not message.has_text_content():
            return None
        content = message.content or ""
        if STOP_TOKEN in content:
            return "stop"
        if TRANSFER_TOKEN in content:
            return "transfer"
        if OUT_OF_SCOPE_TOKEN in content:
            return "out_of_scope"
        return None

    @staticmethod
    def is_stop(message: UserMessage) -> bool:
        return Tau2StandardUserSimulator.stop_kind(message) is not None

    def _build_system_prompt(self) -> str:
        guidelines = _load_guidelines(use_tools=bool(self.user_tools))
        guidelines = guidelines.replace("<PERSONA_GUIDELINES>", "")

        blocks = [
            guidelines.strip(),
            f"<scenario>\n{self.task.user_scenario}\n</scenario>",
        ]
        if self.user_tools:
            tools_desc = "\n\n".join(f"{idx + 1}. {tool.to_str()}" for idx, tool in enumerate(self.user_tools))
            blocks.append(f"""Available user tools:
{tools_desc}

Output format:
1. To send a message to the agent:
<message>Your natural reply to the agent</message>
2. To perform exactly one user-side tool action:
<tool>tool_name(arg1=value1, arg2=value2)</tool>

Rules:
- Each turn must contain exactly one <message> block or one <tool> block.
- Only use a user-side tool when the agent clearly asks you to do it or when the scenario requires it to answer honestly.
- Do not include plain English outside the XML block.
- Use valid Python-like keyword arguments inside <tool>.
""")
        else:
            blocks.append("""Output format:
1. Reply to the agent with:
<message>Your natural reply to the agent</message>

Rules:
- Reply with exactly one <message> block.
- Do not include plain English outside the XML block.
""")
        return "\n\n".join(blocks)

    def _get_client(self) -> AzureOpenAI:
        if self._client is not None:
            return self._client

        api_key = os.getenv("TAU2_STANDARD_USER_API_KEY", "").strip()
        api_version = os.getenv("TAU2_STANDARD_USER_API_VERSION", "2024-02-01").strip()
        azure_endpoint = os.getenv("TAU2_STANDARD_USER_AZURE_ENDPOINT", "").strip()
        model = os.getenv("TAU2_STANDARD_USER_MODEL", DEFAULT_USER_MODEL).strip()
        if not api_key or not azure_endpoint or not model:
            raise ValueError(
                "Standard Tau2 user simulator requires TAU2_STANDARD_USER_API_KEY, "
                "TAU2_STANDARD_USER_AZURE_ENDPOINT, and TAU2_STANDARD_USER_MODEL."
            )

        logid = os.getenv("TAU2_STANDARD_USER_LOGID", "").strip() or _default_logid()
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "api_version": api_version,
            "azure_endpoint": azure_endpoint,
            "default_headers": {"X-TT-LOGID": logid},
        }
        self._client = AzureOpenAI(**client_kwargs)
        return self._client

    def _api_model(self) -> str:
        model = os.getenv("TAU2_STANDARD_USER_MODEL", DEFAULT_USER_MODEL).strip()
        if not model:
            raise ValueError("TAU2_STANDARD_USER_MODEL is not set.")
        return model

    def _history_to_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for item in self.history:
            if isinstance(item, AssistantMessage):
                if item.has_text_content():
                    messages.append({"role": "user", "content": item.content or ""})
            elif isinstance(item, UserMessage):
                if item.is_tool_call():
                    tool_call = item.tool_calls[0]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"<tool>{_format_tool_call(tool_call.name, tool_call.arguments)}</tool>",
                        }
                    )
                elif item.has_text_content():
                    messages.append({"role": "assistant", "content": item.content or ""})
            elif isinstance(item, ToolMessage):
                messages.append({"role": "user", "content": f"Tool result: {item.content}"})
        return messages

    def _extract_action(self, raw_text: str) -> UserMessage:
        text = (raw_text or "").strip()
        match = _OUTPUT_BLOCK_RE.search(text)
        if match:
            block_type = match.group(1)
            content = match.group(2).strip()
            if block_type == "message":
                return UserMessage(role="user", content=content)
            parsed = parse_action_string(content, requestor="user")
            if not isinstance(parsed, UserMessage) or not parsed.is_tool_call() or len(parsed.tool_calls) != 1:
                raise ValueError("User simulator tool output must contain exactly one valid user tool call.")
            return parsed

        if text.startswith("<tool") and text.endswith(">"):
            parsed = parse_action_string(text, requestor="user")
            if isinstance(parsed, UserMessage) and parsed.is_tool_call():
                return parsed

        parsed = parse_action_string(text, requestor="user")
        if isinstance(parsed, UserMessage) and parsed.is_tool_call():
            return parsed
        return UserMessage(role="user", content=text)

    def _build_format_retry_prompt(self, error_text: str) -> str:
        if self.user_tools:
            return (
                "Your previous reply could not be parsed. Reply again using exactly one XML block: "
                "either <message>...</message> or <tool>tool_name(arg1=value1, arg2=value2)</tool>. "
                "If you use <tool>, it must contain exactly one valid user-side tool call and no extra text. "
                f"Parse error: {error_text}"
            )
        return (
            "Your previous reply could not be parsed. Reply again using exactly one "
            "<message>...</message> block and no extra text. "
            f"Parse error: {error_text}"
        )

    def _build_disconnect_message(
        self,
        error_text: str,
        *,
        kind: str = "runtime_error",
        raw_response: str | None = None,
    ) -> UserMessage:
        message = UserMessage(role="user", content=self.disconnect_message)
        message.raw_data = {
            "disconnect_fallback": True,
            "disconnect_kind": kind,
            "disconnect_error": error_text,
        }
        if raw_response is not None:
            message.raw_data["raw_response"] = raw_response
        self.history.append(message)
        return message

    def generate_next_message(self) -> UserMessage:
        client = self._get_client()
        request_kwargs_base: dict[str, Any] = {
            "model": self._api_model(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            request_kwargs_base["seed"] = self.seed
        last_error: Exception | None = None
        repair_prompt: str | None = None
        for attempt in range(1, max(1, self.max_retries) + 1):
            request_kwargs = dict(request_kwargs_base)
            messages = self._history_to_messages()
            if repair_prompt:
                messages.append({"role": "user", "content": repair_prompt})
            request_kwargs["messages"] = messages
            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                last_error = exc
                retryable = _is_retryable_api_error(exc)
                if attempt >= max(1, self.max_retries) or not retryable:
                    return self._build_disconnect_message(str(exc), kind="api_error")
                time.sleep(self.retry_base_seconds * (2 ** (attempt - 1)))
                continue

            raw = ""
            try:
                raw = response.choices[0].message.content or ""
                user_message = self._extract_action(raw)
            except Exception as exc:
                last_error = exc
                repair_prompt = self._build_format_retry_prompt(str(exc))
                if attempt >= max(1, self.max_retries):
                    return self._build_disconnect_message(
                        str(exc),
                        kind="format_error",
                        raw_response=raw,
                    )
                continue

            user_message.raw_data = {"raw_response": raw}
            self.history.append(user_message)
            return user_message

        return self._build_disconnect_message(str(last_error) if last_error is not None else "unknown_error")

    def append_assistant_message(self, message: AssistantMessage) -> None:
        self.history.append(message)

    def append_tool_result(self, message: ToolMessage) -> None:
        self.history.append(message)
