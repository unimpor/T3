from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from search_r1.tau2_adapter.action_parser import parse_action_string
from search_r1.tau2_adapter.core.message import AssistantMessage, ToolMessage, UserMessage
from search_r1.tau2_adapter.core.toolkit import ToolType
from search_r1.tau2_adapter.eval.action import extract_tool_calls
from search_r1.tau2_adapter.eval.evaluator import evaluate_task
from search_r1.tau2_adapter.eval.schema import RewardInfo
from search_r1.tau2_adapter.loader.registry import get_env_constructor
from search_r1.tau2_adapter.loader.tasks import get_tasks
from search_r1.tau2_adapter.standard_user_simulator import (
    DEFAULT_FIRST_AGENT_MESSAGE,
    OUT_OF_SCOPE_TOKEN,
    STOP_TOKEN,
    Tau2StandardUserSimulator,
)

_TELECOM_ACCOUNT_FAULTS = {
    "data_usage_exceeded",
    "user_abroad_roaming_disabled_on",
    "user_abroad_roaming_disabled_off",
    "overdue_bill_suspension",
    "contract_end_suspension",
}

_TELECOM_USER_REPAIR_TOOLS_BY_FAULT = {
    "airplane_mode_on": {"toggle_airplane_mode", "turn_airplane_mode_off"},
    "bad_network_preference": {"set_network_mode_preference"},
    "bad_wifi_calling": {"toggle_wifi_calling", "set_wifi_calling"},
    "break_apn_mms_setting": {"reset_apn_settings", "set_apn_settings", "reboot_device"},
    "break_apn_settings": {"reset_apn_settings", "set_apn_settings", "reboot_device"},
    "break_app_sms_permission": {"grant_app_permission"},
    "break_app_storage_permission": {"grant_app_permission"},
    "break_app_both_permissions": {"grant_app_permission"},
    "data_mode_off": {"toggle_data", "turn_data_on"},
    "data_saver_mode_on": {"toggle_data_saver_mode", "turn_data_saver_mode_off"},
    "bad_vpn": {"disconnect_vpn"},
    "unseat_sim_card": {"reseat_sim_card"},
    "user_abroad_roaming_enabled_off": {"toggle_roaming", "turn_roaming_on"},
    "overdue_bill_suspension": {"check_payment_request", "make_payment"},
}

_TELECOM_USER_DISCOVERY_TOOLS_BY_FAULT = {
    "airplane_mode_on": {"check_network_status", "check_status_bar"},
    "bad_network_preference": {"check_network_status", "check_network_mode_preference"},
    "bad_wifi_calling": {"check_wifi_calling_status"},
    "break_apn_mms_setting": {"check_apn_settings"},
    "break_apn_settings": {"check_apn_settings"},
    "break_app_sms_permission": {"check_installed_apps", "check_app_status", "check_app_permissions"},
    "break_app_storage_permission": {"check_installed_apps", "check_app_status", "check_app_permissions"},
    "break_app_both_permissions": {"check_installed_apps", "check_app_status", "check_app_permissions"},
    "data_mode_off": {"check_network_status", "check_status_bar"},
    "data_saver_mode_on": {"check_data_restriction_status", "run_speed_test"},
    "bad_vpn": {"check_vpn_status"},
    "unseat_sim_card": {"check_network_status", "check_status_bar", "check_sim_status"},
    "lock_sim_card_pin": {"check_network_status", "check_status_bar", "check_sim_status"},
    "user_abroad_roaming_enabled_off": {"check_network_status", "check_status_bar"},
    "user_abroad_roaming_disabled_on": {"check_network_status", "check_status_bar"},
    "user_abroad_roaming_disabled_off": {"check_network_status", "check_status_bar"},
    "data_usage_exceeded": {"run_speed_test"},
    "overdue_bill_suspension": {"check_payment_request"},
}

_TELECOM_ASSISTANT_TOOLS_BY_FAULT = {
    "data_usage_exceeded": {
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "get_data_usage",
        "refuel_data",
        "set_data_usage",
    },
    "user_abroad_roaming_disabled_on": {
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "enable_roaming",
    },
    "user_abroad_roaming_disabled_off": {
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "enable_roaming",
    },
    "overdue_bill_suspension": {
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "get_bills_for_customer",
        "send_payment_request",
        "resume_line",
    },
    "contract_end_suspension": {
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "transfer_to_human_agents",
    },
}

_TELECOM_AREW_ASSISTANT_PROGRESS_READ_TOOLS = {
    "get_customer_by_phone",
    "get_customer_by_id",
    "get_customer_by_name",
    "get_details_by_id",
    "get_bills_for_customer",
    "get_data_usage",
}

_TELECOM_AREW_ASSISTANT_PROGRESS_WRITE_TOOLS = {
    "resume_line",
    "send_payment_request",
    "enable_roaming",
    "refuel_data",
}

_TELECOM_AREW_USER_DIAGNOSTIC_TOOLS = {
    "check_status_bar",
    "check_network_status",
    "check_network_mode_preference",
    "check_data_restriction_status",
    "check_apn_settings",
    "check_wifi_calling_status",
    "check_sim_status",
    "check_vpn_status",
    "check_app_status",
    "check_app_permissions",
    "check_installed_apps",
    "check_payment_request",
    "can_send_mms",
    "run_speed_test",
}

_TELECOM_AREW_USER_REPAIR_TOOLS = {
    "toggle_airplane_mode",
    "turn_airplane_mode_on",
    "turn_airplane_mode_off",
    "set_network_mode_preference",
    "toggle_data",
    "turn_data_on",
    "turn_data_off",
    "toggle_roaming",
    "turn_roaming_on",
    "turn_roaming_off",
    "toggle_data_saver_mode",
    "turn_data_saver_mode_on",
    "turn_data_saver_mode_off",
    "set_apn_settings",
    "reset_apn_settings",
    "toggle_wifi_calling",
    "set_wifi_calling",
    "disconnect_vpn",
    "grant_app_permission",
    "reseat_sim_card",
    "reboot_device",
    "make_payment",
}


class Tau2StandardSpace:
    supports_answer = False

    def __init__(
        self,
        controller: dict,
        max_turns: int,
        global_step: int = 0,
        trunc_strength=None,
        hard_tolerate_num: int = 2,
        strict_progress_label: bool = False,
        completion_bonus: float = 0.0,
        progress_mode: str = "legacy",
        progress_min_turn: int = 6,
        arew_label_mode: str = "clean",
        arew_min_turn: int | None = None,
        defer_bootstrap: bool = False,
    ):
        del global_step
        self.controller = controller
        self.domain = controller["domain"]
        self.task_split = controller.get("task_split")
        self.task_pool = controller.get("task_pool")
        self.task_id = controller["task_id"]
        self.reward_mode = controller.get("reward_mode", "task_basis")
        self.env_kwargs = controller.get("env_kwargs", {})
        self.cut = 0.0
        self.must_stop = 0.0
        self.query_counts = 0
        self.repeat_counts = 0
        self.stalling_counts = 0
        self.invalid_action_count = 0
        self.tool_error_count = 0
        self.no_state_change_write_count = 0
        self.hard_truncation = 0
        self.soft_truncation = 0
        self.max_counts = max_turns - 1
        self.final_reward = 0.0
        self.reward_info_official = None
        self.reward_info_fractional = None
        self.tolerate_num = int(trunc_strength) if trunc_strength is not None else 0
        self.hard_tolerate_num = int(hard_tolerate_num)
        self.strict_progress_label = bool(strict_progress_label)
        self.progress_mode = str(progress_mode or "legacy")
        self.progress_min_turn = max(1, int(progress_min_turn))
        self.arew_label_mode = str(arew_label_mode or "clean").strip().lower()
        if self.arew_label_mode not in {"clean", "new_family", "family", "e", "binary", "f", "targeted", "phase"}:
            raise ValueError(
                f"Unsupported tau2_arew_label_mode={arew_label_mode}. "
                "Choose from clean|new_family|family|e|binary|f|targeted|phase."
            )
        self.arew_min_turn = self.progress_min_turn if arew_min_turn is None else max(1, int(arew_min_turn))
        self.hard_bad_streak = 0
        self.soft_no_progress_streak = 0
        self.prev_matched_action_count = 0
        self.prev_assistant_matched_action_count = 0
        self.seen_raw_observation_signatures: set[str] = set()
        self.seen_normalized_observation_signatures: set[str] = set()
        self.seen_observation_families: set[str] = set()
        self.proxy_reason_history: list[str] = []
        self.last_step_label = 0
        self.last_step_reason = None
        self.step_label_history: list[int] = []
        self.last_arew_label = 0
        self.arew_label_history: list[int] = []
        self.last_arew_fallback_eligible = 0
        self.arew_fallback_eligible_history: list[int] = []
        self.all_actions: list[tuple[str, str]] = []
        self.visible_initial_turns: list[tuple[str, str]] = []
        self.seen_visible_slots: set[str] = set()
        self.num_errors = 0
        self.completion_bonus = float(completion_bonus)
        self.user_stop_reason: str | None = None
        self.assistant_message_turn_count = 0
        self.assistant_tool_turn_count = 0
        self.assistant_read_tool_turn_count = 0
        self.assistant_write_tool_turn_count = 0
        self.assistant_generic_tool_turn_count = 0
        self.user_tool_hop_count = 0
        self.bootstrap_user_tool_hop_count = 0
        self.assistant_tool_error_count = 0
        self.user_tool_error_count = 0
        self.user_disconnect_count = 0
        self.bootstrap_user_disconnect_count = 0
        self.disconnect_repeat_neutral_count = 0
        self.last_user_disconnect = False
        self.state_changed_step_count = 0
        self.action_progress_step_count = 0
        self.new_raw_information_step_count = 0
        self.new_normalized_information_step_count = 0
        self.new_family_step_count = 0
        self.max_soft_no_progress_streak = 0
        self.t3_truncation_turn = 0

        lookup_task_split = None if self.task_pool is not None else self.task_split
        tasks = get_tasks(self.domain, task_split_name=lookup_task_split, task_ids=[self.task_id], solo_only=False)
        self.task = tasks[0]
        self.task_issue, self.task_faults = self._parse_telecom_task_id(self.task.id)
        self.expected_action_names = {
            action.name
            for action in (getattr(self.task.evaluation_criteria, "actions", None) or [])
        }
        self.expected_user_name, self.expected_user_phone_number = self._extract_expected_user_info()
        self.expected_customer_ids: set[str] = set()
        self.expected_line_ids: set[str] = set()
        self.expected_bill_ids: set[str] = set()
        self.expected_customer_dobs: set[str] = set()
        self.env_constructor = get_env_constructor(self.domain)
        self.env = self.env_constructor(solo_mode=False, **self.env_kwargs)
        self.initial_history: list[AssistantMessage | UserMessage | ToolMessage] = []

        if self.task.initial_state is not None:
            self.initial_history = deepcopy(self.task.initial_state.message_history or [])
            if self.initial_history:
                raise NotImplementedError(
                    "Tau2 standard mode currently expects empty task.initial_state.message_history."
                )
            self.env.set_state(
                initialization_data=self.task.initial_state.initialization_data,
                initialization_actions=self.task.initial_state.initialization_actions,
                message_history=[],
            )
        self._resolve_expected_identity()

        self.trajectory: list[AssistantMessage | UserMessage | ToolMessage] = []
        self.user_simulator = Tau2StandardUserSimulator(self.task, self.env, controller)
        self.user_stopped = False
        self.last_agent_db_hash = self.env.get_db_hash()
        self.last_user_db_hash = self.env.get_user_db_hash()
        self._bootstrapped = False

        if not defer_bootstrap:
            self.bootstrap_conversation()

    def _get_tool_type(self, tool_name: str) -> ToolType:
        if self.env.tools is not None and self.env.tools.has_tool(tool_name):
            return self.env.tools.tool_type(tool_name)
        return ToolType.GENERIC

    def _assistant_has_tool(self, tool_name: str) -> bool:
        return self.env.tools is not None and self.env.tools.has_tool(tool_name)

    def _user_has_tool(self, tool_name: str) -> bool:
        return self.env.user_tools is not None and self.env.user_tools.has_tool(tool_name)

    def _format_assistant_tool_names(self) -> str:
        if self.env.tools is None:
            return ""
        return ", ".join(sorted(self.env.tools.get_tools().keys()))

    def _get_env_state_signature(self) -> tuple[str | None, str | None]:
        return self.env.get_db_hash(), self.env.get_user_db_hash()

    def _extract_expected_user_info(self) -> tuple[str | None, str | None]:
        if self.task.initial_state is None or self.task.initial_state.initialization_actions is None:
            return None, None
        for action in self.task.initial_state.initialization_actions:
            if action.env_type == "user" and action.func_name == "set_user_info":
                args = action.arguments or {}
                name = str(args.get("name", "")).strip() or None
                phone_number = str(args.get("phone_number", "")).strip() or None
                return name, phone_number
        return None, None

    def _resolve_expected_identity(self) -> None:
        if not self.expected_user_phone_number:
            return
        db = getattr(getattr(self.env, "tools", None), "db", None)
        if db is None:
            return

        lines_by_id = {getattr(line, "line_id", None): line for line in getattr(db, "lines", [])}
        for customer in getattr(db, "customers", []):
            customer_id = getattr(customer, "customer_id", None)
            line_ids = set(getattr(customer, "line_ids", []) or [])
            owned_target_line_ids = {
                line_id
                for line_id in line_ids
                if getattr(lines_by_id.get(line_id), "phone_number", None) == self.expected_user_phone_number
            }
            customer_phone_matches = getattr(customer, "phone_number", None) == self.expected_user_phone_number
            if not customer_phone_matches and not owned_target_line_ids:
                continue

            if customer_id:
                self.expected_customer_ids.add(customer_id)
            self.expected_line_ids.update(line_id for line_id in owned_target_line_ids if line_id)
            self.expected_bill_ids.update(getattr(customer, "bill_ids", []) or [])
            dob = getattr(customer, "date_of_birth", None)
            if dob:
                self.expected_customer_dobs.add(str(dob))

    @staticmethod
    def _parse_telecom_task_id(task_id: str) -> tuple[str | None, set[str]]:
        match = re.match(r"\[([^\]]+)\]([^\[]*)\[PERSONA:([^\]]+)\]", task_id or "")
        if match is None:
            return None, set()
        faults = {item for item in match.group(2).split("|") if item}
        return match.group(1), faults

    @staticmethod
    def _union_fault_tools(mapping: dict[str, set[str]], faults: set[str]) -> set[str]:
        tools: set[str] = set()
        for fault in faults:
            tools.update(mapping.get(fault, set()))
        return tools

    def _relevant_user_repair_tools(self) -> set[str]:
        return self._union_fault_tools(_TELECOM_USER_REPAIR_TOOLS_BY_FAULT, self.task_faults)

    def _relevant_user_discovery_tools(self) -> set[str]:
        return self._union_fault_tools(_TELECOM_USER_DISCOVERY_TOOLS_BY_FAULT, self.task_faults)

    def _relevant_assistant_tools(self) -> set[str]:
        tools = self._union_fault_tools(_TELECOM_ASSISTANT_TOOLS_BY_FAULT, self.task_faults)
        if self.task_faults & _TELECOM_ACCOUNT_FAULTS:
            tools.update({"get_customer_by_phone", "get_customer_by_id", "get_customer_by_name", "get_details_by_id"})
        return tools

    @staticmethod
    def _json_loads_or_none(content: str) -> Any | None:
        try:
            return json.loads(content)
        except Exception:
            return None

    @staticmethod
    def _normalize_identifier(value: Any) -> str:
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _normalize_digits(value: Any) -> str:
        return "".join(ch for ch in str(value or "") if ch.isdigit())

    def _text_mentions_identifier(
        self,
        normalized_text: str,
        digit_text: str,
        identifier: str | None,
        *,
        allow_digit_fuzzy: bool = True,
    ) -> bool:
        normalized_identifier = self._normalize_identifier(identifier)
        if not normalized_identifier:
            return False
        normalized_identifier_lower = normalized_identifier.lower()
        if normalized_identifier_lower in normalized_text:
            return True

        if not allow_digit_fuzzy:
            return False

        normalized_identifier_digits = self._normalize_digits(normalized_identifier)
        return bool(normalized_identifier_digits) and normalized_identifier_digits in digit_text

    def _extract_visible_slots(self, text: str) -> set[str]:
        if not text:
            return set()

        normalized_text = self._normalize_whitespace(text)
        digit_text = self._normalize_digits(text)
        slots: set[str] = set()

        if self.expected_user_phone_number and self._text_mentions_identifier(
            normalized_text,
            digit_text,
            self.expected_user_phone_number,
            allow_digit_fuzzy=True,
        ):
            slots.add("phone_number")

        for dob in sorted(self.expected_customer_dobs):
            if self._text_mentions_identifier(normalized_text, digit_text, dob, allow_digit_fuzzy=True):
                slots.add(f"dob:{dob}")

        for customer_id in sorted(self.expected_customer_ids):
            if self._text_mentions_identifier(normalized_text, digit_text, customer_id, allow_digit_fuzzy=False):
                slots.add(f"customer_id:{customer_id}")

        for line_id in sorted(self.expected_line_ids):
            if self._text_mentions_identifier(normalized_text, digit_text, line_id, allow_digit_fuzzy=False):
                slots.add(f"line_id:{line_id}")

        for bill_id in sorted(self.expected_bill_ids):
            if self._text_mentions_identifier(normalized_text, digit_text, bill_id, allow_digit_fuzzy=False):
                slots.add(f"bill_id:{bill_id}")

        return slots

    def _record_visible_slots(self, text: str) -> bool:
        current_slots = self._extract_visible_slots(text)
        new_slots = current_slots - self.seen_visible_slots
        self.seen_visible_slots.update(current_slots)
        return bool(new_slots)

    def _argument_references_wrong_identity(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        if self.expected_user_phone_number:
            phone_number = self._normalize_identifier(arguments.get("phone_number"))
            if phone_number and phone_number != self.expected_user_phone_number:
                return True

        if self.expected_user_name and tool_name == "get_customer_by_name":
            full_name = self._normalize_identifier(arguments.get("full_name"))
            if full_name and self._normalize_whitespace(full_name) != self._normalize_whitespace(self.expected_user_name):
                return True

        if self.expected_customer_dobs and tool_name == "get_customer_by_name":
            dob = self._normalize_identifier(arguments.get("dob"))
            if dob and dob not in self.expected_customer_dobs:
                return True

        customer_id = self._normalize_identifier(arguments.get("customer_id"))
        if customer_id and self.expected_customer_ids and customer_id not in self.expected_customer_ids:
            return True

        line_id = self._normalize_identifier(arguments.get("line_id"))
        if line_id and self.expected_line_ids and line_id not in self.expected_line_ids:
            return True

        bill_id = self._normalize_identifier(arguments.get("bill_id"))
        if bill_id and self.expected_bill_ids and bill_id not in self.expected_bill_ids:
            return True

        generic_id = self._normalize_identifier(arguments.get("id"))
        if generic_id.startswith("C") and self.expected_customer_ids and generic_id not in self.expected_customer_ids:
            return True
        if generic_id.startswith("L") and self.expected_line_ids and generic_id not in self.expected_line_ids:
            return True
        if generic_id.startswith("B") and self.expected_bill_ids and generic_id not in self.expected_bill_ids:
            return True

        return False

    def _response_references_wrong_identity(self, tool_name: str, response_text: str) -> bool:
        payload = self._json_loads_or_none(response_text)
        if payload is None:
            return False

        if tool_name == "get_customer_by_name" and isinstance(payload, list) and self.expected_customer_ids:
            returned_customer_ids = {
                self._normalize_identifier(item.get("customer_id"))
                for item in payload
                if isinstance(item, dict) and item.get("customer_id") is not None
            }
            return bool(returned_customer_ids) and returned_customer_ids.isdisjoint(self.expected_customer_ids)

        if isinstance(payload, dict):
            customer_id = self._normalize_identifier(payload.get("customer_id"))
            line_id = self._normalize_identifier(payload.get("line_id"))
            bill_id = self._normalize_identifier(payload.get("bill_id"))
            if customer_id and self.expected_customer_ids and customer_id not in self.expected_customer_ids:
                return True
            if line_id and self.expected_line_ids and line_id not in self.expected_line_ids:
                return True
            if bill_id and self.expected_bill_ids and bill_id not in self.expected_bill_ids:
                return True

        return False

    def _assistant_tool_wrong_identity(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        response_text: str,
    ) -> bool:
        return self._argument_references_wrong_identity(tool_name, arguments) or self._response_references_wrong_identity(
            tool_name,
            response_text,
        )

    def _matched_action_count(self, requestor: str | None = None) -> int:
        actions = getattr(self.task.evaluation_criteria, "actions", None)
        if not actions:
            return 0
        full_trajectory = [*self.initial_history, *self.trajectory]
        predicted_tool_calls = extract_tool_calls(full_trajectory)
        if requestor is not None:
            actions = [action for action in actions if action.requestor == requestor]
            predicted_tool_calls = [tool_call for tool_call in predicted_tool_calls if tool_call.requestor == requestor]
        return sum(
            any(action.compare_with_tool_call(tool_call) for tool_call in predicted_tool_calls)
            for action in actions
        )

    @staticmethod
    def _normalize_whitespace(content: str) -> str:
        return " ".join(content.strip().split()).lower()

    @classmethod
    def _normalize_observation_signature(cls, content: str) -> str:
        return cls._normalize_whitespace(content)

    @classmethod
    def _parse_key_value_lines(cls, text: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[cls._normalize_whitespace(key)] = cls._normalize_whitespace(value)
        return result

    @classmethod
    def _normalize_user_tool_signature(cls, tool_name: str, tool_result_text: str) -> str:
        text = tool_result_text.strip()
        if not text:
            return f"{tool_name}:empty"

        if tool_name == "check_network_status":
            fields = cls._parse_key_value_lines(text)
            kept = {
                "airplane mode": fields.get("airplane mode", ""),
                "sim card status": fields.get("sim card status", ""),
                "cellular connection": fields.get("cellular connection", ""),
                "cellular signal": fields.get("cellular signal", ""),
                "cellular network type": fields.get("cellular network type", ""),
                "mobile data enabled": fields.get("mobile data enabled", ""),
                "data roaming enabled": fields.get("data roaming enabled", ""),
                "wi-fi connected": fields.get("wi-fi connected", ""),
            }
            return f"{tool_name}:{json.dumps(kept, sort_keys=True, ensure_ascii=False)}"

        if tool_name == "check_status_bar":
            return f"{tool_name}:{cls._normalize_whitespace(text)}"

        if tool_name == "check_network_mode_preference":
            fields = cls._parse_key_value_lines(text)
            mode = fields.get("network mode preference", cls._normalize_whitespace(text))
            return f"{tool_name}:{mode}"

        if tool_name == "check_apn_settings":
            fields = cls._parse_key_value_lines(text)
            kept = {
                "apn": fields.get("current apn name", ""),
                "mmsc": fields.get("mmsc url (for picture messages)", ""),
            }
            return f"{tool_name}:{json.dumps(kept, sort_keys=True, ensure_ascii=False)}"

        if tool_name == "check_app_permissions":
            normalized = cls._normalize_whitespace(text)
            app_name = ""
            permissions: tuple[str, ...] = ()
            if "app '" in text.lower():
                try:
                    app_name = text.split("app '", 1)[1].split("'", 1)[0].strip().lower()
                except Exception:
                    app_name = ""
            lower_text = text.lower()
            if "permission for:" in lower_text:
                try:
                    raw_permissions = lower_text.split("permission for:", 1)[1]
                    permissions = tuple(sorted(p.strip().lower() for p in raw_permissions.split(",") if p.strip()))
                except Exception:
                    permissions = ()
            if app_name or permissions:
                return f"{tool_name}:{json.dumps({'app': app_name, 'perms': permissions}, sort_keys=True, ensure_ascii=False)}"
            return f"{tool_name}:{normalized}"

        if tool_name == "check_wifi_calling_status":
            return f"{tool_name}:{'on' if 'on' in text.lower() else 'off'}"

        if tool_name == "check_sim_status":
            return f"{tool_name}:{cls._normalize_whitespace(text)}"

        if tool_name == "can_send_mms":
            lowered = text.lower()
            if "cannot send" in lowered or "can't send" in lowered:
                return f"{tool_name}:false"
            if "can send" in lowered:
                return f"{tool_name}:true"
            return f"{tool_name}:{cls._normalize_whitespace(text)}"

        if tool_name == "run_speed_test":
            normalized = cls._normalize_whitespace(text)
            return f"{tool_name}:{'failed' if 'failed' in normalized else normalized}"

        if tool_name == "check_installed_apps":
            lowered = text.lower()
            if "installed on the phone:" in lowered:
                raw_apps = lowered.split("installed on the phone:", 1)[1]
                apps = tuple(sorted(app.strip().lower() for app in raw_apps.split(",") if app.strip()))
                return f"{tool_name}:{json.dumps(apps, ensure_ascii=False)}"
            return f"{tool_name}:{cls._normalize_whitespace(text)}"

        return f"{tool_name}:{cls._normalize_whitespace(text)}"

    @classmethod
    def _normalize_assistant_tool_signature(cls, tool_name: str, tool_result_text: str) -> str:
        text = tool_result_text.strip()
        if not text:
            return f"{tool_name}:empty"

        if text.startswith("{") or text.startswith("["):
            try:
                payload = json.loads(text)
            except Exception:
                return f"{tool_name}:{cls._normalize_whitespace(text)}"

            if tool_name in {"get_customer_by_phone", "get_customer_by_id"}:
                kept = {
                    "customer_id": payload.get("customer_id"),
                    "phone_number": payload.get("phone_number"),
                    "line_ids": payload.get("line_ids"),
                    "account_status": payload.get("account_status"),
                }
                return f"{tool_name}:{json.dumps(kept, sort_keys=True, ensure_ascii=False)}"
            if tool_name == "get_details_by_id":
                kept = {
                    "line_id": payload.get("line_id"),
                    "phone_number": payload.get("phone_number"),
                    "status": payload.get("status"),
                    "roaming_enabled": payload.get("roaming_enabled"),
                    "data_used_gb": payload.get("data_used_gb"),
                }
                return f"{tool_name}:{json.dumps(kept, sort_keys=True, ensure_ascii=False)}"
            if tool_name == "get_data_usage":
                kept = {
                    "line_id": payload.get("line_id"),
                    "data_used_gb": payload.get("data_used_gb"),
                    "data_limit_gb": payload.get("data_limit_gb"),
                    "data_refueling_gb": payload.get("data_refueling_gb"),
                }
                return f"{tool_name}:{json.dumps(kept, sort_keys=True, ensure_ascii=False)}"
            return f"{tool_name}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"

        return f"{tool_name}:{cls._normalize_whitespace(text)}"

    def _extract_user_tool_signatures(
        self,
        user_tool_results: list[tuple[str, str, bool]],
    ) -> list[tuple[str, str]]:
        signatures: list[tuple[str, str]] = []
        for tool_name, result_text, _ in user_tool_results:
            signatures.append((tool_name, self._normalize_user_tool_signature(tool_name, result_text)))
        return signatures

    def _extract_step_signatures(
        self,
        *,
        turn_kind: str,
        assistant_tool_name: str | None,
        assistant_tool_type: ToolType | None,
        assistant_observation: str,
        user_tool_results: list[tuple[str, str, bool]],
        final_user_text: str,
    ) -> list[str]:
        signatures: list[str] = []
        if turn_kind == "assistant_tool":
            if assistant_tool_name is not None and assistant_tool_type == ToolType.READ:
                signatures.append(self._normalize_assistant_tool_signature(assistant_tool_name, assistant_observation))
            return signatures

        if user_tool_results:
            for tool_name, result_text, _ in user_tool_results:
                signatures.append(self._normalize_user_tool_signature(tool_name, result_text))
            return signatures

        if final_user_text:
            signatures.append(f"user_message:{self._normalize_whitespace(final_user_text)}")
        return signatures

    def _record_step_label(self, label: int, reason: str | None = None) -> None:
        self.last_step_label = int(label)
        self.last_step_reason = reason
        self.step_label_history.append(int(label))

    def _record_arew_label(self, label: int, fallback_eligible: bool = False) -> None:
        self.last_arew_label = int(label)
        self.arew_label_history.append(int(label))
        self.last_arew_fallback_eligible = int(bool(fallback_eligible))
        self.arew_fallback_eligible_history.append(int(bool(fallback_eligible)))

    @staticmethod
    def _is_disconnect_fallback_message(message: UserMessage) -> bool:
        raw_data = getattr(message, "raw_data", None)
        return isinstance(raw_data, dict) and bool(raw_data.get("disconnect_fallback"))

    def _update_proxy_state(self, *, progress: bool, hard_bad: bool, reason: str | None, validate: bool) -> bool:
        if reason is not None:
            self.proxy_reason_history.append(reason)

        if progress:
            self.hard_bad_streak = 0
            self.soft_no_progress_streak = 0
            return False

        if self.query_counts < self.progress_min_turn:
            return False

        self.soft_no_progress_streak += 1
        self.max_soft_no_progress_streak = max(self.max_soft_no_progress_streak, self.soft_no_progress_streak)
        if self.progress_mode == "legacy" and hard_bad:
            self.hard_bad_streak += 1
            self.cut = 1.0
        else:
            self.hard_bad_streak = 0
            self.stalling_counts += 1

        if validate or self.tolerate_num <= 0:
            return False

        if self.progress_mode == "legacy" and self.hard_bad_streak >= self.hard_tolerate_num:
            self.must_stop = 1.0
            self.cut = 1.0
            self.hard_truncation = 1
            return True

        if self.soft_no_progress_streak >= self.tolerate_num:
            self.must_stop = 1.0
            self.cut = 1.0
            self.soft_truncation = 1
            if self.t3_truncation_turn == 0:
                self.t3_truncation_turn = self.query_counts
            return True

        return False

    def _derive_targeted_arew_label(
        self,
        *,
        assistant_tool_name: str | None,
        user_tool_error: bool,
        final_user_stop: str | None,
        tool_type: ToolType,
        state_changed: bool,
        action_progress: bool,
        new_family: bool,
        new_normalized_information: bool,
        user_diagnostic_progress: bool,
        slot_fill_progress: bool,
        user_repair_neutral: bool,
        arew_bad: bool,
    ) -> int:
        if final_user_stop == "stop":
            return 1
        if arew_bad:
            return -1

        assistant_read_progress = (
            assistant_tool_name in _TELECOM_AREW_ASSISTANT_PROGRESS_READ_TOOLS
            and tool_type == ToolType.READ
            and (new_family or new_normalized_information)
        )
        assistant_write_progress = (
            assistant_tool_name in _TELECOM_AREW_ASSISTANT_PROGRESS_WRITE_TOOLS
            and tool_type != ToolType.READ
            and state_changed
        )

        if (
            action_progress
            or assistant_read_progress
            or assistant_write_progress
            or user_diagnostic_progress
            or slot_fill_progress
        ):
            return 1

        expected_transfer = "transfer_to_human_agents" in self.expected_action_names
        premature_transfer = (
            final_user_stop in {"transfer", "out_of_scope"}
            or (assistant_tool_name == "transfer_to_human_agents" and not expected_transfer)
        )
        if premature_transfer:
            return -1

        if self.query_counts < self.arew_min_turn:
            return 0

        explicit_neutral = user_tool_error or user_repair_neutral
        if explicit_neutral:
            return 0

        return -1

    def _apply_progress_label(
        self,
        *,
        tool_type: ToolType,
        state_before: tuple[str | None, str | None],
        response_text: str,
        step_signatures: list[str] | None,
        hard_bad: bool,
        reason: str | None,
        validate: bool,
        assistant_tool_name: str | None = None,
        user_tool_names: list[str] | None = None,
        user_tool_signatures: list[tuple[str, str]] | None = None,
        user_tool_error: bool = False,
        final_user_stop: str | None = None,
        slot_fill_progress: bool = False,
        arew_hard_bad: bool | None = None,
        force_arew_label: int | None = None,
    ) -> bool:
        matched_action_count = self._matched_action_count()
        assistant_matched_action_count = self._matched_action_count(requestor="assistant")
        action_progress = matched_action_count > self.prev_matched_action_count
        assistant_action_progress = assistant_matched_action_count > self.prev_assistant_matched_action_count
        self.prev_matched_action_count = max(self.prev_matched_action_count, matched_action_count)
        self.prev_assistant_matched_action_count = max(
            self.prev_assistant_matched_action_count,
            assistant_matched_action_count,
        )

        state_after = self._get_env_state_signature()
        state_changed = state_after != state_before
        if state_changed:
            self.state_changed_step_count += 1

        raw_obs_signature = self._normalize_observation_signature(response_text)
        new_raw_information = bool(raw_obs_signature) and raw_obs_signature not in self.seen_raw_observation_signatures
        if raw_obs_signature:
            self.seen_raw_observation_signatures.add(raw_obs_signature)
        if new_raw_information:
            self.new_raw_information_step_count += 1

        normalized_signatures = [sig for sig in (step_signatures or []) if sig]
        seen_normalized_before = set(self.seen_normalized_observation_signatures)
        seen_families_before = set(self.seen_observation_families)
        new_normalized_information = any(
            signature not in seen_normalized_before for signature in normalized_signatures
        )
        for signature in normalized_signatures:
            self.seen_normalized_observation_signatures.add(signature)
        if new_normalized_information:
            self.new_normalized_information_step_count += 1

        step_families = [signature.split(":", 1)[0] for signature in normalized_signatures]
        new_family = any(family not in seen_families_before for family in step_families)
        for family in step_families:
            self.seen_observation_families.add(family)
        if new_family:
            self.new_family_step_count += 1
        if action_progress:
            self.action_progress_step_count += 1

        user_tools = set(user_tool_names or [])
        user_diagnostic_progress = False
        if not user_tool_error:
            for tool_name, signature in user_tool_signatures or []:
                if tool_name not in _TELECOM_AREW_USER_DIAGNOSTIC_TOOLS:
                    continue
                family = signature.split(":", 1)[0]
                if signature not in seen_normalized_before or family not in seen_families_before:
                    user_diagnostic_progress = True
                    break

        user_repair_neutral = (
            bool(user_tools & _TELECOM_AREW_USER_REPAIR_TOOLS)
            and not user_tool_error
            and state_changed
        )

        progress = False
        updated_reason = reason
        updated_hard_bad = hard_bad

        if self.progress_mode == "state_only":
            progress = state_changed
            if not progress and updated_reason is None:
                updated_reason = "no_state_change"
        elif self.progress_mode == "raw_obs":
            progress = state_changed or new_raw_information
            if not progress and updated_reason is None:
                updated_reason = "no_raw_progress"
        elif self.progress_mode == "norm_obs":
            progress = state_changed or new_normalized_information
            if not progress and updated_reason is None:
                updated_reason = "no_normalized_progress"
        elif self.progress_mode == "goal_aware":
            progress = state_changed or action_progress
            if not progress and updated_reason is None:
                updated_reason = "no_goal_progress"
        elif self.progress_mode == "new_family":
            progress = state_changed or new_family
            if not progress and updated_reason is None:
                updated_reason = "no_new_family"
        elif self.progress_mode == "new_family_goal":
            progress = state_changed or action_progress or new_family
            if not progress and updated_reason is None:
                updated_reason = "no_new_family_or_goal_progress"
        elif self.strict_progress_label:
            progress = action_progress
            if not progress and updated_reason is None:
                if tool_type == ToolType.READ:
                    updated_reason = "read_new_information" if new_normalized_information else "read_no_new_information"
                elif tool_type == ToolType.WRITE and state_changed:
                    updated_reason = "state_changed_without_action_progress"
                elif tool_type == ToolType.GENERIC and new_normalized_information:
                    updated_reason = "message_new_information"
                else:
                    updated_reason = "no_progress"
        else:
            progress = action_progress or (tool_type == ToolType.WRITE and state_changed)
            progress = progress or new_normalized_information
            if not progress and updated_reason is None:
                updated_reason = "no_progress"

        if self.progress_mode == "legacy":
            step_label = 1 if progress else -1 if updated_hard_bad else 0
        else:
            step_label = 1 if progress else 0
        self._record_step_label(step_label, updated_reason)

        arew_bad = updated_hard_bad if arew_hard_bad is None else bool(arew_hard_bad)
        if force_arew_label is not None:
            arew_label = int(force_arew_label)
            arew_fallback_eligible = False
        elif self.arew_label_mode in {"targeted", "phase"}:
            arew_label = self._derive_targeted_arew_label(
                assistant_tool_name=assistant_tool_name,
                user_tool_error=user_tool_error,
                final_user_stop=final_user_stop,
                tool_type=tool_type,
                state_changed=state_changed,
                action_progress=assistant_action_progress,
                new_family=new_family,
                new_normalized_information=new_normalized_information,
                user_diagnostic_progress=user_diagnostic_progress,
                slot_fill_progress=slot_fill_progress,
                user_repair_neutral=user_repair_neutral,
                arew_bad=arew_bad,
            )
            arew_fallback_eligible = arew_label == 0 and user_repair_neutral
        elif state_changed or action_progress or STOP_TOKEN in (response_text or ""):
            arew_label = 1
            arew_fallback_eligible = False
        elif arew_bad:
            arew_label = -1
            arew_fallback_eligible = False
        elif self.arew_label_mode in {"new_family", "family", "e"}:
            arew_label = 0 if new_family else -1
            arew_fallback_eligible = False
        elif self.arew_label_mode in {"binary", "f"}:
            arew_label = -1
            arew_fallback_eligible = False
        else:
            arew_label = 0
            arew_fallback_eligible = False
        self._record_arew_label(arew_label, fallback_eligible=arew_fallback_eligible)

        should_stop = self._update_proxy_state(
            progress=progress,
            hard_bad=updated_hard_bad,
            reason=updated_reason,
            validate=validate,
        )
        self.last_agent_db_hash, self.last_user_db_hash = state_after
        return should_stop

    def _resolve_user_turn(
        self, target: list[AssistantMessage | UserMessage | ToolMessage]
    ) -> tuple[UserMessage, list[tuple[str, str, bool]]]:
        tool_hops = 0
        user_tool_results: list[tuple[str, str, bool]] = []
        while True:
            user_message = self.user_simulator.generate_next_message()
            target.append(user_message)
            if self._is_disconnect_fallback_message(user_message):
                self.user_disconnect_count += 1
                if not self._bootstrapped:
                    self.bootstrap_user_disconnect_count += 1
                return user_message, user_tool_results

            if not user_message.is_tool_call():
                stop_kind = Tau2StandardUserSimulator.stop_kind(user_message)
                if stop_kind is not None:
                    self.must_stop = 1.0
                    self.user_stopped = True
                    self.user_stop_reason = stop_kind
                return user_message, user_tool_results

            tool_call = user_message.tool_calls[0]
            tool_result = self.env.get_response(tool_call)
            target.append(tool_result)
            self.user_simulator.append_tool_result(tool_result)
            user_tool_results.append((tool_call.name, tool_result.content, bool(tool_result.error)))
            self.user_tool_hop_count += 1
            if not self._bootstrapped:
                self.bootstrap_user_tool_hop_count += 1
            if tool_result.error:
                self.tool_error_count += 1
                self.user_tool_error_count += 1
                self.num_errors += 1
            tool_hops += 1

            if tool_hops >= self.user_simulator.max_tool_hops:
                fallback = UserMessage(role="user", content=OUT_OF_SCOPE_TOKEN)
                self.user_simulator.history.append(fallback)
                target.append(fallback)
                self.must_stop = 1.0
                self.user_stopped = True
                self.user_stop_reason = "out_of_scope"
                return fallback, user_tool_results

    def _bootstrap_conversation(self) -> None:
        assistant_greeting = AssistantMessage(role="assistant", content=DEFAULT_FIRST_AGENT_MESSAGE)
        self.initial_history.append(assistant_greeting)
        self.user_simulator.append_assistant_message(assistant_greeting)
        first_user_message, _ = self._resolve_user_turn(self.initial_history)
        self.visible_initial_turns.append(("assistant", DEFAULT_FIRST_AGENT_MESSAGE))
        if first_user_message.has_text_content():
            self.visible_initial_turns.append(("user", first_user_message.content or ""))
            self._record_visible_slots(first_user_message.content or "")

    def bootstrap_conversation(self) -> None:
        if self._bootstrapped:
            return
        self._bootstrap_conversation()
        self.last_agent_db_hash = self.env.get_db_hash()
        self.last_user_db_hash = self.env.get_user_db_hash()
        self._bootstrapped = True

    def bootstrap_visible_turns(self) -> list[tuple[str, str]]:
        return list(self.visible_initial_turns)

    def register_invalid_action(self, reason: str = "invalid_action_format") -> None:
        self.last_user_disconnect = False
        self.invalid_action_count += 1
        self._record_step_label(-1, reason)
        self._record_arew_label(-1)

    def register_answer(self, answer_text: str | None = None) -> None:
        del answer_text
        self.register_invalid_action("answer_not_supported_in_standard_mode")

    def compute_feedback(self, action_text: str, validate: bool = False):
        if not validate and self.must_stop == 1.0:
            self._record_step_label(0, "already_stopped")
            self._record_arew_label(0)
            return None

        self.last_user_disconnect = False
        self.query_counts += 1
        action_key = ("interact", action_text.strip())
        repeated_action = action_key in self.all_actions
        if repeated_action:
            self.repeat_counts += 1
        self.all_actions.append(action_key)

        try:
            message = parse_action_string(action_text, requestor="assistant")
        except Exception as exc:
            self.invalid_action_count += 1
            self._record_step_label(-1, "invalid_tool_call")
            self._record_arew_label(-1)
            should_stop = self._update_proxy_state(
                progress=False,
                hard_bad=True,
                reason="invalid_tool_call",
                validate=validate,
            )
            if should_stop:
                return None
            return {"role": "user", "content": f"Invalid tool call. Error: {exc}"}

        if not isinstance(message, AssistantMessage) or not message.is_tool_call() or len(message.tool_calls) != 1:
            self.invalid_action_count += 1
            self._record_step_label(-1, "invalid_action_format")
            self._record_arew_label(-1)
            should_stop = self._update_proxy_state(
                progress=False,
                hard_bad=True,
                reason="invalid_action_format",
                validate=validate,
            )
            if should_stop:
                return None
            return {
                "role": "user",
                "content": (
                    "Invalid action. Use exactly one <interact>tool_name(...)</interact> block when you want to call a tool."
                ),
            }

        tool_call = message.tool_calls[0]
        if not self._assistant_has_tool(tool_call.name):
            self.invalid_action_count += 1
            specific_reason = "user_tool_called_by_assistant" if self._user_has_tool(tool_call.name) else "unknown_assistant_tool"
            self._record_step_label(-1, specific_reason)
            self._record_arew_label(-1)
            should_stop = self._update_proxy_state(
                progress=False,
                hard_bad=True,
                reason=specific_reason,
                validate=validate,
            )
            if should_stop:
                return None
            allowed_tools = self._format_assistant_tool_names()
            if self._user_has_tool(tool_call.name):
                return {
                    "role": "user",
                    "content": (
                        "Invalid action. That tool is a user-side action, not an assistant tool. "
                        "Ask the user to do it with <message>...</message>. "
                        f"Callable assistant tools: {allowed_tools}"
                    ),
                }
            return {
                "role": "user",
                "content": (
                    "Invalid action. That tool is not available to the assistant. "
                    f"Callable assistant tools: {allowed_tools}"
                ),
            }
        tool_type = self._get_tool_type(tool_call.name)
        self.assistant_tool_turn_count += 1
        if tool_type == ToolType.READ:
            self.assistant_read_tool_turn_count += 1
        elif tool_type == ToolType.WRITE:
            self.assistant_write_tool_turn_count += 1
        else:
            self.assistant_generic_tool_turn_count += 1
        state_before = self._get_env_state_signature()
        self.trajectory.append(message)
        response = self.env.get_response(tool_call)
        self.trajectory.append(response)

        hard_bad = repeated_action
        arew_hard_bad = repeated_action
        reason = "repeat_action" if repeated_action else None

        if response.error:
            self.tool_error_count += 1
            self.assistant_tool_error_count += 1
            self.num_errors += 1
            hard_bad = True
            arew_hard_bad = True
            reason = "tool_error"
        elif self._assistant_tool_wrong_identity(tool_call.name, tool_call.arguments, response.content):
            hard_bad = True
            arew_hard_bad = True
            reason = "wrong_identity"
        else:
            state_after = self._get_env_state_signature()
            if tool_type == ToolType.WRITE and state_after == state_before:
                self.no_state_change_write_count += 1
                hard_bad = True
                reason = "write_no_state_change"

        should_stop = self._apply_progress_label(
            tool_type=tool_type,
            state_before=state_before,
            response_text=response.content,
            step_signatures=self._extract_step_signatures(
                turn_kind="assistant_tool",
                assistant_tool_name=tool_call.name,
                assistant_tool_type=tool_type,
                assistant_observation=response.content,
                user_tool_results=[],
                final_user_text="",
            ),
            assistant_tool_name=tool_call.name,
            user_tool_names=[],
            user_tool_signatures=[],
            user_tool_error=False,
            final_user_stop=None,
            slot_fill_progress=False,
            hard_bad=hard_bad,
            reason=reason,
            validate=validate,
            arew_hard_bad=arew_hard_bad,
        )
        if should_stop:
            return None

        if self.query_counts >= self.max_counts:
            self.must_stop = 1.0
            return None

        return {"role": "tool", "content": response.content}

    def compute_message_feedback(self, message_text: str, validate: bool = False):
        if not validate and self.must_stop == 1.0:
            self._record_step_label(0, "already_stopped")
            self._record_arew_label(0)
            return None

        message_text = message_text.strip()
        if not message_text:
            self.register_invalid_action("empty_message")
            should_stop = self._update_proxy_state(
                progress=False,
                hard_bad=True,
                reason="empty_message",
                validate=validate,
            )
            if should_stop:
                return None
            return {
                "role": "user",
                "content": "Invalid action. Use <message>...</message> when you want to speak to the user.",
            }

        self.query_counts += 1
        action_key = ("message", message_text)
        repeated_message = action_key in self.all_actions
        repeated_after_disconnect = repeated_message and self.last_user_disconnect
        if repeated_message:
            self.repeat_counts += 1
        if repeated_after_disconnect:
            self.disconnect_repeat_neutral_count += 1
        self.all_actions.append(action_key)

        assistant_message = AssistantMessage(role="assistant", content=message_text)
        self.assistant_message_turn_count += 1
        state_before = self._get_env_state_signature()
        self.trajectory.append(assistant_message)
        self.user_simulator.append_assistant_message(assistant_message)
        user_message, user_tool_results = self._resolve_user_turn(self.trajectory)
        if self._is_disconnect_fallback_message(user_message):
            if repeated_after_disconnect:
                self._record_step_label(0, "repeat_after_user_disconnect")
                self._record_arew_label(0)
            elif repeated_message:
                self._record_step_label(-1, "repeat_message")
                self._record_arew_label(-1)
            else:
                self._record_step_label(0, "user_disconnect")
                self._record_arew_label(0)
            self.last_user_disconnect = True
            if self.query_counts >= self.max_counts:
                self.must_stop = 1.0
                return None
            return {"role": "user", "content": user_message.content or ""}

        self.last_user_disconnect = False
        final_user_stop = Tau2StandardUserSimulator.stop_kind(user_message)
        user_tool_signatures = self._extract_user_tool_signatures(user_tool_results)
        slot_fill_progress = self._record_visible_slots(user_message.content or "")
        should_stop = self._apply_progress_label(
            tool_type=ToolType.GENERIC,
            state_before=state_before,
            response_text=user_message.content or "",
            step_signatures=self._extract_step_signatures(
                turn_kind="assistant_message",
                assistant_tool_name=None,
                assistant_tool_type=None,
                assistant_observation="",
                user_tool_results=user_tool_results,
                final_user_text=user_message.content or "",
            ),
            assistant_tool_name=None,
            user_tool_names=[tool_name for tool_name, _, _ in user_tool_results],
            user_tool_signatures=user_tool_signatures,
            user_tool_error=any(error for _, _, error in user_tool_results),
            final_user_stop=final_user_stop,
            slot_fill_progress=slot_fill_progress,
            hard_bad=repeated_message and not repeated_after_disconnect,
            reason=(
                "repeat_after_user_disconnect"
                if repeated_after_disconnect
                else "repeat_message"
                if repeated_message
                else None
            ),
            validate=validate,
            arew_hard_bad=repeated_message and not repeated_after_disconnect,
            force_arew_label=0 if repeated_after_disconnect else None,
        )
        if should_stop:
            return None

        if self.must_stop == 1.0:
            return None

        if self.query_counts >= self.max_counts:
            self.must_stop = 1.0
            return None

        return {"role": "user", "content": user_message.content or ""}

    def finalize(self, official: bool = False):
        cached = self.reward_info_official if official else self.reward_info_fractional
        if cached is not None:
            return cached
        if official and not self.user_stopped:
            reward_info = RewardInfo(
                reward=0.0,
                info={"note": "Simulation terminated prematurely before user stop in standard mode."},
            )
            self.reward_info_official = reward_info
            self.final_reward = 0.0
            return reward_info
        full_trajectory = [*self.initial_history, *self.trajectory]
        reward_info = evaluate_task(
            task=self.task,
            full_trajectory=full_trajectory,
            environment_constructor=self.env_constructor,
            reward_mode=self.reward_mode,
            solo_mode=False,
            env_kwargs=self.env_kwargs,
            fractional=not official,
        )
        if not official and self.user_stop_reason == "stop" and self.completion_bonus != 0.0:
            reward_info.reward = min(1.0, float(reward_info.reward) + self.completion_bonus)
            reward_info.reward_breakdown["COMPLETION_BONUS"] = float(self.completion_bonus)
            if reward_info.info is None:
                reward_info.info = {}
            reward_info.info["completion_bonus_applied"] = float(self.completion_bonus)
        if official:
            self.reward_info_official = reward_info
        else:
            self.reward_info_fractional = reward_info
        self.final_reward = float(reward_info.reward)
        return reward_info

    def format_trajectory(self) -> str:
        full_trajectory = [*self.initial_history, *self.trajectory]
        lines = []
        official_reward = self.reward_info_official.reward if self.reward_info_official is not None else None
        fractional_reward = self.reward_info_fractional.reward if self.reward_info_fractional is not None else None
        stop_reason = self.user_stop_reason or "max_turn_or_runtime_end"
        lines.append(
            "[summary] "
            f"stop_reason={stop_reason} "
            f"official_reward={official_reward if official_reward is not None else 'n/a'} "
            f"train_fractional_reward={fractional_reward if fractional_reward is not None else 'n/a'}"
        )
        for item in full_trajectory:
            role = getattr(item, "role", "unknown")
            if role in {"assistant", "user"} and getattr(item, "tool_calls", None):
                for tool_call in item.tool_calls:
                    args = ", ".join(
                        f"{key}={json.dumps(value, ensure_ascii=False)}"
                        for key, value in tool_call.arguments.items()
                    )
                    if role == "assistant":
                        lines.append(f"[assistant-tool-call][agent-output] {tool_call.name}({args})")
                    else:
                        lines.append(f"[user-tool-call][invisible-to-agent] {tool_call.name}({args})")
            elif role == "tool":
                prefix = "[tool][error]" if getattr(item, "error", False) else "[tool]"
                requestor = getattr(item, "requestor", None)
                if requestor:
                    visibility = "agent-visible" if requestor == "assistant" else "invisible-to-agent"
                    prefix = f"{prefix}[to={requestor}][{visibility}]"
                lines.append(f"{prefix} {item.content}")
            else:
                content = getattr(item, "content", None)
                if content:
                    if role == "assistant":
                        lines.append(f"[assistant][agent-output] {content}")
                    elif role == "user":
                        stop_kind = Tau2StandardUserSimulator.stop_kind(item)
                        visibility = "invisible-to-agent" if stop_kind is not None else "agent-visible"
                        suffix = f"[stop={stop_kind}]" if stop_kind is not None else ""
                        lines.append(f"[user][{visibility}]{suffix} {content}")
                    else:
                        lines.append(f"[{role}] {content}")
        return "\n".join(lines)
