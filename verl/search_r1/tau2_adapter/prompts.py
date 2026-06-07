import re

from search_r1.tau2_adapter.core.environment import Environment


def _format_tool_list(tools) -> str:
    return "\n\n".join(f"{idx + 1}. {tool.to_str()}" for idx, tool in enumerate(tools))


def _format_tools(task, env: Environment) -> str:
    assistant_tools = env.get_tools_description("assistant") or ""
    if not env.solo_mode:
        return assistant_tools
    try:
        user_tools = _format_tool_list(env.get_user_tools(include=task.user_tools)) or ""
    except Exception:
        user_tools = ""
    if assistant_tools and user_tools:
        return assistant_tools + "\n\n" + user_tools
    return assistant_tools or user_tools


_TELECOM_SECTION_BY_ISSUE = {
    "service_issue": "# Understanding and Troubleshooting Your Phone's Cellular Service",
    "mobile_data_issue": "# Understanding and Troubleshooting Your Phone's Mobile Data",
    "mms_issue": "# Understanding and Troubleshooting MMS (Picture/Video Messaging)",
}


def _get_telecom_issue_type(task) -> str | None:
    task_id = getattr(task, "id", "") or ""
    match = re.match(r"^\[([^\]]+)\]", task_id)
    if match:
        return match.group(1)

    ticket = (task.ticket or "").lower()
    if "mms" in ticket or "picture" in ticket or "video messaging" in ticket:
        return "mms_issue"
    if "mobile data" in ticket or "browse the internet" in ticket or "speed test" in ticket:
        return "mobile_data_issue"
    if "no service" in ticket or "cellular service" in ticket or "sim card" in ticket:
        return "service_issue"
    return None


def _extract_top_level_section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start == -1:
        return ""
    tail = text[start:]
    next_match = re.search(r"\n# ", tail[len(heading) :])
    if next_match is None:
        return tail.strip()
    end = len(heading) + next_match.start()
    return tail[:end].strip()


def _extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\n?(.*?)\n?</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _build_telecom_policy(task, env: Environment) -> str:
    policy = env.get_policy()
    main_policy = _extract_tag_block(policy, "main_policy")
    tech_policy = _extract_tag_block(policy, "tech_support_policy")

    issue_type = _get_telecom_issue_type(task)
    relevant_heading = _TELECOM_SECTION_BY_ISSUE.get(issue_type)
    relevant_section = _extract_top_level_section(tech_policy, relevant_heading) if relevant_heading else ""

    blocks = []
    if main_policy:
        blocks.append(f"<main_policy>\n{main_policy}\n</main_policy>")
    if relevant_section:
        blocks.append(f"<tech_support_policy>\n{relevant_section}\n</tech_support_policy>")
    elif tech_policy:
        blocks.append(f"<tech_support_policy>\n{tech_policy}\n</tech_support_policy>")
    return "\n".join(blocks)


def _build_policy(task, env: Environment) -> str:
    if env.get_domain_name() in {"telecom", "telecom-workflow"}:
        return _build_telecom_policy(task, env)
    return env.get_policy()


def _build_output_format(enable_think: bool, think_mode: str) -> tuple[str, str]:
    if not enable_think:
        output_format = """Output format:
1. To call exactly one tool, respond with:
<interact>tool_name(arg1=value1, arg2=value2)</interact>
2. When you believe the ticket is solved, respond with:
<answer>done()</answer>

Rules:
- Only one tool call per turn.
- Do not speak to the user. There is no live user here.
- Do not put plain English inside <interact>.
- Use valid Python-like keyword arguments inside the tool call.
"""
        return output_format, "Start with your first tool call."

    if think_mode != "short":
        raise ValueError(f"Unsupported think_mode={think_mode}. Only 'short' is currently supported.")

    output_format = """Output format:
1. Before every tool call, first write a short thought and then exactly one tool call:
<think>Briefly state the current blocker and best next action in 1-2 short sentences.</think>
<interact>tool_name(arg1=value1, arg2=value2)</interact>
2. When you believe the ticket is solved, first write a short thought and then finish with:
<think>Briefly state why the task is resolved.</think>
<answer>done()</answer>

Rules:
- Keep <think> short. Do not restate the whole ticket or policy.
- Do not put tool syntax inside <think>.
- Only one tool call per turn.
- Do not speak to the user. There is no live user here.
- Do not put plain English inside <interact>.
- Use valid Python-like keyword arguments inside the tool call.
"""
    return output_format, "Start with your first brief thought and tool call."


def build_solo_prompt(task, env: Environment, enable_think: bool = False, think_mode: str = "short") -> str:
    ticket = task.ticket or str(task.user_scenario.instructions)
    tools_desc = _format_tools(task, env)
    policy = _build_policy(task, env)
    output_format, closing_instruction = _build_output_format(enable_think, think_mode)
    return f"""You are solving a customer-service ticket in a tool environment.

Follow the domain policy carefully.

<policy>
{policy}
</policy>

<ticket>
{ticket}
</ticket>

Available tools:
{tools_desc}

{output_format}

{closing_instruction}
"""


_STANDARD_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
Only the tools listed in "Available tools" are callable by you.
If the policy mentions phone-side checks or actions that are not listed in "Available tools", those are user-side actions.
For user-side actions, ask the user to do them via <message> instead of calling them yourself.
""".strip()


def _build_standard_output_format(enable_think: bool, think_mode: str) -> tuple[str, str]:
    if not enable_think:
        output_format = """Output format:
1. To send a message to the user, respond with:
<message>Your message to the user</message>
2. To call exactly one tool, respond with:
<interact>tool_name(arg1=value1, arg2=value2)</interact>

Rules:
- Each turn must contain exactly one <message> block or one <interact> block.
- Do not include tool syntax inside <message>.
- Do not include plain English outside the XML block.
- Use valid Python-like keyword arguments inside <interact>.
"""
        return output_format, "Wait for the user message and then take the best next action."

    if think_mode != "short":
        raise ValueError(f"Unsupported think_mode={think_mode}. Only 'short' is currently supported.")

    output_format = """Output format:
1. To send a message to the user:
<think>Briefly state the next conversational goal in 1-2 short sentences.</think>
<message>Your message to the user</message>
2. To call exactly one tool:
<think>Briefly state the current blocker and the best next tool.</think>
<interact>tool_name(arg1=value1, arg2=value2)</interact>

Rules:
- Keep <think> short and operational.
- Each turn must contain exactly one <message> block or one <interact> block.
- Only the tools listed in "Available tools" are callable inside <interact>.
- Do not include tool syntax inside <message> or <think>.
- Do not include plain English outside the XML blocks.
- Use valid Python-like keyword arguments inside <interact>.
"""
    return output_format, "Wait for the user message and then take the best next action."


def build_standard_system_prompt(
    task,
    env: Environment,
    enable_think: bool = False,
    think_mode: str = "short",
) -> str:
    policy = env.get_policy()
    tools_desc = _format_tools(task, env)
    output_format, closing_instruction = _build_standard_output_format(
        enable_think=enable_think,
        think_mode=think_mode,
    )
    tools_block = f"\nAvailable tools:\n{tools_desc}\n" if tools_desc else ""
    return f"""<instructions>
{_STANDARD_AGENT_INSTRUCTION}
</instructions>
<policy>
{policy}
</policy>
{tools_block}

{output_format}

{closing_instruction}
"""


def build_standard_prompt_messages(
    task,
    env: Environment,
    enable_think: bool = False,
    think_mode: str = "short",
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_standard_system_prompt(
                task,
                env,
                enable_think=enable_think,
                think_mode=think_mode,
            ),
        }
    ]
