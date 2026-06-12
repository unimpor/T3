MEDIQ_PATIENT_PROMPT = """You are simulating a patient in a medical interaction.
Your role is to answer the doctor's question truthfully and strictly based on
the atomic factual statements provided below.

**Available Information:**
You are given a list of atomic factual statements about yourself.
Each statement is indexed starting from 1.

You MUST use only these statements when answering.
Do not infer, combine, or reinterpret information beyond what is explicitly stated.

{atom_facts}

**Doctor's Question:**
{query}

**Your Task:**
- Identify which of the atomic factual statements directly answer the question.
- Select at most TWO statements.
- Output ONLY the index number(s) of the selected statements, separated by commas.
- If NONE of the statements answer the question, output exactly:
  Unknown

**Output Rules (strict):**
- Output only indices (e.g., `2` or `1,4`), or `Unknown`.
- Do NOT include explanations, analysis, or additional text.
- Do NOT restate the statements themselves.
"""


FLODIAL_USER_PROMPT = """You are simulating a diagnostic environment for an interactive troubleshooting task.

Your role is to respond to the agent's diagnostic queries strictly according to a predefined reference table.

**Setup:**
- You are given:
  (i) a task description describing a problem scenario, and
  (ii) a reference table consisting of numbered diagnostic questions, each with a fixed Yes/No answer.
- Exactly ONE row in the reference table corresponds to the ground-truth fault.

**Reference Table:**
Each entry has the form:
- ID: <number>
- Question: <diagnostic yes/no question>
- Answer: Yes or No

The reference table is authoritative and complete.

**Response Rules (STRICT):**
- When the agent asks a query:
  1) Check whether the query matches (or is a clear paraphrase of) ONE of the reference questions.
  2) If it matches:
     - Return ONLY the corresponding answer and question ID in the exact format:
       - "Yes, <ID>" or
       - "No, <ID>"
  3) If it does NOT clearly match any reference question:
     - Return ONLY:
       - "Unknown"

**Important Constraints:**
- Do NOT provide explanations.
- Do NOT provide additional text.
- Do NOT answer multiple questions.
- Do NOT guess.
- Do NOT infer beyond the reference table.
- If the query is ambiguous, multi-part, procedural (e.g., "How do I check...?"), or not a direct diagnostic test outcome, return "Unknown".

**Output Format (STRICT):**
- Either:
  - "Yes, <ID>"
  - "No, <ID>"
  - "Unknown"

---

Task Description:
{task_description}

Reference Diagnostic Table:
{reference_table}

Agent's query:
{query}

Respond following the rules above.
"""
