import re
from collections import Counter
from typing import List

import numpy as np

from .prompts import FLODIAL_USER_PROMPT, MEDIQ_PATIENT_PROMPT


def token_f1(str1: str, str2: str) -> float:
    def compute_f1_scores(prediction: List[str], ground_truth: List[str]) -> float:
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = float(num_same) / len(prediction)
        recall = float(num_same) / len(ground_truth)
        return (2.0 * precision * recall) / (precision + recall)

    return compute_f1_scores(str1.split(), str2.split())


class PreferenceEstimationSpace:
    def __init__(self, controller, max_turns: int, version: str):
        self.controller = controller
        self.user_pref = controller["user_preference_function"]
        self.seen_movies_scores = controller["seen_movies_attribute_scores"]
        self.version = version
        self.max_counts = max_turns
        self.query_counts = 0
        self.repeat_counts = 0
        self.stalling_counts = 0
        self.consecutive_drop_counts = 0
        self.must_stop = 0.0
        self.cut = 0.0

        self.dims = len(self.user_pref)
        self.all_queries = []
        self.all_acinfo = []
        self.asp1_record_vec = [np.array([0.5] * self.dims)]
        self.asp1_record_mij = []
        self.asp1_record_yij = []
        self.asp1_record_focus = []

        guess_fmt = ",".join([f"w{i}" for i in range(1, self.dims + 1)])
        focus_fmt = "k1,k2,k3" if version == "gated3" else "k1,k2"
        self.answer_instruct = (
            "**Final Turn**: Briefly think in <scratch>...</scratch> "
            "about how the latest feedback updates your estimate, then output ONLY the final updated vector "
            f"in the exact format of <answer>{guess_fmt}</answer> (comma-separated numbers)."
        )
        self.guess_inst = (
            "\nThe next round is **Guess Round**.\n"
            "Briefly reason (only essential reasoning in at most 2-3 sentences) about how the feedback updates the estimate in <scratch>.\n"
            "Then provide your best decisive estimate of the user's preference vector in:\n"
            "<interact>\n"
            f"Guess: {guess_fmt}\n"
            "</interact>\n"
            "NOTE: Use the feedback to make clear relative distinctions across dimensions when supported by evidence.\n"
            "Do not keep values near 0.5 unless the evidence truly remains ambiguous.\n"
        )
        if version == "full":
            self.action_inst = (
                "\nThe next round is **Action Round**.\n"
                "Briefly Reason (only essential reasoning in at most 2-3 sentences.) about what to ask in <scratch> blocks. Ask in the format of \n"
                "<interact>\n"
                "Pair: p1,p2\n"
                "</interact>\n"
                "**NOTE: Do NOT repeat any previous question.**\n"
            )
        else:
            self.action_inst = (
                "\nThe next round is **Action Round**.\n"
                "Briefly Reason (only essential reasoning in at most 2-3 sentences.) about what to ask in <scratch> blocks. Ask in the format of \n"
                "<interact>\n"
                f"Focus: {focus_fmt}\n"
                "Pair: p1,p2\n"
                "</interact>\n"
                "**NOTE: Do NOT repeat any previous question.**\n"
            )

    def compute_feedback(self, content: str, step: int) -> str:
        self.query_counts += 1
        if step % 2 == 1:
            try:
                feedback = "The feedback of your latest query: " + self._compute_feedback(content)
            except Exception as exc:
                print(exc)
                self.cut = 1.0
                self.asp1_record_yij.append(None)
                self.asp1_record_mij.append(None)
                self.asp1_record_focus.append(None)
                self.all_queries.append(None)
                feedback = "The last query you made is invalid, so there is no user feedback this Action Round."

            if step == self.max_counts - 1:
                return feedback + f"\n{self.answer_instruct}\n"
            return feedback + self.guess_inst

        try:
            cur_pref = self._extract_guess(content)
            feedback = ""
        except Exception as exc:
            print(exc)
            self.cut = 1.0
            cur_pref = None
            feedback = "The estimate you made is invalid."
        self.asp1_record_vec.append(cur_pref)
        return feedback + self.action_inst

    def check_asinfo(self, validate: bool = False) -> int:
        del validate
        if self.asp1_record_mij[-1] is None or self.asp1_record_focus[-1] is None or self.asp1_record_vec[-1] is None:
            self.all_acinfo.append(-1)
            return -1
        if len(self.all_queries) > 1 and self.all_queries[-1] in self.all_queries[:-1]:
            self.all_acinfo.append(-1)
            return -1
        mij = self.asp1_record_mij[-1]
        eps = 1e-5
        as_label = -1 if np.all(mij >= eps) or np.all(mij <= -eps) else 1
        self.all_acinfo.append(as_label)
        return as_label

    def check_margin(self, validate: bool = False) -> int:
        del validate
        v_t = self.asp1_record_vec[-1]
        if v_t is None:
            return -1
        mij = self.asp1_record_mij[-1]
        y = self.asp1_record_yij[-1]
        if any(x is None for x in (mij, y)):
            return 0
        if self.all_acinfo and self.all_acinfo[-1] == -1:
            return 0
        last_valid = [item for item in self.asp1_record_vec if item is not None][-2]
        margin = self._cosine(v_t, self.user_pref) - self._cosine(last_valid, self.user_pref)
        return 1 if margin > 0 else -1

    def _compute_feedback(self, content: str) -> str:
        p1, p2 = self._extract_focus(content, target="Pair")
        x1 = np.array(self.seen_movies_scores[p1])
        x2 = np.array(self.seen_movies_scores[p2])
        w = np.array(self.user_pref)

        if self.version == "gated3":
            ks = self._extract_focus(content, target="Focus", n_dims=3)
            score1 = sum(w[k] * x1[k] for k in ks)
            score2 = sum(w[k] * x2[k] for k in ks)
            mij = x1[list(ks)] - x2[list(ks)]
        elif self.version == "gated2":
            ks = self._extract_focus(content, target="Focus", n_dims=2)
            score1 = sum(w[k] * x1[k] for k in ks)
            score2 = sum(w[k] * x2[k] for k in ks)
            mij = x1[list(ks)] - x2[list(ks)]
        else:
            ks = (0,)
            score1 = np.dot(w, x1)
            score2 = np.dot(w, x2)
            mij = x1 - x2

        eps = 1e-12
        if score1 > score2 + eps:
            feedback, y = "Yes.", 1
        elif score2 > score1 + eps:
            feedback, y = "No.", -1
        else:
            feedback, y = "Equal.", 0

        self.asp1_record_yij.append(y)
        self.asp1_record_mij.append(mij)
        self.asp1_record_focus.append(ks)

        action = (p1, p2, ks)
        if action in self.all_queries:
            self.repeat_counts += 1
            self.cut = 1.0
        self.all_queries.append(action)
        return feedback

    def _extract_guess(self, guess: str):
        numbers = re.findall(r"([\d\.]+)", guess)
        assert len(numbers) == self.dims, f"Expected {self.dims} dim vector. Output {len(numbers)}"
        return np.array(numbers, dtype=float)

    def _extract_focus(self, content: str, target: str = "Focus", n_dims: int = 2):
        if target == "Focus":
            max_idx = len(self.user_pref)
        elif target == "Pair":
            max_idx = len(self.seen_movies_scores)
        else:
            raise ValueError(f"Unknown target: {target}")

        if n_dims == 2:
            pattern = rf"^\s*{target}:\s*(\d+)\s*,\s*(\d+)\s*$"
        else:
            pattern = rf"^\s*{target}:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$"
        match = re.search(pattern, content, flags=re.MULTILINE)
        assert match, f"Failed to strictly extract {target} with n_dims={n_dims}: {content}"
        idxs = [int(match.group(i)) for i in range(1, n_dims + 1)]
        for value in idxs:
            assert 1 <= value <= max_idx, f"{target} index out of range: {value} (max={max_idx})"
        assert len(set(idxs)) == len(idxs), f"{target} indices must be distinct, got {idxs}"
        return tuple(sorted(value - 1 for value in idxs))

    @staticmethod
    def _cosine(vec1, vec2) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


class MediQSpace:
    unknown_reply = "I cannot answer this question, please do not ask this question again."

    def __init__(self, controller, max_turns: int):
        self.controller = controller
        self.max_counts = max_turns
        self.query_counts = 0
        self.cut = 0
        self.all_feedbacks = []
        self.all_unique_feedbacks = set()
        self.all_queries = []
        self.all_queries_cf = []
        self.all_estimates = [np.array([0.5, 0.5, 0.5, 0.5])]
        self.all_estimates_cf = [np.array([0.5, 0.5, 0.5, 0.5])]
        self.ground_truth = {"A": 0, "B": 1, "C": 2, "D": 3}[controller["answer_idx"]]
        self.answer_instruct = (
            "**Final Turn**: Briefly think in <scratch>...</scratch> "
            "about how the latest feedback updates your estimate, then output ONLY the final updated vector "
            "in the exact format of <answer>wA,wB,wC,wD</answer>, where wA,wB,wC,wD must be comma-separated numbers."
        )

    def generate_prompt(self, query: str) -> str:
        return MEDIQ_PATIENT_PROMPT.format(
            atom_facts="\n".join(self.controller["atomic_facts"]),
            query=query,
        )

    def update_query(self, value, step: int) -> None:
        self.query_counts += 1
        if step % 2 == 0:
            self.all_estimates.append(value)
        else:
            self.all_queries.append(value)

    def update_feedback(self, feedback, step: int) -> str:
        if feedback is None:
            feedback = ""
        else:
            feedback = "The feedback of your latest query: " + self.convert_patient_response(feedback)
        if step % 2 == 1:
            self.all_feedbacks.append(feedback)

        if (step % 2 == 1 and self.all_queries[-1] is None) or (step % 2 == 0 and self.all_estimates[-1] is None):
            feedback = "The previous action is invalid. If you want to make an interaction, you should put it between <interact> and </interact>.\n"

        if step == self.max_counts - 1:
            return feedback + f"\n{self.answer_instruct}\n"
        if step % 2 == 1:
            return feedback + (
                "\nThe next round is **Guess Round**.\n"
                "Briefly Reason (only essential reasoning in at most 2-3 sentences.) about how the feedback supports or weakens hypotheses in <scratch> blocks. Update the state vector in\n"
                "<interact>\n"
                "Guess: wA,wB,wC,wD\n"
                "</interact>\n"
            )
        return feedback + (
            "\nThe next round is **Action Round**.\n"
            "Briefly Reason (only essential reasoning in at most 2-3 sentences.) about what to ask in <scratch> blocks. Ask ONE atomic query to reduce uncertainty among the hypotheses in\n"
            "<interact>\n"
            "Query: ...\n"
            "</interact>\n"
        )

    def check_asinfo(self, validate: bool = False) -> int:
        del validate
        if len(self.all_queries) > 1 and self.all_queries[-1] in self.all_queries[:-1]:
            return -1
        feedback = self.all_feedbacks[-1]
        if feedback == "" or self.unknown_reply in feedback:
            return -1
        if len(self.all_queries_cf) == 0:
            return 1
        if self.all_queries_cf[-1] is None:
            return 0
        return 1 if token_f1(self.all_queries_cf[-1], self.all_queries[-1]) < 0.6 else -1

    def check_margin(self, validate: bool = False) -> int:
        del validate
        if self.all_estimates[-1] is None:
            return -1
        last_valid = [item for item in self.all_estimates if item is not None][-2]
        if self.all_feedbacks[-1] == "" or self.unknown_reply in self.all_feedbacks[-1]:
            margin = self._margin(self.all_estimates[-1]) - self._margin(last_valid)
            return 1 if margin == 0 else -1
        if self.all_estimates_cf[-1] is None:
            return -1
        v_t, v_cf, v_tm1 = self.all_estimates[-1], self.all_estimates_cf[-1], last_valid
        return 1 if np.linalg.norm(v_cf - v_tm1) <= np.linalg.norm(v_t - v_tm1) else -1

    def extract_query(self, content: str, step: int):
        if step % 2 == 0:
            try:
                return self._extract_guess(content)
            except Exception as exc:
                print(exc)
                return None
        return self._extract_query(content)

    def counterfactual(self, value, step: int) -> None:
        if step % 2 == 0:
            self.all_estimates_cf.append(value)
        else:
            self.all_queries_cf.append(value)

    def gen_counterfactual(self) -> str:
        weights = np.full(4, 0.3)
        last_valid = [item for item in self.all_estimates if item is not None][-1]
        best_idx = np.where(last_valid == np.max(last_valid))[0].tolist()
        candidates = [i for i in range(4)] if len(best_idx) == 4 else [i for i in range(4) if i not in best_idx]
        weights[int(np.random.choice(candidates))] = 0.7
        w_a, w_b, w_c, w_d = weights
        self.all_estimates_cf.append(weights)
        return f"<interact>\nGuess: {w_a},{w_b},{w_c},{w_d}\n</interact>\n"

    def counterfactual_feedback_observation(self) -> str:
        return "The feedback of your latest query: " + self.unknown_reply

    def convert_patient_response(self, response: str) -> str:
        text = response.strip()
        if re.search(r"\bunknown\b", text, re.IGNORECASE):
            return self.unknown_reply
        indices = re.findall(r"\b\d+\b", text)
        if not indices:
            return self.unknown_reply
        selected = []
        for idx_str in indices:
            idx = int(idx_str)
            if 1 <= idx <= len(self.controller["atomic_facts"]) and idx not in self.all_unique_feedbacks:
                selected.append(self.controller["atomic_facts"][idx - 1])
                self.all_unique_feedbacks.add(idx)
        if not selected:
            return self.unknown_reply
        return "\n".join(selected)

    def _extract_guess(self, guess: str, target_num: int = 4):
        numbers = re.findall(r"([\d\.]+)", guess)
        assert len(numbers) == target_num, f"Expected {target_num} dim vector. Output {len(numbers)}"
        return np.array(numbers, dtype=float)

    def _extract_query(self, text: str) -> str | None:
        match = re.search(r"Query:\s*(.*)", text)
        return match.group(1).strip() if match else None

    def _margin(self, weights) -> float:
        return float(weights[self.ground_truth] - max(weights[j] for j in range(len(weights)) if j != self.ground_truth))


class FloDialSpace:
    def __init__(self, controller, max_turns: int, btv: str = "v1"):
        self.controller = controller
        self.max_counts = max_turns
        self.btv = btv
        self.query_counts = 0
        self.cut = 0
        self.dims = len(controller["candidates"])
        self.ground_truth = controller["standard_index"]
        self.all_feedbacks = []
        self.all_unique_feedbacks = set()
        self.all_queries = []
        self.all_estimates = [np.array([0.5] * self.dims)]
        self.eg = ",".join([f"w{i + 1}" for i in range(self.dims)])
        self.answer_instruct = (
            "**Final Turn**: Briefly think in <scratch>...</scratch> "
            "about how the latest feedback updates your estimate, then output ONLY the final updated vector "
            f"in the exact format of <answer>{self.eg}</answer> (comma-separated numbers)."
        )

    def generate_prompt(self, query: str) -> str:
        return FLODIAL_USER_PROMPT.format(
            task_description=self.controller["task_description"],
            reference_table="\n".join(
                [f"{idx}. {item['utterance']} -> {item['answer']}" for idx, item in enumerate(self.controller["all_questions"])]
            ),
            query=query,
        )

    def update_query(self, value, step: int) -> None:
        self.query_counts += 1
        if step % 2 == 0:
            self.all_estimates.append(value)
        else:
            self.all_queries.append(value)

    def update_feedback(self, feedback, step: int) -> str:
        if feedback is None:
            feedback = ""
        else:
            feedback = "The feedback of your latest query: " + self.convert_user_response(feedback)
        if step % 2 == 1:
            self.all_feedbacks.append(feedback)

        if (step % 2 == 1 and self.all_queries[-1] is None) or (step % 2 == 0 and self.all_estimates[-1] is None):
            feedback = "The previous action is invalid. If you want to make an interaction, you should put it between <interact> and </interact>.\n"

        if step == self.max_counts - 1:
            return feedback + f"\n{self.answer_instruct}\n"
        if step % 2 == 1:
            return feedback + (
                "\nThe next round is **Guess Round**.\n"
                "Briefly Reason (only essential reasoning in at most 2-3 sentences.) about how the feedback supports or weakens hypotheses in <scratch> blocks. Update the state vector in\n"
                "<interact>\n"
                f"Guess: {self.eg}\n"
                "</interact>\n"
            )
        return feedback + (
            "\nThe next round is **Action Round**.\n"
            "Briefly Reason (only essential reasoning in at most 2-3 sentences.) about what to ask in <scratch> blocks. Ask ONE atomic query to reduce uncertainty among the hypotheses in\n"
            "<interact>\n"
            "Query: ...\n"
            "</interact>\n"
        )

    def check_asinfo(self, validate: bool = False) -> int:
        del validate
        feedback = self.all_feedbacks[-1]
        if "yes" in feedback:
            return 1
        if " no" in feedback:
            return 0
        return -1

    def check_margin(self, validate: bool = False) -> int:
        del validate
        if self.all_estimates[-1] is None:
            return -1
        last_valid = [item for item in self.all_estimates if item is not None][-2]
        if "yes" in self.all_feedbacks[-1] or " no" in self.all_feedbacks[-1]:
            return 1 if self.all_estimates[-1][self.ground_truth] > last_valid[self.ground_truth] else -1
        if self.btv == "v0":
            return 0
        return 1 if np.allclose(self.all_estimates[-1], last_valid, rtol=1e-6) else -1

    def extract_query(self, content: str, step: int):
        if step % 2 == 0:
            try:
                return self._extract_guess(content)
            except Exception as exc:
                print(exc)
                return None
        return self._extract_query(content)

    def convert_user_response(self, response: str) -> str:
        yesno_pattern = re.compile(r"^\s*(Yes|No)\s*,\s*([0-9]+)\s*$", re.IGNORECASE)
        unknown_pattern = re.compile(r"^\s*Unknown\s*$", re.IGNORECASE)
        text = "" if response is None else response.strip()
        if unknown_pattern.match(text):
            return "Unknown."
        match = yesno_pattern.match(text)
        if match:
            yn = match.group(1).lower()
            qid = int(match.group(2))
            if qid in self.all_unique_feedbacks:
                return "Unknown\nYou seem to be repeating a previously raised question; returning Unknown."
            self.all_unique_feedbacks.add(qid)
            return yn
        return "Unknown."

    def _extract_guess(self, guess: str):
        numbers = re.findall(r"([\d\.]+)", guess)
        assert len(numbers) == self.dims, f"Expected {self.dims} dim vector. Output {len(numbers)}"
        return np.array(numbers, dtype=float)

    def _extract_query(self, text: str) -> str | None:
        match = re.search(r"Query:\s*(.*)", text)
        return match.group(1).strip() if match else None
