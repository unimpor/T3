# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from collections import Counter
from typing import List
import itertools

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1_scores(prediction: List[str], ground_truth: List[str], **kwargs) -> float:
    """
    Calculate F1 score between prediction and ground truth lists.
    
    Args:
        prediction: Predicted list of items
        ground_truth: Ground truth list of items
        **kwargs: Additional keyword arguments (ignored)
        
    Returns:
        F1 score as float
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def validate_format(prompt, response):
    """
    validate the template format
    return: (is valid)
    """
    if '<refine>' in prompt:
        token_list = ['think', 'search', 'refine', 'answer']
    else:
        token_list = ['think', 'search', 'answer']

    if not response:
        return 0

    for special_tags in token_list:
        start_token = f"<{special_tags}>"
        end_token = f"</{special_tags}>"
        start_count = response.count(start_token)
        end_count = response.count(end_token)
        if start_count != end_count:
            return 0
        if start_count == 0:
            return 0
    return 1

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def cover_em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def extract_information(responses_str):
    """Extract and concatenate information from <documents> tags, skipping the first."""
    info_pattern = r'<documents>(.*?)</documents>'
    matches = re.findall(info_pattern, responses_str, re.DOTALL)
    
    if len(matches) <= 1:
        return None
    
    # Concatenate from the second match onward
    combined_info = ' '.join(matches[1:]).strip()
    return combined_info

def extract_information_list(responses_str):
    """Extract and concatenate information from <documents> tags, skipping the first."""
    info_pattern = r'<documents>(.*?)</documents>'
    matches = re.findall(info_pattern, responses_str, re.DOTALL)
    
    if len(matches) <= 1:
        return None
    matches = matches[1:]
    return matches

def extract_refine(responses_str):
    info_pattern = r'<refine>(.*?)</refine>'
    matches = re.findall(info_pattern, responses_str, re.DOTALL)
    
    if len(matches) == 0:
        return None
    
    # Concatenate from the second match onward
    combined_info = ' '.join(matches).strip()
    return combined_info

def extract_solution(responses_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, responses_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def compute_score_format(responses_str, ground_truth):
    format_validity = validate_format(responses_str, responses_str)
    return format_validity

def compute_reward(solution_str, responses_str, ground_truth, format_score=0., score=1., refine_score=0.0, do_print_frac=-1, score_func=em_check):
    answer = extract_solution(responses_str)
    do_print = random.randint(1, do_print_frac) == 1 if do_print_frac > 0 else False
    
    if do_print:
        print(f"--------------Begin Case--------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"--------------End Case--------------")

    if answer is None:
        return 0
    else:
        answer_score = score_func(answer, ground_truth['target'])
        format_validity = validate_format(solution_str, responses_str)
        refine_subem = compute_refine_score_subem(responses_str, ground_truth)

        if answer_score > 0:
            return answer_score
        else:
            score = 0.0
            if format_validity:
                score += format_score
            if refine_subem > 0:
                score += refine_score
            return score

def compute_score_em(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return em_check(answer, ground_truth)

def compute_score_f1(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return compute_f1_scores(answer.split(), ground_truth.split())

def compute_score_f1_char(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return compute_f1_scores(answer, ground_truth)

def compute_score_cem(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return cover_em_check(answer, ground_truth['target'])


def compute_information_score_subem(responses_str, ground_truth):
    information = extract_information(responses_str)
    
    if information is None:
        return 0.0
    elif 'no' in ground_truth['target'] or 'yes' in ground_truth['target']:
        return 0.5
    else:
        return cover_em_check(information, ground_truth['target'])

def compute_information_reverse_rank(responses_str, ground_truth):
    doc_list = extract_information_list(responses_str)
    info_score = 0.0
    
    if doc_list is None:
        return 0.0
    elif 'no' in ground_truth['target'] or 'yes' in ground_truth['target']:
        return 0.5
    else:
        for idx, doc in enumerate(doc_list):
            if cover_em_check(doc, ground_truth['target']):
                info_score += 1 / float(idx + 1)
    return info_score

def compute_refine_score_subem(responses_str, ground_truth):
    refined_info = extract_refine(responses_str)
    if refined_info is None:
        return 0.0
    else:
        return cover_em_check(refined_info, ground_truth['target'])



def eval_circuit(expr: str, num_inputs: int = 3) -> str:
    results = []
    for bits in itertools.product([0, 1], repeat=num_inputs):
        env = {f"x{i}": bool(bits[i]) for i in range(num_inputs)}
        val = eval(expr, {"__builtins__": None}, env)
        results.append("1" if val else "0")
    return "".join(results)


def compute_circuit_accuracy(responses_str: str, ground_truths) -> float:

    num_inputs = ground_truths['num_inputs']
    num_circuits = ground_truths['num_circuits']
    joint_lookup = ground_truths['target']
    candidates = ground_truths['candidates']

    answer = extract_solution(responses_str)
    if not answer:
        return 0.0

    # pattern = r'(\d+)\s*[,]?\s*(\d+)'
    # match = re.search(pattern, answer)
    numbers = re.findall(r'\d+', answer)
    # if not match:
    #     print("Not a valid expression:" + answer)
    #     return 0.
    if len(numbers) != num_circuits:
        print("Not a valid expression:" + answer)
        return 0.
    # return int(match.group(1)), int(match.group(2))
    if max(int(i) for i in numbers) > len(candidates):
        print("Not a valid expression:" + answer)
        return 0.
    circuits = [candidates[int(i)-1] for i in numbers]
    chunk_size = 2 ** num_inputs
    correct = 0

    for i in range(num_circuits):
        formula = circuits[i]
        gt_table = joint_lookup[i*chunk_size:(i+1)*chunk_size]

        try:
            pred_table = eval_circuit(formula, num_inputs)
        except Exception:
            print("Not a valid expression." + formula)
            continue

        if pred_table == gt_table:
            correct += 1

    return float(correct / num_circuits)


def compute_rank(responses_str, ground_truths):

    unseen_names = ground_truths['unseen_names'].tolist()
    sort_unseen = ground_truths['sort_unseen']

    answer = extract_solution(responses_str)
    if not answer:
        return 0.
    
    if answer not in unseen_names:
        print("Illegal answer name.")
        return 0.
    
    reward = sort_unseen[unseen_names.index(answer)]

    return 1. if reward == len(sort_unseen) else 0.


def compute_circuit_accuracy__(responses_str: str, ground_truths) -> float:

    num_inputs = ground_truths['num_inputs']
    num_circuits = ground_truths['num_circuits']
    joint_lookup = ground_truths['target']

    answer = extract_solution(responses_str)
    if not answer:
        return 0.0
    
    chunk_size = 2 ** num_inputs

    expected_length = num_circuits * chunk_size

    if len(answer) != expected_length:
        return 0.0

    correct = 0
    for i in range(num_circuits):
        pred_chunk = answer[i*chunk_size:(i+1)*chunk_size]
        gt_chunk = joint_lookup[i*chunk_size:(i+1)*chunk_size]
        if pred_chunk == gt_chunk:
            correct += 1

    return correct / num_circuits

import numpy as np

def _extract_guess(guess, target_num):
    vec_pattern = r'([\d\.]+)'
    numbers = re.findall(vec_pattern, guess)
    assert len(numbers) == target_num, f"Expected {target_num} dim vector. Output {len(numbers)}"
    assert all(float(k)>=0 for k in numbers), f"All guess must be positive. Ouput {numbers}"

    return np.array(numbers, dtype=float)


def _extract_guess_any(guess):
    numbers = re.findall(r'([\d\.]+)', guess)
    assert len(numbers) > 0, "Expected a non-empty vector."
    assert all(float(k) >= 0 for k in numbers), f"All guess must be positive. Ouput {numbers}"
    return np.array(numbers, dtype=float)


def _target_index(target):
    if hasattr(target, "item"):
        target = target.item()
    ans_maps = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    if isinstance(target, str):
        target = target.strip()
        if target in ans_maps:
            return ans_maps[target]
        return int(target)
    return int(target)


def compute_pe_g(responses_str, ground_truths):
    return compute_sim(responses_str, ground_truths)


def compute_pe_f(responses_str, ground_truths):
    return compute_sim(responses_str, ground_truths)


def compute_score_multi_opt(responses_str, ground_truths):
    target = ground_truths.get('target', ground_truths.get('answer_idx'))
    answer = _target_index(target)

    resp = extract_solution(responses_str)
    if not resp:
        return 0.

    try:
        pred = _extract_guess_any(resp)
    except Exception as e:
        print(e)
        return 0.

    if answer >= len(pred):
        return 0.
    max_indices = np.where(pred == np.max(pred))[0]
    return 1. if len(max_indices) == 1 and max_indices[0] == answer else 0.


def compute_score_multi_opt_int(responses_str, ground_truths):
    return compute_score_multi_opt(responses_str, ground_truths)


def compute_sim(responses_str, ground_truths):
    target = np.array(ground_truths['target'])

    answer = extract_solution(responses_str)
    if not answer:
        return 0.
    
    try:
        pred = _extract_guess(answer, len(target))
    except:
        print("Illegal answer name.")
        return 0.
    cos_sim = np.dot(pred, target) / (np.linalg.norm(pred) * np.linalg.norm(target))
    cos_sim = (cos_sim + 1) / 2
    return 1. if cos_sim > 0.94 else 0.
