"""VQAv2 answer normalization and pair-level metrics.

Normalization follows the official VQA evaluator's punctuation, number, article,
and contraction rules: https://github.com/GT-Vision-Lab/VQA/blob/master/
PythonEvaluationTools/vqaEvaluation/vqaEval.py (Aishwarya Agrawal et al.).
The soft score is implemented from the assignment's leave-one-out equation.
"""
from collections import defaultdict
from functools import lru_cache
import re

import numpy as np

_NUMBERS = dict(zip("none zero one two three four five six seven eight nine ten".split(),
                    "0 0 1 2 3 4 5 6 7 8 9 10".split()))
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "im": "i'm", "ive": "i've",
    "id've": "i'd've", "idve": "i'd've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
_PUNCTUATION = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']


@lru_cache(maxsize=65536)
def normalize_answer(value):
    text = str(value or "").replace("\n", " ").replace("\t", " ").strip()
    original = text
    for punctuation in _PUNCTUATION:
        remove = punctuation + ' ' in original or ' ' + punctuation in original or re.search(r"\d,\d", original)
        text = text.replace(punctuation, '' if remove else ' ')
    text = re.sub(r"(?!<=\d)(\.)(?!\d)", "", text)
    words = [_NUMBERS.get(word, word) for word in text.lower().split()]
    return " ".join(_CONTRACTIONS.get(word, word) for word in words if word not in {"a", "an", "the"})


def vqa_score(prediction, references):
    if len(references) != 10:
        raise ValueError("Natural VQAv2 requires all ten human answers")
    prediction = normalize_answer(prediction)
    references = [normalize_answer(answer) for answer in references]
    if not prediction:
        return 0.0
    return sum(min(1.0, sum(prediction == other for j, other in enumerate(references) if j != i) / 3)
               for i in range(10)) / 10


def pair_bootstrap(values, repetitions=4000, seed=6304):
    """Values must contain one scalar per complete pair, including differences."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Pair-level bootstrap needs finite observations")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[indices].mean(axis=1)
    return np.quantile(estimates, [.025, .975]).tolist()


def summarize(records):
    pairs = defaultdict(list)
    for record in records:
        pairs[record["pair_id"]].append(record)
    if any(len(rows) != 2 for rows in pairs.values()):
        raise ValueError("Evaluation must retain complete complementary pairs")
    pair_rows = []
    for pair_id, rows in sorted(pairs.items()):
        predictions = [normalize_answer(row["prediction"]) for row in rows]
        canonical = [normalize_answer(row["answer"]) for row in rows]
        soft = [vqa_score(row["prediction"], row["answers"]) if row["kind"] == "natural"
                else float(bool(pred) and pred == answer)
                for row, pred, answer in zip(rows, predictions, canonical)]
        pair_rows.append({"pair_id": pair_id, "kind": rows[0]["kind"], "answer_type": rows[0]["answer_type"],
            "accuracy": sum(soft) / 2,
            "pair_accuracy": float(all(bool(p) and p == a for p, a in zip(predictions, canonical))),
            "identical_answer_rate": float(predictions[0] == predictions[1]),
            "empty_answer_rate": sum(not p for p in predictions) / 2,
            "reference_nll": sum(row["reference_nll"] for row in rows) / 2,
            "seconds_per_answer": sum(row["generation_seconds"] for row in rows) / 2,
            "partner_following_rate": sum(bool(predictions[i]) and predictions[i] == canonical[1-i] for i in range(2)) / 2})
    metrics = ("accuracy", "pair_accuracy", "identical_answer_rate", "empty_answer_rate",
               "reference_nll", "seconds_per_answer", "partner_following_rate")
    result = {}
    groups = {"all": pair_rows}
    groups.update({kind: [p for p in pair_rows if p["answer_type"] == kind]
                   for kind in sorted({p["answer_type"] for p in pair_rows})})
    for group, rows in groups.items():
        result[group] = {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                                  "ci95": pair_bootstrap([r[metric] for r in rows])}
                         for metric in metrics}
        result[group]["pairs"] = len(rows)
    return {"aggregate": result, "pairs": pair_rows}


def paired_advantages(correct, swapped):
    a = {row["pair_id"]: row for row in correct["pairs"]}
    b = {row["pair_id"]: row for row in swapped["pairs"]}
    if a.keys() != b.keys():
        raise ValueError("Conditions must use identical pair IDs")
    accuracy = [a[key]["accuracy"] - b[key]["accuracy"] for key in sorted(a)]
    nll = [b[key]["reference_nll"] - a[key]["reference_nll"] for key in sorted(a)]
    return {"accuracy_advantage": {"mean": float(np.mean(accuracy)), "ci95": pair_bootstrap(accuracy)},
            "image_swap_nll_penalty": {"mean": float(np.mean(nll)), "ci95": pair_bootstrap(nll)}}
