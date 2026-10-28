"""Gold-free selection and stopping. Inputs are revealed candidate prefixes only."""
from collections import defaultdict
import math
import numpy as np

NO_ANSWER = "NO_ANSWER"


def scores(candidates):
    if not candidates:
        raise ValueError("Empty candidate set")
    kl = [np.asarray(c["kl_u_p"], dtype=np.float64) for c in candidates]
    logp = [np.asarray(c["logp"], dtype=np.float64) for c in candidates]
    if any(len(k) == 0 or len(k) != len(p) for k, p in zip(kl, logp)):
        raise ValueError("Empty or misaligned probability traces")
    if not all(np.isfinite(x).all() for x in kl + logp):
        raise ValueError("Non-finite probability trace")
    out = {"logp": np.array([x.mean() for x in logp]),
           "certainty": np.array([x.mean() for x in kl]),
           "shortest": -np.array([len(x) for x in kl])}
    if "img_attn_total" in candidates[0]:
        lower, upper = np.quantile(np.concatenate(kl), [.05, .95], method="linear")
        confidence = [np.clip((x - lower) / max(upper - lower, 1e-8), 0, 1) for x in kl]
        vision = []
        for candidate, trace in zip(candidates, kl):
            arrays = [np.asarray(candidate[k], dtype=np.float64) for k in
                      ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total")]
            if any(len(a) != len(trace) or not np.isfinite(a).all() or (a < 0).any() for a in arrays):
                raise ValueError("Invalid visual attention trace")
            total = sum(arrays)
            if (total <= 0).any():
                raise ValueError("Cannot normalize zero eligible attention")
            vision.append(arrays[0] / total)
        out["vision"] = np.array([x.mean() for x in vision])
        out["visconf"] = np.array([v.mean() * c.mean() for v, c in zip(vision, confidence)])
        out["token_fusion"] = np.array([(v * c).mean() for v, c in zip(vision, confidence)])
    return out


def rank_weights(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 1:
        return np.ones(1)
    # Average descending ranks; no arbitrary ordering inside tied score groups.
    ranks = np.array([1 + np.count_nonzero(values > x) + (np.count_nonzero(values == x) - 1) / 2
                      for x in values])
    return (n - ranks) / (n - 1)


def votes(candidates, weights=None):
    weights = np.ones(len(candidates)) if weights is None else weights
    totals = defaultdict(float)
    for candidate, weight in zip(candidates, weights):
        totals[candidate["answer_group"]] += float(weight)
    # dict preserves first supporting rollout, resolving vote ties as required.
    maximum = max(totals.values())
    winner = next(answer for answer, value in totals.items()
                  if math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12))
    ordered = sorted(totals.values(), reverse=True)
    tied = len(ordered) > 1 and math.isclose(ordered[0], ordered[1], rel_tol=0.0, abs_tol=1e-12)
    margin = 0.0 if tied else (ordered[0] - (ordered[1] if len(ordered) > 1 else 0)) / max(sum(ordered), 1e-8)
    unique = not tied
    return winner, margin, totals, unique


def select(candidates, method):
    if method == "first":
        return candidates[0]["answer_group"]
    if method == "consistency":
        return votes(candidates)[0]
    quality = method.removesuffix("_rwc")
    values = scores(candidates)[quality]
    if method.endswith("_rwc"):
        return votes(candidates, rank_weights(values))[0]
    return candidates[int(np.argmax(values))]["answer_group"]


def adaptive(candidates, tau, gamma=None, quality="certainty"):
    for n in (4, 8, 16):
        prefix = candidates[:n]
        answer, _, totals, unique = votes(prefix)
        if not unique or answer == NO_ANSWER or totals[answer] / n < tau:
            continue
        if gamma is not None:
            ranked, margin, _, _ = votes(prefix, rank_weights(scores(prefix)[quality]))
            if ranked != answer or margin < gamma:
                continue
        return answer, n
    return votes(candidates)[0], len(candidates)


def paired_interval(differences, repetitions=1000, seed=6304):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap question-level differences")
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)]
    return np.quantile(means, [.025, .975]).tolist()
