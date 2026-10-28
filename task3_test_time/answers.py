"""Fixed answer rules; candidate grouping never inspects the reference answer.

MathVista follows released evaluation/calculate_score.py (choice edit distance,
integer conversion and precision rounding), with letters as canonical MC labels
and missing extractions retained as NO_ANSWER as required by the PA3 manual.
"""
from functools import lru_cache
import atexit
import importlib.metadata
import math
import multiprocessing
import re
import time

from task3_test_time.selection import NO_ANSWER

SCORER_VERSION = "pa3-answer-rules-v2-bounded"
MATH_VERIFY_VERSION = "0.8.0"
COMPARISON_TIMEOUT_SECONDS = 10
MATHVISTA_SOURCE = "https://github.com/lupantech/MathVista/blob/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb/evaluation/calculate_score.py"
_SCORER = None
_DIAGNOSTICS = []
_CONTEXT = {}


def missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def last_box(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip() or None
    return None


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def normalize_mathvista(value, metadata):
    if missing(value):
        return NO_ANSWER
    text = str(value).strip()
    if metadata["question_type"] == "multi_choice":
        choices = [str(choice) for choice in metadata["choices"]]
        if not choices:
            raise ValueError("Multi-choice question has no choices")
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        match = re.findall(r"\(([a-zA-Z])\)", text)
        if match:
            text = match[0].upper()
        if text in letters:
            return text
        index = min(range(len(choices)), key=lambda i: edit_distance(text, choices[i]))
        return letters[index]
    kind = metadata["answer_type"]
    try:
        if kind == "integer":
            return str(int(float(text)))
        if kind == "float":
            value = round(float(text), int(metadata["precision"]))
            return str(value) if math.isfinite(value) else NO_ANSWER
        if kind == "list":
            return text
    except (ValueError, TypeError, OverflowError):
        return NO_ANSWER
    raise ValueError(f"Undocumented MathVista answer type: {kind}")


@lru_cache(maxsize=65536)
def parsed_math(value):
    from math_verify import LatexExtractionConfig, parse
    # Supplied extracted strings are expressions, not full response text. Wrap
    # them in a math environment so LaTeX fractions/sets are parsed faithfully.
    return parse("$" + value.strip("$") + "$", extraction_config=[LatexExtractionConfig()],
                 parsing_timeout=None)


def _unbounded_equivalent(a, b, progress=lambda stage, expression: None):
    if missing(a) or missing(b) or a == NO_ANSWER or b == NO_ANSWER:
        return False
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    from math_verify import verify
    progress("parse_a", a)
    pa = parsed_math(a)
    progress("parse_b", b)
    pb = parsed_math(b)
    if not pa or not pb:
        return False
    # No reference is designated while grouping: accept either representation
    # direction for interval/inequality equivalence, identically when grading.
    progress("verify", None)
    return bool(verify(pa, pb, timeout_seconds=None) or verify(pb, pa, timeout_seconds=None))


def _symbolic_worker(connection):
    # Disable the Windows package's nested-process timeout only inside this
    # isolated worker. The parent enforces the wall-clock bound instead.
    import logging
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    import math_verify
    connection.send("ready")
    try:
        while True:
            pair = connection.recv()
            if pair is None:
                break
            try:
                progress = lambda stage, expression: connection.send(("stage", stage, expression))
                connection.send(("result", bool(_unbounded_equivalent(*pair, progress=progress)), None))
            except Exception as error:
                connection.send(("result", False, repr(error)))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class _ScorerWorker:
    def __init__(self, seconds=COMPARISON_TIMEOUT_SECONDS, target=_symbolic_worker):
        self.seconds = seconds
        self.target = target
        self.process = None
        self.connection = None
        self.failed_parses = set()

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            self.process.close()
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None

    def compare(self, a, b):
        if a in self.failed_parses or b in self.failed_parses:
            _DIAGNOSTICS.append({**_CONTEXT, "a": a, "b": b,
                                 "status": "cached_parse_timeout", "decision": False})
            return False
        if self.process is None:
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(target=self.target, args=(child,), daemon=True)
            self.process.start()
            child.close()
            if not self.connection.poll(60):
                self.close()
                raise RuntimeError("Symbolic scorer worker failed to start")
            if self.connection.recv() != "ready":
                self.close()
                raise RuntimeError("Invalid scorer startup handshake")
        try:
            self.connection.send((a, b))
            deadline = time.monotonic() + self.seconds
            stage, expression = "comparison", None
            while True:
                if not self.connection.poll(max(0, deadline - time.monotonic())):
                    _DIAGNOSTICS.append({**_CONTEXT, "a": a, "b": b, "status": "timeout",
                                         "stage": stage, "seconds": self.seconds, "decision": False})
                    if stage.startswith("parse") and expression is not None:
                        self.failed_parses.add(expression)
                    self.close()
                    return False
                message = self.connection.recv()
                if message[0] == "stage":
                    _, stage, expression = message
                    continue
                if message[0] != "result":
                    raise RuntimeError("Invalid symbolic worker response")
                _, value, error = message
                break
            if error:
                _DIAGNOSTICS.append({**_CONTEXT, "a": a, "b": b, "status": "error", "error": error, "decision": False})
            return bool(value)
        except (EOFError, BrokenPipeError, OSError) as error:
            _DIAGNOSTICS.append({**_CONTEXT, "a": a, "b": b, "status": "worker_failure", "error": repr(error), "decision": False})
            self.close()
            return False


def close_scorer():
    global _SCORER
    if _SCORER is not None:
        _SCORER.close()
        _SCORER = None


def scorer_context(dataset, sample_id):
    _CONTEXT.update(dataset=dataset, sample_id=sample_id)


def scorer_diagnostics():
    return list(_DIAGNOSTICS)


def reset_scorer():
    close_scorer()
    _DIAGNOSTICS.clear()
    _CONTEXT.clear()
    math_equivalent.cache_clear()
    parsed_math.cache_clear()


@lru_cache(maxsize=131072)
def math_equivalent(a, b):
    global _SCORER
    if missing(a) or missing(b) or a == NO_ANSWER or b == NO_ANSWER:
        return False
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    if _SCORER is None:
        _SCORER = _ScorerWorker()
    return _SCORER.compare(a, b)


atexit.register(close_scorer)


def answer_groups(extractions, dataset, metadata=None):
    if dataset == "mathvista":
        return [normalize_mathvista(value, metadata) for value in extractions]
    representatives = []
    groups = []
    for value in extractions:
        if missing(value):
            groups.append(NO_ANSWER)
            continue
        value = str(value).strip()
        group = None
        for representative in representatives:
            if math_equivalent(value, representative):
                group = representative
                break
        if group is None:
            representatives.append(value)
            group = value
        groups.append(group)
    return groups


def is_correct(group, gold, dataset, metadata=None):
    if group == NO_ANSWER:
        return False
    if dataset == "mathvista":
        return group == normalize_mathvista(gold, metadata)
    return math_equivalent(group, gold)


def scorer_manifest():
    installed = importlib.metadata.version("math-verify")
    if installed != MATH_VERIFY_VERSION:
        raise RuntimeError(f"Scorer must be pinned to math-verify=={MATH_VERIFY_VERSION}; found {installed}")
    return {"rules": SCORER_VERSION, "math-verify": installed,
            "sympy": importlib.metadata.version("sympy"),
            "latex2sympy2_extended": importlib.metadata.version("latex2sympy2_extended"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "mathvista_source": MATHVISTA_SOURCE, "missing": NO_ANSWER,
            "grouping": "source-rollout-order, representative-equivalence; no gold consulted",
            "math_verify_timeouts": "persistent isolated symbolic worker; timeout/error returns False and is logged",
            "comparison_timeout_seconds": COMPARISON_TIMEOUT_SECONDS}
