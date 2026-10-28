from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.io import sha256, write_json
from task2_vlm.data import natural_from_preflight, natural_pool, select_natural


class PreflightTests(unittest.TestCase):
    def test_selected_only_streaming_reproduces_ids_and_following_rng_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, pools = {}, {}
            for split, counts, offset in (("train", {"yes/no": 640, "number": 256, "other": 640}, 1),
                                           ("val", {"yes/no": 60, "number": 40, "other": 60}, 10000)):
                questions, annotations, pairs, files = [], [], [], {}
                qid = offset
                for kind, number in counts.items():
                    for _ in range(number):
                        pair_ids = [qid, qid + 1]
                        pairs.append(pair_ids)
                        for i, answer in zip(pair_ids, ("yes", "no") if kind == "yes/no" else ("1", "2") if kind == "number" else ("red", "blue")):
                            questions.append({"question_id": i, "image_id": i, "question": "Same question?"})
                            annotations.append({"question_id": i, "answer_type": kind,
                                                "answers": [{"answer": answer}] * 10})
                            files[i] = f"{i}.jpg"
                        qid += 2
                original = {"questions": {"questions": questions}, "annotations": {"annotations": annotations}, "pairs": pairs}
                for kind, value in original.items():
                    path = root / f"{kind}_{split}.json"
                    path.write_text(json.dumps(value), encoding="utf-8")
                    paths[f"{kind}_{split}"] = str(path)
                pools[split] = natural_pool(original["questions"], original["annotations"], pairs,
                                            "train2014" if split == "train" else "val2014", files)
            rng = np.random.default_rng(6304)
            train, _ = select_natural(pools["train"], {"yes/no": 640, "number": 256, "other": 640}, rng)
            dev, reserved = select_natural(pools["val"], {"yes/no": 12, "number": 8, "other": 12}, rng, disjoint=True)
            test, _ = select_natural(pools["val"], {"yes/no": 36, "number": 24, "other": 36}, rng,
                                      excluded=reserved, disjoint=True)
            expected_rng = rng.random(20)
            preflight = root / "preflight.json"
            write_json(preflight, {"seed": 6304,
                "source_sha256": {key: sha256(value) for key, value in paths.items()},
                "eligible_pairs": {split: dict(Counter(pair["answer_type"] for pair in pool)) for split, pool in pools.items()},
                "selected": {phase: {"pair_ids": [pair["pair_id"] for pair in rows]}
                             for phase, rows in (("train", train), ("dev", dev), ("test", test))}})
            replay = np.random.default_rng(6304)
            actual = natural_from_preflight(paths, preflight, replay, 6304)
            for rows, expected in zip(actual, (train, dev, test)):
                self.assertEqual([pair["pair_id"] for pair in rows], [pair["pair_id"] for pair in expected])
            np.testing.assert_array_equal(replay.random(20), expected_rng)


if __name__ == "__main__":
    unittest.main()
