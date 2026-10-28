import copy
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from task2_vlm.data import consensus, grounding_pool, natural_pool, select_natural, validate_isolation
from task2_vlm.evaluate import DevelopmentDeadline, condition_examples, development, final_evaluation, mismatched_mapping
from task2_vlm.metrics import normalize_answer, paired_advantages, summarize, vqa_score
from task2_vlm.train import CurriculumSampler, stage_at


def example(image, answer="yes", kind="presence"):
    return {"image_id": image, "split": "train2014", "image_file": f"{image}.jpg",
            "question": "Is there a cat?", "answer": answer, "answers": [answer] * 10,
            "kind": kind, "answer_type": kind}


def pair(pair_id, first, second):
    return {"pair_id": pair_id, "kind": first["kind"], "answer_type": first["answer_type"],
            "examples": [first, second]}


class VLMMethodTests(unittest.TestCase):
    def test_vqa_normalization_and_exact_leave_one_out(self):
        self.assertEqual(normalize_answer("The TWO cats!"), "2 cats")
        self.assertEqual(normalize_answer("1,000"), "1000")
        self.assertEqual(normalize_answer("dont"), "don't")
        self.assertAlmostEqual(vqa_score("yes", ["yes"] * 3 + ["no"] * 7), .9)
        self.assertEqual(vqa_score("", [""] * 10), 0)
        self.assertEqual(vqa_score("two", ["2"] * 10), 1)

    def test_consensus_preserves_original_answer(self):
        target, normalized, count, answers = consensus(["Two"] * 8 + ["three"] * 2)
        self.assertEqual((target, normalized, count), ("Two", "2", 8))
        self.assertEqual(len(answers), 10)

    def test_natural_filter_keeps_strong_different_consensus_only(self):
        questions = {"questions": [{"question_id": i, "image_id": i, "question": "What color?"} for i in range(1, 7)]}
        annotations = {"annotations": [{"question_id": i, "answer_type": "other",
                        "answers": ["red" if i % 2 else "blue"] * (7 if i == 3 else 10) +
                                   (["green"] * 3 if i == 3 else [])} for i in range(1, 7)]}
        pool = natural_pool(questions, annotations, [[1, 2], [2, 1], [3, 4], [5, 6]],
                            "val2014", {i: f"{i}.jpg" for i in range(1, 7)})
        self.assertEqual(len(pool), 2)
        selected, used = select_natural(pool, {"other": 1}, np.random.default_rng(6304), disjoint=True)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(used), 2)

    def test_grounding_labels_follow_annotations_and_images_are_disjoint(self):
        annotations = []
        for image in range(40):
            annotations.extend({"image_id": image, "category_id": 1} for _ in range(image % 5))
        instances = {"images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(40)],
                     "categories": [{"id": 1, "name": "cat"}], "annotations": annotations}
        pools, used = grounding_pool(instances, "val2014", np.random.default_rng(6304), 3,
                                     excluded={0, 1}, disjoint=True)
        picked = []
        for kind, pairs in pools.items():
            self.assertEqual(len(pairs), 3)
            for pair_value in pairs:
                self.assertNotEqual(pair_value["examples"][0]["answer"], pair_value["examples"][1]["answer"])
                for row in pair_value["examples"]:
                    picked.append(row["image_id"])
                    expected = ("yes" if row["image_id"] % 5 else "no") if kind == "presence" else str(row["image_id"] % 5)
                    self.assertEqual(row["answer"], expected)
        self.assertEqual(len(picked), len(set(picked)))
        self.assertFalse(set(picked) & {0, 1})

    def test_controls_keep_original_reference_and_omit_visual(self):
        pairs = [pair("a", example(1, "yes"), example(2, "no")),
                 pair("b", example(3, "yes"), example(4, "no"))]
        mapping = mismatched_mapping(pairs)
        swapped = condition_examples(pairs, "complementary")
        self.assertEqual(swapped[0][0]["answer"], "yes")
        self.assertEqual(swapped[0][1]["image_id"], 2)
        self.assertTrue(all(visual is None for _, visual in condition_examples(pairs, "no_image")))
        self.assertNotIn(mapping["a:0"]["image_id"], {1, 2})
        self.assertEqual(mapping, mismatched_mapping(pairs))

    def test_image_leakage_is_rejected(self):
        manifest = {"captions_train": [example(1)], "natural_train": [], "presence_train": [], "counting_train": [],
                    "captions_dev": [example(1)], "grounding_dev": [], "grounding_test": [],
                    "natural_dev": [], "natural_test": []}
        with self.assertRaises(ValueError):
            validate_isolation(manifest)
        manifest["captions_dev"] = [example(2)]
        self.assertEqual(validate_isolation(manifest)["train_images"], 1)

    def test_sampler_resume_replays_identical_pairs_and_mixture(self):
        manifest = {"captions_train": [example(i, kind="caption") for i in range(18000)]}
        for kind in ("natural", "presence", "counting"):
            manifest[f"{kind}_train"] = [pair(str(i), example(i*2, kind=kind), example(i*2+1, kind=kind))
                                         for i in range(12)]
        sampler = CurriculumSampler(manifest)
        sampler.batch(1)
        state = copy.deepcopy(sampler.state_dict())
        expected = sampler.batch(2)
        second = CurriculumSampler(manifest)
        second.load_state_dict(state)
        self.assertEqual(expected, second.batch(2))
        self.assertEqual(sampler.state_dict(), second.state_dict())
        captions = CurriculumSampler(manifest)
        for _ in range(2250):
            captions.batch(0)
        self.assertEqual(captions.caption_position, 36000)
        self.assertEqual(captions.mixture["caption"], 36000)

    def test_stage_boundaries(self):
        stages = [{"updates": 2250}, {"updates": 1500}, {"updates": 750}]
        self.assertEqual(stage_at(2249, stages), (0, 2249))
        self.assertEqual(stage_at(2250, stages), (1, 0))
        self.assertEqual(stage_at(3750, stages), (2, 0))
        self.assertEqual(stage_at(4500, stages), (2, 750))


class VLMGradientTests(unittest.TestCase):
    def test_visual_insertion_masks_and_connector_gradients(self):
        import torch
        from torch import nn
        from common.training import causal_loss
        from task2_vlm.model import VisualLanguageModel

        class Tokenizer:
            pad_token_id = 0
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(c) % 100 + 10 for c in text]}
            def apply_chat_template(self, rows, **kwargs):
                return "USER\n" + rows[0]["content"] + "END\nASSISTANT\n"
            def convert_tokens_to_ids(self, token):
                return 7

        class Language(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=1024)
                self.embedding = nn.Embedding(128, 1024)
                self.head = nn.Linear(1024, 128)
            def get_input_embeddings(self):
                return self.embedding
            def forward(self, inputs_embeds, **kwargs):
                return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(1)))

        settings = {"question_cap": 80, "caption_cap": 64, "answer_cap": 16, "visual_tokens": 196}
        model = VisualLanguageModel(Language(), Tokenizer(), settings)
        row = example(1)
        feature = torch.randn(196, 768).half()
        with_image, labels = model.batch([row], [feature])
        without_image, labels_no = model.batch([row], [None])
        self.assertEqual(with_image["inputs_embeds"].shape[1] - without_image["inputs_embeds"].shape[1], 196)
        self.assertEqual(int((labels != -100).sum()), len(row["answer"]) + 1)
        self.assertEqual(labels[0, -1].item(), 7)
        self.assertEqual(with_image["position_ids"].tolist()[0], list(range(labels.shape[1])))
        causal_loss(model(**with_image).logits, labels).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.connector.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.language_model.parameters()))
        with self.assertRaises(ValueError):
            model.calibrate([{**row, "split": "val2014"}], [feature])


class VLMRecoveryTests(unittest.TestCase):
    def test_development_deadline_preserves_training_mode(self):
        import time
        state = {"training": True}
        model = SimpleNamespace(training=True, eval=lambda: state.update(training=False),
                                train=lambda enabled=True: state.update(training=enabled))
        with self.assertRaises(DevelopmentDeadline):
            development(model, {"captions_dev": [example(1)]}, ".", update=0,
                        output_dir=".", deadline=time.monotonic() - 1)
        self.assertTrue(state["training"])

    def test_partial_checkpoint_is_preserved_and_retry_path_is_indexed(self):
        import torch
        from common.io import read_json
        from task2_vlm.train import save_checkpoint
        model = torch.nn.Linear(2, 3)
        model.adapters_enabled = False
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cpu")
        sampler = SimpleNamespace(state_dict=lambda: {"position": 1})
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "step-0250"
            original.mkdir()
            (original / "partial-marker").write_text("preserve")
            saved = save_checkpoint(model, optimizer, scaler, sampler, original, update=250,
                config={"stages": [{"updates": 4500}]}, identity={"test": True}, seconds=1,
                commit=lambda: None)
            self.assertNotEqual(saved, original)
            self.assertEqual((original / "partial-marker").read_text(), "preserve")
            self.assertEqual(read_json(saved / "complete.json")["update"], 250)
            self.assertIn(str(saved).replace("\\", "\\\\"), (Path(directory) / "checkpoint_index.jsonl").read_text())
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, optimizer, scaler, sampler, saved, update=250,
                    config={"stages": [{"updates": 4500}]}, identity={"test": True}, seconds=1,
                    commit=lambda: None)

    def test_final_evaluation_reuses_complete_conditions_and_preserves_failed_attempts(self):
        import torch
        model = SimpleNamespace(device=torch.device("cpu"), eval=lambda: None)
        manifest = {}
        for suite, kind in (("grounding_test", "presence"), ("natural_test", "natural")):
            manifest[suite] = [pair(suite + str(i), example(i*2+1, "yes", kind), example(i*2+2, "no", kind)) for i in range(2)]
        def prediction(model, rows, features):
            return [{"prediction": row["answer"], "generated_token_ids": [1],
                     "truncated": False, "generation_seconds": .01} for row in rows]
        identity = {"checkpoint": "test-only"}
        with tempfile.TemporaryDirectory() as directory:
            with patch("task2_vlm.evaluate.load_feature", return_value=None), \
                 patch("task2_vlm.evaluate.reference_nll", side_effect=lambda model, rows, features: [1.] * len(rows)), \
                 patch("task2_vlm.evaluate.generate", side_effect=prediction) as generated:
                result = final_evaluation(model, manifest, ".", directory, identity=identity)
                self.assertEqual(generated.call_count, 8)
                first_paths = {path.name for path in Path(directory).glob("*.jsonl")}
                resumed = final_evaluation(model, manifest, ".", directory, identity=identity)
                self.assertEqual(generated.call_count, 8)
                self.assertEqual(first_paths, {path.name for path in Path(directory).glob("*.jsonl")})
                self.assertEqual(result["grounding_test"]["correct"]["aggregate"]["all"]["accuracy"]["mean"], 1)
                self.assertEqual(resumed["evaluation_attempts"], 2)
                self.assertGreaterEqual(resumed["runtime_seconds"], result["runtime_seconds"])
                self.assertTrue(resumed["runtime_accounting_complete"])
                with self.assertRaises(FileExistsError):
                    final_evaluation(model, manifest, ".", directory, identity={"checkpoint": "different"})


if __name__ == "__main__":
    unittest.main()
