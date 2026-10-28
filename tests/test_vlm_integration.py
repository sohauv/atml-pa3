"""Real tiny-Qwen integration; random weights, synthetic images, no downloads."""
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from common.training import causal_loss, seed_everything
from task2_vlm.evaluate import generate
from task2_vlm.model import VisualLanguageModel


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) % 240 + 8 for character in text]}

    def apply_chat_template(self, messages, **kwargs):
        return ("<|im_start|>user\n" + messages[0]["content"] +
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def convert_tokens_to_ids(self, token):
        return 7

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token) for token in ids if token not in (0, 7))


class VLMRealQwenTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        seed_everything(6304)
        backbone = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=256, hidden_size=1024, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2,
            head_dim=128, max_position_embeddings=512, sliding_window=None,
            use_sliding_window=False, attention_dropout=0.0,
            tie_word_embeddings=False, pad_token_id=0, eos_token_id=7, use_cache=False))
        settings = {"question_cap": 80, "caption_cap": 64, "answer_cap": 16,
                    "visual_tokens": 196}
        self.model = VisualLanguageModel(backbone, CharacterTokenizer(), settings)
        self.rows = [{"question": "Is there a cat?", "answer": "yes", "kind": "presence"},
                     {"question": "Cat?", "answer": "no", "kind": "presence"}]
        self.features = [torch.randn(196, 768).half() for _ in self.rows]

    def test_real_qwen_frozen_backbone_propagates_connector_and_adapter_gradients(self):
        self.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()
        batch, labels = self.model.batch(self.rows, self.features)
        loss = causal_loss(self.model(**batch).logits, labels, reduction="examples") / 2
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.connector.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.model.connector.parameters()), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for p in self.model.language_model.parameters()))
        self.model.zero_grad(set_to_none=True)
        self.model.enable_adapters()
        batch, labels = self.model.batch(self.rows, self.features)
        loss = causal_loss(self.model(**batch).logits, labels, reduction="examples") / 2
        loss.backward()
        adapter_names = [name for name, p in self.model.language_model.named_parameters() if p.requires_grad]
        self.assertEqual(len(adapter_names), 8)  # A/B for q/k/v/o in the one real layer.
        self.assertTrue(all(name.endswith((".A", ".B")) for name in adapter_names))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters() if p.requires_grad))

    def test_actual_cached_generation_advances_positions_for_left_padded_visual_prompts(self):
        self.model.eval()
        self.model.language_model.config.use_cache = True
        # A zero head deterministically emits token0 and never EOS7, ensuring all
        # sixteen cached decode steps execute. This test checks positions/masks,
        # not prediction quality from random weights.
        with torch.no_grad():
            self.model.language_model.lm_head.weight.zero_()
        batch, _ = self.model.batch(self.rows, self.features, target=False, left_pad=True)
        lengths = batch["attention_mask"].sum(1).tolist()
        observed = []

        def capture(module, args, kwargs):
            observed.append({"positions": kwargs["position_ids"].detach().clone(),
                             "attention": kwargs["attention_mask"].detach().clone()})

        handle = self.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            records = generate(self.model, self.rows, self.features)
        finally:
            handle.remove()
        self.assertEqual(len(records), 2)
        self.assertTrue(all(len(row["generated_token_ids"]) == 16 for row in records))
        self.assertEqual(len(observed), 16)
        for index, length in enumerate(lengths):
            first = observed[0]
            torch.testing.assert_close(first["positions"][index][first["attention"][index].bool()],
                                       torch.arange(length))
            for step, snapshot in enumerate(observed[1:], start=1):
                self.assertEqual(snapshot["positions"][index].item(), length + step - 1)
        no_image, _ = self.model.batch(self.rows, [None, None], target=False, left_pad=True)
        self.assertEqual((batch["attention_mask"].sum(1) - no_image["attention_mask"].sum(1)).tolist(), [196, 196])


if __name__ == "__main__":
    unittest.main()
