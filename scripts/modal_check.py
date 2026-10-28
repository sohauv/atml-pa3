from common.modal_config import (
    app,
    image,
    data_volume,
    checkpoint_volume,
)

@app.function(
    image=image,
    gpu="T4",
    timeout=900,
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
)
def check():
    import json
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(6304)
    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # FP32 parameters; the forward pass uses FP16 autocast.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).to("cuda")

    # Persist the download before testing training.
    data_volume.commit()

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = 4."},
    ]

    prompt = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]

    batch = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Only the assistant response and end-of-turn contribute to loss.
    assert batch.input_ids[0, :len(prompt_ids)].tolist() == prompt_ids
    labels = batch.input_ids.clone()
    labels[:, :len(prompt_ids)] = -100
    end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end = (batch.input_ids[0] == end_token).nonzero()[-1].item()
    labels[:, end + 1:] = -100

    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()

    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(**batch, labels=labels).loss

    scaler.scale(loss).backward()
    assert all(torch.isfinite(p.grad).all().item()
               for p in model.parameters() if p.grad is not None), "Non-finite scaled gradients"
    torch.cuda.synchronize()

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "loss": loss.item(),
        "response_tokens": (labels != -100).sum().item(),
        "peak_allocated_GiB": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
        "backward_completed": True,
    }

    assert torch.isfinite(loss).item(), "Non-finite loss"

    with open("/checkpoints/setup_check.json", "w") as f:
        json.dump(result, f, indent=2)

    checkpoint_volume.commit()
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def main():
    check.remote()
