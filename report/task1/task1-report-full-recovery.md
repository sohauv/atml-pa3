# Task 1: Fine-tuning and retention

We evaluated the original Qwen3-0.6B model, the Full FT checkpoint at update 625, and all five final checkpoints at update 1,250. Training used the same 40,000 examples and effective batch of 32. The verified training logs record 1,250 successful updates and zero skips for each method. Benchmarks used full splits and the same pinned evaluation protocol.

## Specialization and retention

Scores are percentages. Parentheses show changes from the baseline in percentage points.

| Condition | HellaSwag norm. | BoolQ | ARC-Easy norm. | GSM8K flexible |
|---|---:|---:|---:|---:|
| Baseline | 47.31 (+0.00) | 64.34 (+0.00) | 55.93 (+0.00) | 43.37 (+0.00) |
| Full FT midpoint | 45.53 (-1.78) | 58.47 (-5.87) | 62.42 (+6.48) | 56.63 (+13.27) |
| Full FT final | 46.16 (-1.16) | 58.01 (-6.33) | 63.80 (+7.87) | 63.91 (+20.55) |
| L2-SP | 46.34 (-0.98) | 62.32 (-2.02) | 63.38 (+7.45) | 64.75 (+21.38) |
| LoRA4 | 47.20 (-0.11) | 68.38 (+4.04) | 64.18 (+8.25) | 59.51 (+16.15) |
| LoRA16 | 47.35 (+0.04) | 68.90 (+4.56) | 62.96 (+7.03) | 62.62 (+19.26) |
| OPLoRA16 | 47.27 (-0.04) | 68.65 (+4.31) | 63.01 (+7.07) | 62.62 (+19.26) |

Every trained model improved GSM8K flexible extraction by at least 16.15 points and ARC-Easy normalized accuracy by at least 7.03 points. L2-SP produced the highest GSM8K score, 64.75%, with a 2.02-point decline in BoolQ. LoRA16 scored 62.62% on GSM8K and improved BoolQ by 4.56 points. HellaSwag varied less across final models, with changes between −1.16 and +0.04 points.

Full FT gained 7.28 GSM8K points from midpoint to final while BoolQ fell another 0.46 points. L2-SP scored 0.83 points above Full FT on GSM8K and 4.31 points above it on BoolQ. Its final parameter displacement was 6.27 versus 10.78 for Full FT, consistent with the starting-weight penalty limiting drift in this run. The Full FT midpoint displacement was 10.12. Among the adapters, increasing rank from 4 to 16 gained 3.11 GSM8K points and 0.52 BoolQ points, with a 1.22-point decrease in ARC-Easy normalized accuracy. OPLoRA16 and LoRA16 tied on GSM8K flexible extraction. Their other benchmark differences were under 0.25 points. These are single-run differences. Full FT/L2-SP and adapters also used different peak learning rates (2e-5 versus 2e-4), which limits attribution to parameter constraints alone.

GSM8K strict match was 0% for every condition. The table uses the specified flexible extraction metric. Both filters remain in the archived results. Raw HellaSwag and ARC-Easy accuracy are retained in the verified evaluation records.

## Training and resource use

[Training loss](training-loss-full-recovery.pdf) uses nonoverlapping 25-update means. Over the final 25 updates, Full FT and L2-SP averaged 0.1874 and 0.1876 SFT loss. LoRA4, LoRA16, and OPLoRA16 averaged 0.2150, 0.2019, and 0.2034. The [weighted L2-SP penalty](l2sp-penalty-full-recovery.pdf) is plotted separately from the SFT loss. Full FT uses a zero penalty coefficient.

| Method | Runtime (h) | Peak GPU GiB | Trainable parameters | Final checkpoint GB |
|---|---:|---:|---:|---:|
| Full FT | 3.52 | 20.42 | 596,049,920 | 10.160 |
| L2-SP | 4.09 | 7.49 | 596,049,920 | 10.160 |
| LoRA4 | 4.07 | 9.10 | 2,523,136 | 0.041 |
| LoRA16 | 3.96 | 9.22 | 10,092,544 | 0.162 |
| OPLoRA16 | 4.88 | 9.51 | 10,092,544 | 0.485 |

These are recorded run summaries across different GPUs, sessions, and memory strategies. They describe observed resource use and cannot isolate method speed. L2-SP stored a 2.38 GB fixed FP32 reference on CPU (1.72 s initial storage), and offloaded AdamW state and FP32 host parameters. Its 7.49 GiB GPU peak reflects that implementation, including the blockwise loss. Per-step transfer time was not separately measured. OPLoRA16 spent 32.04 s preparing its bases and stored 323 MB of basis tensors. Its final checkpoint was about three times the LoRA16 checkpoint. See [resource data](resources-full-recovery.csv) for checkpoint write and export timings.

The full-recovery L2-SP rerun saved resumable checkpoints at updates 312, 625, 937, and 1,250.

## Weight and activation changes

The diagnostics compare effective changes in the down-projection matrices at layers 0, 13, and 27. The ratio rho128 measures the fraction of update energy in the top 128 left singular directions of the starting weight, and eta measures relative update norm. R(z) measures the relative correction to the original projection of a cached baseline input. We summarize medians separately for 32 math and 32 non-math probes.

OPLoRA16 reduced rho128 to roughly 0.9–1.6e-9 across these layers, versus 0.11–0.14 for LoRA16. Its eta remained comparable to LoRA16. Small projected energy did not imply a small activation correction. At layer 13, OPLoRA16's median math R(z) was 0.088 versus LoRA16's 0.076. Their non-math medians were both near 0.031. L2-SP reduced eta relative to Full FT at all three sampled layers. Its median math correction was lower at layers 0 and 13. Several methods had larger math corrections at layers 0 and 13, but the pattern did not consistently hold at layer 27. These local probes hold inputs fixed, so they cannot capture changes propagated through earlier layers or establish a causal link to benchmark accuracy. [Activation figure](activation-ratios-full-recovery.png) · [Diagnostic data](diagnostics-full-recovery.csv)

## Behavior beyond benchmark scores

The qualitative run generated 24 responses for each of seven conditions. The annotation subset contains the first and last underlying plain questions, each with plain, JSON, and template variants. The 42 ratings are assistant judgments with per-response rationales, pending student review. Five of 14 JSON responses were valid standalone objects and seven of 14 template responses followed the requested structure. Eight responses repeated content, six included math-style intrusions, and five hit the token limit.

All seven plain wish responses reversed the roles by asking the user to name a wish. With the same wish prompt in template form, six supplied a wish and reason. For example, LoRA16's plain response began “I can fulfill your one wish,” while its template response gave “I want to be able to travel anywhere in the world.” The Full FT final plain wish response appended a long series of numbered answer markers and hit the cap. On the relationship prompt, the baseline JSON response was fenced, while several trained conditions put prose before a JSON object. These examples show format sensitivity and repetition on the fixed subset. They do not support a population estimate of instruction following.
