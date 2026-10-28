# ATML PA3 progress

<!-- SUMMARY:START -->
**Overall verified completion: 54/148 mandatory requirements (36.5%).**
<!-- SUMMARY:END -->

## Counting rule

The denominator is every mandatory checklist item below, grouped by manual page.
An item is checked only after implementation **and** verification with identified
evidence. Code awaiting tests, a submitted job, and a partial experiment do not count.
Optional compression is tracked separately and excluded. Percentages are calculated
by `python -m scripts.update_progress`; they are not estimates of time remaining.
Fine-grained items intentionally include both implementation and required evidence.

## Active work

- Parent: monitor the one A100 Task3livecomparison; FullFT625midpoint is safelysaved; budget, integration, Git and tracker.
- DetachedModalqueue8684d1013aa74addbd5cbef0ed183d81 is active independently of
  Codex/clientpolling; controllercallfc-01M629CPCHC6S9134N9DA74FJW. It firstwaits
  forexistingGPUcall, thencachesbaselineprobes/resumeslive/evaluateslikelihoods.
  Ledgerownership prevents competingstandaloneGPUjobs; totalceiling$14.915625.
- Task1 review: completed numerical/recovery/evaluation/figure and realtinyQwenVLM checks.
- Task2: fullmanifest and ID-onlyexports validated; all 21,749 selected images downloaded and verified; no Task2 GPU run.
- Task3: both1000-question offline analyses completed; live GPU comparison completed all 50 questions; detailed artifact audit is active.

## Manual checklist

### General instructions and submission (pages 1, 15)

- [ ] G01 Individual code/experiments/results; no sharing with other students.
- [ ] G02 Student understands and is responsible for submitted code.
- [x] G03 Public GitHub repository exists and source is pushed. Evidence: https://github.com/sohauv/atml-pa3/tree/dev/pa3-experiments ; commit c4ca244.
- [ ] G04 All materially reused external code, data, papers and tools attributed.
- [ ] G05 Fix seeds where practical; record all reported hyperparameters and preserved configurations.
- [ ] G06 Reproduction README covers every task and required command.
- [ ] G07 Environment specification covers all experiments and scoring versions.
- [ ] G08 Selected split/example IDs are preserved and available for submission.
- [ ] G09 Every reported number traces to saved machine-readable results or reproducible commands.
- [ ] G10 Repository excludes secrets, raw datasets and unnecessary large checkpoints; verify final tracked files.
- [ ] G11 Student-written report contains eight main-content pages in course PA0 NeurIPS format.
- [ ] G12 Report has abstract, introduction, concise setup/methods, separate Task 1-3 results/analysis, conclusion and references.
- [ ] G13 Repository link appears at the end of the report abstract.
- [ ] G14 Report language, interpretation and analysis are written entirely by the student, not generative AI.
- [ ] G15 Main report includes central evidence; optional appendix is supplemental and referenced.
- [ ] G16 Figures/tables have labels, axes, units, methods, useful captions and are discussed in the report.
- [ ] G17 Report distinguishes baselines, checkpoints, splits, metrics and percentage-point versus relative changes.
- [ ] G18 Connected evidence-based analysis discusses limitations, unexpected/inconclusive findings, incomplete experiments and deviations.
- [ ] G19 Accessible GitHub link and PDF submitted to LMS by 06 November 2026 in announced format.
- [ ] G20 Final leakage audit: evaluation data never influences training/hyperparameter choice outside prescribed protocols.

### Task 1: preparation and common settings (pages 2-3)

- [x] T101 Use original post-trained Qwen/Qwen3-0.6B identity. Evidence: completed data metadata and original setup_check.json.
- [x] T102 Select first 40,000 eligible MetaMathQA records in source order. Evidence: preparation completed with retained=40000; internal order/count assertions.
- [x] T103 Reject empty questions/responses. Evidence: preparation filtering implementation executed on full retained dataset.
- [x] T104 Reject formatted sequences over 2,048 tokens, without answer truncation. Evidence: too_long=2; max retained length=1942; preparation assertions.
- [x] T105 Filter query/original_question normalized matches to GSM8K test questions. Evidence: executed normalization/filter in preparation; metadata normalization specification.
- [x] T106 Preserve source IDs, exact upstream revisions and training-data checksum. Evidence: preparation artifacts committed to data volume; results/task1_finetuning/data_metadata.json.
- [x] T107 Format user/supplied-assistant conversations with thinking disabled and verify prompt prefix/EOT. Evidence: full preparation execution with template assertions.
- [x] T108 Mask prompts/padding; supervise assistant response plus EOT; verify actual prepared labels. Evidence: actual first prepared record read from volume:213tokens/159response tokens, EOT151645, contiguous response labels, one masked suffix token; numerical padding test passed.
- [x] T109 Accumulated causal SFT loss is normalized by all valid response tokens in effective batch; verify in actual training. Evidence: variable-length accumulation gradient-equivalence test and eight real GPU optimizer updates with per-effective-batch response-token logs.
- [ ] T110 Every run independently starts from the identical original checkpoint; same examples/order/tokenization/prompts, seed 6304; no old-task data/replay.
- [ ] T111 Each variant completes one epoch, 40,000 examples and 1,250 updates with microbatch 2, accumulation 16, effective batch 32.
- [x] T112 AdamW betas (.9,.999), epsilon 1e-8, decay .01 excluding biases/parameters under two dimensions. Evidence: parameter-group numerical test and executed pilot with saved prescribed configuration.
- [x] T113 Cosine schedule, 38 warmup updates, global norm clipping 1.0 and gradient checkpointing. Evidence: schedule boundary tests, GPU pilot warmup LR/gradient logs, executed checkpointed/clipped training source.
- [ ] T114 FP16 AMP, dynamic scaler, no quantization; unscale before clipping; FP32 master/adapters and optimizer states.
- [ ] T115 Save recovery at updates 312/625/937/1250 including optimizer/scheduler/scaler/RNG/data position; validate resumed trajectory.
- [ ] T116 Log SFT loss/LR versus update and skipped AMP updates for all five methods.
- [ ] T117 Record runtime, peak GPU memory, trainable parameters, checkpoint sizes and method-specific overhead.

### Task 1: five methods (page 4)

- [ ] T118 Full FT updates all original parameters, peak LR 2e-5, lambda=0, completed.
- [ ] T119 Full FT saves full-model displacement at midpoint and final.
- [ ] T120 L2-SP independently restarts, peak LR 2e-5, lambda=1e-3, completed.
- [ ] T121 L2-SP fixed starting reference; sum .5*distance-squared over unique parameters including bias/norm, tied parameters counted once, no averaging.
- [ ] T122 L2-SP logs Omega/lambda-Omega/total objective/final displacement plus reference storage/transfer overhead.
- [ ] T123 LoRA rank4/alpha8 and rank16/alpha32 are separate completed runs, peak LR2e-4.
- [ ] T124 LoRA backbone frozen FP16; FP32 random-A/zero-B adapters, dropout .05, no trainable bias.
- [ ] T125 LoRA targets q/k/v/o/gate/up/down in every layer; record total adapter counts.
- [ ] T126 OPLoRA rank16/alpha32/protection128, same LR/settings and initial unprojected factors as LoRA16, completed.
- [ ] T127 OPLoRA uses fixed leading FP32 U/V, factored two-sided projections, no dense projection matrices; numerical constraints verified.
- [ ] T128 OPLoRA computes bases once, exports projected factors, records basis storage/preprocessing/export overhead.
- [ ] T129 Merge each adapter into a fresh backbone and check merged/unmerged logits, using projected OPLoRA factors.

### Task 1: evaluation, diagnostics and evidence (pages 3-6)

- [ ] T130 Evaluate shared unchanged baseline once; Full FT midpoint and all five final checkpoints: seven conditions.
- [ ] T131 Pinned lm-eval definitions, full splits, zero-shot evaluation in FP16 with dropout disabled.
- [ ] T132 GSM8K test1319: gsm8k_cot, non-thinking chat, greedy cap1024, stock stopping/filters, flexible-extraction exact match, invalid answers incorrect.
- [ ] T133 HellaSwag validation10042: stock completion likelihood, normalized and raw accuracy.
- [ ] T134 BoolQ validation3270: stock completion likelihood accuracy.
- [ ] T135 ARC-Easy test2376: stock completion likelihood normalized and raw accuracy.
- [ ] T136 Save aggregate and per-example outputs, extracted GSM8K answers and truncation flags at all required conditions.
- [x] T137 Fixed probe set: seeded 32 GSM8K test and32 HellaSwag validation prompts; save IDs before inspecting model results. Evidence: completed baseline probe run; 32 unique IDs per task saved before forward passes, seed6304; probe_verification.json.
- [x] T138 Cache baseline down_proj inputs at last non-padding prompt token in layers0/13/27, before answer continuation. Evidence: downloaded cache verified finite FP32 [64,3072] at each layer, SHA256 and ID-only audit in probe_verification.json; capture source inspected.
- [ ] T139 For five finals and Full FT midpoint compute FP32 rho128 and eta using shared starting bases and effective updates.
- [ ] T140 For same checkpoints compute FP32 local activation R(z), shared cached inputs, math/non-math separated, denominator epsilon1e-8.
- [ ] T141 Generate all24 supplied qualitative prompts under seven conditions, greedy non-thinking cap512; save168 responses by question/format.
- [ ] T142 Annotate first/last underlying plain questions, all3 formats/all7 conditions:42 responses; compliance/coherence/relevance/repetition/math intrusion/inappropriate content/truncation.
- [ ] T143 Seven-condition four-benchmark table with primary scores and baseline changes in percentage points.
- [ ] T144 Combined five-run SFT loss plot, Full FT/L2-SP penalty and displacement comparison.
- [ ] T145 Resource comparison table across five methods including overhead.
- [ ] T146 Compact diagnostics figure/table with geometry and math/non-math activation corrections and limitations.
- [ ] T147 Fixed42-response findings and aligned excerpts with stated selection rule, student interpretation.
- [ ] T148 Analyze rank, regularization, orthogonal protection, Full FT/LoRA16 trade-off and Full FT progression, accounting for cost/learning-rate differences.
- [ ] T149 Address all five Task1 research themes: specialization/retention, restrictions, weight changes/behavior, compute cost, behavior beyond scores.

### Task 2: data and visual interface (pages 6-9)

- [ ] T201 Original Qwen3-0.6B/non-thinking; frozen google/siglip-base-patch16-224; trainable 768->1024 GELU 1024->1024 connector.
- [ ] T202 Deterministic SigLIP196 final patch features in14x14 order, final layer norm exactly once, FP16 cache; remove encoder from GPU before Qwen training.
- [x] T203 COCO2014 captions/instance annotations and VQAv2 original complementary-pair sources prepared. Evidence: all10 released sourceJSONs downloaded, checked, streamed; sourceSHA256s recorded in selection_ids/provenance.json.
- [x] T204 Presence pools pair annotated-present and category-unannotated images, same generated question, yes/no targets. Evidence: actual4096pair trainingpool and evaluationpools built/validated; synthetic annotation-label regression passed.
- [x] T205 Counting pools pair different annotated counts1-4 for same category and generated question. Evidence: actual4096pair trainingpool plus evaluationpools built/validated from original annotations.
- [x] T206 Natural complementary pairs have different normalized consensus answers and >=8/10 agreement on each image; original consensus string target and all10 references retained. Evidence: actual sourcepreflight and fullmanifest validation; filtering regression passed.
- [x] T207 Train2014 training, val2014 development/final; pairs within splits; all development/final sets image-disjoint from one another/training/replay. Evidence: actual isolation assertions;21221trainingimages and mutuallydisjoint evaluation imagecounts96/48/128/64/192.
- [x] T208 Caption train6000 images*3=18000; development96 images*1; save seeded image/example IDs. Evidence: actualmanifest exactcounts and ID-only captionCSVs with hashes.
- [x] T209 Grounding development24pairs/48questions and final64pairs/128questions, presence/counting represented. Evidence: actualmanifest validates12+12devpairs/32+32testpairs; exportedIDs.
- [x] T210 Natural training1536pairs with640/256/640 yes-no/number/other; development32 with12/8/12; final96 with36/24/36. Evidence: actualpreflight/fullmanifest exactquota assertions and exportedIDs.
- [x] T211 Seed6304 selects data; image/question/pair IDs saved; answer normalization consistent. Evidence: deterministic selected-only streaming/RNGreplay regression and actual9ID-onlyCSVs/provenance, manifestSHA68830e40f86509376296ae5bee12470f5b4669bd88c8dd3ebeee16cace79383b.
- [x] T212 Visual embeddings inserted before question inside Qwen user message with correct labels, attention masks, positions; shape/frozen-set/connector-gradient checks. Evidence: realtinyQwen hidden1024 gradient and16-step cached/left-padded position tests plus syntheticmask/control/isolation tests passed.
- [ ] T213 Connector magnitude calibration uses training-only text/visual embeddings; save rule and scale.

### Task 2: curriculum and recovery (pages 8-9)

- [ ] T214 Same model continues through all3 stages; seed6304, microbatch8/accumulation2/effective16.
- [ ] T215 AdamW decay.01/global clipping1; per-stage5% warmup and cosine decay to10% peak; gradient checkpointing/no quantization.
- [ ] T216 FP16 autocast/scaler/frozen backbone; FP32 trainable parameters/optimizer; unscale before clipping.
- [ ] T217 Content caps question80/caption64/answer16; loss only target/EOT and mean tokens within example then examples across effective batch.
- [ ] T218 Stage1 connector-only2250updates/two passes18000captions, peakLR1e-3, completed.
- [ ] T219 Stage2 1500updates, connectorLR1e-4/LoRALR2e-5, draw probabilities presence.5/counting.25/caption.25, completed.
- [ ] T220 Stage2 enables LoRA16/alpha32/dropout.05/no bias, q/k/v/o only, original weights frozen.
- [ ] T221 Stage3 750updates, connector5e-5/LoRA1e-5, natural.7/presence.1/counting.1/caption.1, completed.
- [ ] T222 Each stage2/3 draw yields a complete pair or two captions, seeded random sampling with no enforced exact quotas; training-pool reuse allowed.
- [ ] T223 Complete4500updates/72000presentations; checkpoint every250updates and stage weights at2250/3750/4500.
- [ ] T224 Recovery saves optimizer/scheduler/scaler/sampler/stage/RNG and is verified by resume.
- [ ] T225 Save loss/mixture counts/configs/development metrics, parameter counts by stage, runtime/memory and separate feature-caching cost.

### Task 2: evaluation and required evidence (pages 8-10)

- [ ] T226 Development evaluation before training/every500updates/stage endpoints2250/3750/4500, without final-test checkpoint selection.
- [ ] T227 Caption reference loss and both-suite answer NLL under correct/complementary/no-image; exclude termination from answer NLL and example-average token means.
- [ ] T228 Compute image-swap penalty NLL(partner)-NLL(correct), original references fixed.
- [ ] T229 Final Stage3 greedy non-thinking cap16/batch8 under correct/complementary/mismatched/no-image on both suites.
- [ ] T230 Freeze unrelated-image mappings before generation, exclude both own-pair images, reuse across comparisons; no-image removes visuals/rebuilds masks/positions.
- [ ] T231 Grounding normalized exact-match by presence/counting; VQAv2 local10-answer leave-one-out soft scoring by yes-no/number/other.
- [ ] T232 Pair accuracy, identical-answer rate, empty-answer rate, mean reference NLL, generation time/answer for every suite/condition; empty predictions stay in denominators.
- [ ] T233 Correct-image advantages in accuracy/NLL,95% intervals from4000 complete-pair bootstrap resamples with matched IDs across conditions.
- [ ] T234 Analyze whether complementary-swap answers follow partner versus unsupported answers; fixed illustrative-example selection rule.
- [ ] T235 Learning-curve figure with loss/developmentcaptionloss/answerNLL/swappenalties and stage boundaries.
- [ ] T236 Both-suite four-condition/type results tables, pair diagnostics/NLL/advantages/intervals.
- [ ] T237 Aligned complementary-image examples with question/references/predictions and stated selection rule.
- [ ] T238 Resource table includes stagewise trainable counts/traininggenerationtime/memory and feature caching.
- [ ] T239 Address four Task2 research questions: interface curriculum, grounding/generalization, image-use measures, capabilities/cost; discuss limitations throughout.

### Task 3: supplied data, splits and correctness (pages 11-12)

- [x] T301 Use supplied32rollouts per1000questions for Qwen2.5-3B MATH and Qwen2.5-VL3B MathVista; no offline regeneration.
- [x] T302 Read supplied README and join generations/probability/attention on(dataset,sample_id,rollout_id), verify all token-list alignments and termination exclusions.
- [x] T303 Lexicographically sort question IDs, default_rng6304 permutation, first200development/rest800heldout, all rollouts grouped; save IDs.
- [x] T304 Save ten seeded random rollout orders/question, nested prefixesN1/2/4/8/16/32, reuse across all methods/policies.
- [x] T305 Stored extracted answers are used; math-verify mathematical equivalence and released MathVista metadata normalization/choiceletters/precision; scorer fixed/versioned before heldout.
- [x] T306 Form equivalence groups without gold; use same equivalence for grading; missing extraction oneNO_ANSWER group always incorrect, retained in votes/cost/accuracy.
- [x] T307 Earliest-supporting-rollout tie breaks, average tied RWC ranks; all selectors return sole candidate atN1. Evidence: selector/regressiontests include mathematicalfloatvote ties, average tiedranks andallN1rules.
- [x] T308 Average over orders within questions then across questions; generated-token sum costs, FirstAns consumes only first candidate.
- [x] T309 Main accuracy differences have1000 paired question-bootstrap95% intervals after averaging orders.

### Task 3: fixed-budget selection and signal analysis (pages 12-13)

- [x] T310 FirstAns accuracy/oraclecoverage/distinctanswergroups for everyN, decompose no-correct-generated versus available-but-missed.
- [x] T311 Implement mean sampled-token logp and full-distribution KL self-certainty from raw supplied measurements.
- [x] T312 Normalize visual attention by image+prompt+generated eligible totals, then average token visual engagement.
- [x] T313 VisConf token KL calibration pools current prefix only, linear5th/95thpercentiles, clip0-1/epsilon1e-8; mean confidence times mean vision.
- [x] T314 RWC descending average ranks and (N-rank)/(N-1) weighted answer votes, correct N1 handling. Evidence: numericalrank/consensus/N1tests andfullfixturepipeline passed.
- [x] T315 Both datasets compare FirstAns/self-consistency/highestlogp/highestcertainty/certaintyRWC at allN.
- [x] T316 MathVista additionally highestvision/visionRWC/highestVisConf/VisConfRWC at allN.
- [x] T317 AtN32 analyze within-question length/score/correctness associations and top-bottom-scorequartile correctness; shortest-rollout control.
- [x] T318 MathVista token confidence-vision correlation per rollout, aggregate by question; tokenwise-fusion ablation under highestscore/RWC and covariance identity.
- [x] T319 Fixed illustrative trajectories examine signal agreement/disagreement/variation; no causal grounding claim from attention alone.
- [x] T320 Save question results/candidateIDs/selectedanswers/costs and distinguish sampling opportunity versus selection gap.

### Task 3: live comparison (pages 12-13)

- [ ] T321 Use first50 MATH heldout questions in saved permutation; seed6304 and pinned original Qwen2.5-3B revision from suppliedREADME.
- [ ] T322 FP16/no quantization/batch1; temperature.7/top-p.8/top-k20/repetition1; exact native template/system/problem/instruction.
- [ ] T323 Single response cap3072; parallel3independent cap1024 each with self-consistency; sequential initial1024+critique/revision2048.
- [ ] T324 Reuse first parallel response as refinement initial; exact appended critique instruction; independent seeds; no gold feedback/selection.
- [x] T325 Extract last literal boxed answer by balanced-braces; missing/empty/unclosed last box stays missing, no numeric fallback. Evidence: executed nestedbrace/lastbox/missing/truncatedbox regressiontests.
- [ ] T326 Save all prompts/responses/seeds/inputoutputlengths/truncation; end-to-end time sameGPU excluding model load, peakmemory.
- [ ] T327 Report accuracy/actualtokens/runtime/extractiontruncation and wrong->correct/correct->wrong; charge sharedinitial twice across strategies and all critique/prompt tokens.

### Task 3: adaptive sampling and evidence (pages 13-15)

- [x] T328 Simulate saved orders; check after4/8/16 else32; both controllers return current self-consistency answer.
- [x] T329 Agreement stop requires unique nonmissing leader supported byfractiontau; quality additionally same RWCwinner and normalizedmargin>=gamma.
- [x] T330 Quality uses certainty forMATH/VisConf forMathVista; only revealedprefix for scores/calibration/ranks; onegroup runnerupvote0.
- [x] T331 On development searchtau.5/.625/.75/.875/1 and gamma0/.1/.25/.5 plusalways32fallback; choose lowesttokencost within1percentagepoint of fixed32accuracy.
- [x] T332 Policy ties choose higheraccuracy thensmallertau/gamma; freeze selected policies before heldout.
- [x] T333 Heldoutaccuracy/meantokens/meanrollouts/stoppingdistribution and fixedN comparisons; analyze incorrectearlystops and fixed32recoveries.
- [x] T334 Accuracy/oraclecoverage curves for bothdatasets, compactmethodtable and failuredecomposition.
- [x] T335 Signal-analysis figures/tables forlength/correctness/confidencevision/tokenfusion.
- [ ] T336 Live-comparison table and revision-transition evidence on50questions.
- [x] T337 Adaptive threshold/costaccuracy/distribution figures/tables, matchedobservedcost comparisons.
- [ ] T338 Aligned selection/refinement/stopping examples with statedselectionrule.
- [ ] T339 Interpret token cost as proxy, note scoring/attention/prefill overhead and modeldataset confounding; no gold/futurecandidate leakage.
- [ ] T340 Address five Task3 themes: exploration/selection, evidence/consensus, breadth/refinement, adaptiveallocation, compute/reliability.

## Optional requirement (excluded from percentage)

- [ ] Optional Task2 compression: new connector/adapters; average2x2 patch blocks to49tokens; repeat fullrecipe with fixedseed/data/budgets/eval; compare accuracy/grounding/runtime/memory.

## Verification log

| Check | Outcome | Evidence |
|---|---|---|
| Initial nine Python files syntax parsing | Passed | Audit execution before implementation |
| Budget suite | 8 passed | `python -B -m unittest discover -s tests -p test_budget.py -v`, outside Windows temp-write sandbox; includes successful-call reconciliation |
| Numerical loss/adapter suite | 8 passed | CPU Torch2.6; `tests/test_numerics.py`; includes saved-basis resume without SVD |
| Gold-free selection suite | 7 passed | `python -B -m unittest discover -s tests -p test_selection.py -v` |
| Modal worker declarations | Passed | Imported `scripts.modal_jobs` without launching compute |
| Budget test first sandbox attempt | Environment failure, resolved | Windows denied temporary-file writes; rerun outside sandbox passed |
| Tiny-Qwen five-method/recovery integration | 3 passed | Independently rerun in7.183s: all5backward/FP32AdamW/frozenbackbone/mergedlogits; immutablepayload/RNG; real4-update uninterrupted versus2+2resume model/moments/RNG bit-for-bit |
| Task2 synthetic CPU suite | 9 passed | Connector gradient flow, masks/controls, leakage, pool selection, sampler resume, curriculum and VQA scoring; subagent execution |
| Task3 fixture suites | 9 passed;1 symbolic test pending | Joins, splits, frozen policies, offline integration and250-call live orchestration/resume mocked; actual-result execution still pending |
| First resume launch | Local startup failure, fixed and retried | Rich emitted Unicode to Windows cp1252 before GPU submission; launcher now configures UTF-8; failure reservation retained conservatively |
| Complete CPU suite after dependency installation | **51 passed in22.367s** | `.venv/Scripts/python.exe -B -m unittest discover -s tests -v`; allmethods/recovery, visualtinyQwen, selection/scoring/joins/liveorchestration fixtures; no skippedtests in this execution |
| Updated complete CPU suite | **60 passed in28.648s** | Samecommand; addedfigureintegrity, actualFP16merge/FP32diagnostics, streaming/RNGreplay, VLMtimeout/recovery, bounded symbolic-timeout cleanup; no failed/skippedtests |
| Detachedqueue and batching regression milestone | **74 passed in36.945s** | Fullunittestdiscover executed; includes10queue/idempotency/timeout/atomicbudget tests and5evaluationprecision/batching/cache tests. Initialpuretestimport ofModalfailed; fixed withlazySDKimports andrerunpassed. |

## Modal jobs and spending

### Completed offline-analysis evidence (29 October)

Task3 requirements checked above are verified by complete original-input runs at
`outputs/task3/math-v3` and `outputs/task3/mathvista-v1`, each1000questions with
200development/800heldout, ten savedorders, allsixbudgets and prescribedselectors.
Bothcompletion markers, frozenpolicies, perquestion/perorder records, paired1000
question-bootstrap intervals, signalanalyses, deterministicexamples and inspected
figures exist. Smallsubmission artifacts are exported to results/task3_test_time;
sourcehash manifests retain traceability to the preserved detailedrawresults.

| Held-out result atN32 | MATH | MathVista |
|---|---:|---:|
| FirstAns accuracy |64.1875%|53.6000%|
| Self-consistency accuracy |73.6250%|61.6750%|
| Self-certaintyRWC accuracy |75.3000%|62.5000%|
| VisConfRWC accuracy |Notapplicable|63.1250%|
| Oraclecoverage |91.1250%|89.5000%|
| Full32 generatedtokens/question |21347.0175|8897.7575|

These are supplied-rollout CPU analyses, not newlygenerated or Task1model scores.
The fixed/versioned MATHscorer recorded40symboliccomparison timeouts and returned
false for thosecomparisons; noquestions/candidates were removed. MathVista has
zero symbolic failures. Fullscorer/error records accompany the results; the
timeout rule is disclosed rather than treating ambiguous comparisons as proven.

Authorization: initially$10; the human explicitly raised the PA3 TOTAL cap to$15
on29October, including priorjobs. Workspace cap is$15 with$0 out-of-pocket spend.
Billing limits are never modified. Conservative local reservations are **not**
claimed to equal provider-billed usage.

| Job | Hardware | Status | Reservation | Artifacts |
|---|---|---|---:|---|
| Existing setup check | T4 | Completed before this implementation | Not charged to local ledger; account usage unknown | Existing `/setup_check.json`, finite loss .10308, peakallocated5.038GiB |
| task1-data-v1 | CPU | Completed | $0.296256 | `/data/task1/{train.jsonl,source_ids.json,metadata.json}` |
| full-pilot-20261028 | A10040GB | Completed8updates/256examples; not a final run | Initial$1.165626; reconciled conservative estimate$0.369811 | 141.46s remote runtime,16.110GiB peak, no skippedupdates, immutable recoverycheckpoint10.16GB |
| full-pilot resume attempt1 | No GPU call | Failed at local Unicode output, before submission | $0.900711 retained | Failure trace preserved locally |
| full-pilot resume attempt2 | A10040GB | Completed:8->40updates,1280examples | $0.560254 conservative estimate | Resume succeeded;482.28s remote runtime, recovery preserved |
| L4 cost benchmark | L4 | Cancelled while pending, no worker/GPU execution | $0.310896 retained conservatively | CallgraphPENDING/task_idempty; Modal logs confirmed unavailableL4capacity for40+minutes |
| FullFT required midpoint continuation | A10040GB | Submitted/running:40->625updates,100-minute ceiling | $5.404266 reserved | Existing run/checkpoint lineage; prescribed312/625 recovery checkpoints |
| FullFT continuation outcome | A10040GB | Completed625updates/20000examples; noAMPskips | $5.036586 conservative estimate | Prescribedstep0312/0625 saved; midpointdisplacement10.1242, peak20.4156GiB; full1250run stillincomplete |
| Task3livecomparison | A10040GB | Submitted/running,40-minute ceiling | $2.225286 reserved | task3-live-20261029; fixed50MATHheldoutIDs and pinnedQwen2.5-3B; resumablepercalls |

Local original pilot/preparation reservations were reconciled only after confirmed
successful completion, using their entire local app lifetimes plus the original
safety/overhead allowances. Current exact totals are in ignored `runs/budget.json`.
Preparation conservative estimate:$0.034990. Initialpilot estimate:$0.369811.
Reservations and these conservative estimates are not actual provider charges.
Provider billing total: not yet reconciled; do not describe reservations as spend.

## Git milestones

| Commit | Milestone | Push |
|---|---|---|
| `c4ca244d0a8a1472ad0b534a046add9c9e1268d4` | Tested budget reservations, persistence helpers, safeguards and scaffold docs | Succeeded, `origin/dev/pa3-experiments` |
| `3e1243c52243989bf0074649de785066a499ffa5` | Manual-derived148-item tracker and verified-percentage calculator | Succeeded, `origin/dev/pa3-experiments` |
| `c0a3cacb69dd4c27586ebf88c123a6a64b2ff508` | Tested Task1 methods/recovery/evaluation code and budget-gated Modal runner | Succeeded, `origin/dev/pa3-experiments` |
| `3c9263e3a7ce1341bc2177bf8223508e530433bc` |60-test-verifiedVLM/test-timeanalysis/figurecode and actualvalidatedTask2ID-onlysplits | Succeeded, `origin/dev/pa3-experiments` |
| `60290b6b462eea01aebd0b046e68a57dac27676b` | VerifiedFullFT625midpoint metadata and bothcompleteofflineanalysis metrics/figures | Succeeded, `origin/dev/pa3-experiments` |

## Blockers and next steps

- Selected-image downloader was no longer running at19003files; resumption now
  reuses/verifies existingimages and downloads only missing files. No duplicate
  downloadprocess or additionalModaljob was launched. Earlierrawdata is preserved.
- Prior conservative estimates/reservations total$9.438534. The newly authorized
  TOTAL is$15; detachedqueue plannedadditionalceiling$5.477091, cumulative$14.915625.
  These are guardreservations, not providerbills. UnaffordableFullFTtail stayspending.

- Full required GPU workload exceeds the current$15 allowance; measure throughput
  before committing remaining reservations. Pilot measured roughly8.9sec/update;
  FullFT alone projects about3.1GPUhours before recovery/evaluation overhead. All
  five methods plus required evaluations cannot be claimed affordable on this
  initial allowance. Do not silently reduce prescribed budgets.
- Local dependencies finished installing; the full51-test suite passed. Task2
  metadata parsing was refactored to stream large sourceJSONs after a memory-pressure
  failure; raw data was preserved. CPU analysis/data preparation are coordinated.
- Task2 source preparation and fullmanifest validation completed; ID-only submission
  exports are in results/task2_vlm/selection_ids. Selected21749image download is
  active; GPU feature caching/training/finaltest have not run.
- AllTask3 offlineanalyses completed; earlierMATHv1/v2 partialattempts remain
  preserved. Livecomparison runs separately onA100. The625-update FullFTmidpoint
  is saved; requiredfull1250training and benchmarkconditions remain incomplete.
- The student must write/interpret the PDF report and submit it to LMS; AI-written
  report language is prohibited by the manual. Code, numeric metrics and plot tools
  can be prepared autonomously.
- Next: verify GPU pilot/checkpoint, execute integration suites, integrate Task2/3,
  commit/push tested milestones, run affordable compliant experiments, refresh audit.

- Queue milestone committed and pushed: 9a30739 to origin/dev/pa3-experiments. Detached controller verified in wait_existing; existing Task3 GPU client reported 28/50 questions complete. Overall verified requirements remain 52/148 (35.1%); queued work is not counted as complete.

- Monitor milestone: selected-image downloader exited successfully after verifying all 21,749 images. The original live GPU session ended normally at its time boundary with 29/50 questions saved. The detached controller automatically advanced to baseline probes (child call fc-01M629Q0PJQ36FCTDP99Y2B5HF), confirming serial handoff after the original client exited. Remaining live questions are already reserved in the queue. No new requirement was marked complete; 52/148 (35.1%) remains verified.

- Baseline probes completed; verified all 64 selected examples and finite activation tensors for layers 0/13/27. Two probe requirements now verified. Detached queue resumed the existing live comparison without an overlapping GPU job (child fc-01M62A28B824NHZ07X6JMH5W5P). Raw prompts and activation tensors remain local/volume-only.

- Live comparison completed 50/50 questions: single 64%, parallel 58%, refinement 62%; wrong-to-correct 1, correct-to-wrong 0; zero symbolic comparison failures/timeouts. Aggregate table and figure generated and visually checked. Full raw-call download/audit is in progress before checking additional requirements. Queue automatically advanced to baseline likelihood benchmarks (call fc-01M62CYBZW5VWEF99X1FNA2JJ8). Completion remains 54/148 (36.5%).
