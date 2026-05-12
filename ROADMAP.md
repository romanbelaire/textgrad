# DiffTG research roadmap

Living document. Update the Status column of each table as experiments are
run. Columns are: `id` (stable handle), `status` (pending / running / done /
blocked), `owner`, and `notes / result hash`.

# Current status
The method doesn't yet produce meaningful changes in toy settings, most likely because the latent space step size is not enough to overcome the argmax token decoding/discretization. Upon further thinking, it is also a problem that the embedding space (and therefore the latent step direction) is dependent on the learned representation of the LM, so the end version of this gradient projection model will be non-transferable. Lastly, the motivation of this work is unclear. I started looking into it because the textgrad "textual gradients" irked me as unprincipled speculation, but other than that I do not have a clear problem to address, nor do I currently have an implied second order problem that this project would lead into.

## 0. Research thesis

We claim that reward-grounded, representation-space perturbations (a
"diffusion textual gradient") are a better local policy-improvement operator
than TextGrad's natural-language critiques, in three measurable ways:

- **H1 — Grounded credit.** On tasks with a computable reward, DiffTG
produces strictly positive mean `delta_R` per accepted edit, while
TextGrad (as implemented by its authors) does not reliably do so.
- **H2 — Manifold locality.** Perturbations stay close to the base LM's
manifold: decoded edits are valid tokenizations at rates >>> random, and
cos(z_0, z_K) stays in a controlled range.
- **H3 — Compositional improvement.** Iterating DiffTextGradStep at
inference improves mean reward monotonically on held-out tasks, and
REINFORCE with `delta_R` as credit gives non-trivial gains over a
frozen-BaseLM baseline.

All three are falsifiable by the experiments below.

## 1. Glossary

- **Trial** — one `difftg_step` call: one task, one rollout, one or more
span-perturbation attempts, one final accepted trajectory.
- **Attempt** — one span perturbation inside a trial. Has `orig_span`,
`new_span`, `delta_R`, `accepted`, `cos(z_0, z_K)`.
- **Run** — one full invocation of `python -m src.difftg.main <cfg> <out>`;
produces `results.jsonl` + `summary.json`.
- **Sweep** — a family of runs varying one or more axes (K, step_size,
selector, DiffLM kind, reward, task).

## 2. Datasets / benchmarks


| id               | source                                                 | split                  | reward                            | size           | hf / path                                 | status  |
| ---------------- | ------------------------------------------------------ | ---------------------- | --------------------------------- | -------------- | ----------------------------------------- | ------- |
| D-toy-digitshift | synthetic (this repo)                                  | deterministic seeded   | `ToyStringReward`                 | n configurable | `src/difftg/tasks/toy.py`                 | done    |
| D-toy-reverse    | synthetic                                              | deterministic seeded   | `ToyStringReward`                 | n configurable | `src/difftg/tasks/toy.py`                 | done    |
| D-toy-sumdigits  | synthetic                                              | deterministic seeded   | `ToyStringReward`                 | n configurable | `src/difftg/tasks/toy.py`                 | done    |
| D-gsm8k          | `openai/gsm8k`, config `main`                          | `test` (1319)          | `ExactMatchReward`                | 1319           | `src/difftg/tasks/gsm8k.py`               | done    |
| D-bbh            | `lukaemon/bbh` or `maveriq/bigbenchhard`               | per-subset             | exact-match per subset            | ~6.5k total    | todo — add `tasks/bbh.py`                 | pending |
| D-humaneval      | `openai_humaneval`                                     | `test`                 | test-pass-rate via sandboxed exec | 164            | todo — add `tasks/humaneval.py` + sandbox | pending |
| D-judgerewrite   | synthetic prompt+response pairs judged by a strong LLM | generated once, frozen | `JudgeReward`                     | 500            | todo — `tasks/judge_rewrite.py`           | pending |


Rationale: toy tasks isolate the mechanism; GSM8K validates "math
reasoning" edits; BBH validates "general reasoning" edits; HumanEval
validates "code" edits (rich token structure); judge-rewrite is the most
TextGrad-comparable setting because the original TextGrad paper uses
judges.

## 3. Experiments

### 3.1 Phase v1 — inference-only, frozen BaseLM

Goal: show the pipeline produces non-zero `delta_R` with controllable
magnitude, and that hill-climb acceptance improves reward.


| id                 | config                                   | dataset          | diff_lm  | selector | K   | step_size | tangent | n tasks | primary metric                    | status  |
| ------------------ | ---------------------------------------- | ---------------- | -------- | -------- | --- | --------- | ------- | ------- | --------------------------------- | ------- |
| E-v1-sanity-K0     | `toy_inference.yaml` override            | D-toy-digitshift | gaussian | answer   | 0   | 0.01      | true    | 200     | `R_final == R_orig` exactly       | pending |
| E-v1-gauss-digit   | `toy_inference.yaml`                     | D-toy-digitshift | gaussian | answer   | 20  | 0.01      | true    | 200     | `mean_delta_R > 0`, `accept_rate` | pending |
| E-v1-gauss-reverse | override variant                         | D-toy-reverse    | gaussian | answer   | 20  | 0.01      | true    | 200     | same                              | pending |
| E-v1-gauss-sum     | override variant                         | D-toy-sumdigits  | gaussian | answer   | 20  | 0.01      | true    | 200     | same                              | pending |
| E-v1-gauss-gsm8k   | `gsm8k_inference.yaml` + answer selector | D-gsm8k          | gaussian | answer   | 20  | 0.01      | true    | 200     | `mean_delta_R`, em accuracy gain  | pending |


Status: E-v1-gauss-digit was run once (job 169517) but with the wrong
selector (random_sentence) and the wrong noise scaling (high-d blowup).
After the noise-scaling + answer-selector fixes landing this commit, this
experiment must be re-run as the first real v1 datapoint.

### 3.2 Phase v1.5 — pluggable DiffLM backends

Goal: compare trainable vs pretrained vs Gaussian baselines on the same
tasks. This is the first place we can make research claims about "does a
learned score help".


| id                      | dataset          | diff_lm                         | pretrain source                                  | primary metric                     | status                     |
| ----------------------- | ---------------- | ------------------------------- | ------------------------------------------------ | ---------------------------------- | -------------------------- |
| E-v15-pretrain-denoiser | D-toy-digitshift | trainable                       | `scripts/pretrain_diff_lm.py` on BaseLM rollouts | score-matching loss → plateau      | pending                    |
| E-v15-trainable-digit   | D-toy-digitshift | trainable                       | ↑ ckpt                                           | `mean_delta_R` vs E-v1-gauss-digit | pending                    |
| E-v15-bert-digit        | D-toy-digitshift | pretrained (bert_reconstructor) | bert-base-uncased, frozen                        | `mean_delta_R` vs E-v1-gauss-digit | pending                    |
| E-v15-trainable-gsm8k   | D-gsm8k          | trainable                       | D-gsm8k-train rollouts                           | em accuracy gain                   | pending                    |
| E-v15-bert-gsm8k        | D-gsm8k          | pretrained (bert_reconstructor) | frozen                                           | em accuracy gain                   | pending                    |
| E-v15-plaid-gsm8k       | D-gsm8k          | pretrained (plaid)              | todo — register wrapper                          | em accuracy gain                   | blocked on plaid selection |


### 3.3 Phase v2 — REINFORCE on BaseLM

Goal: show `delta_R`-grounded REINFORCE improves BaseLM beyond frozen
baselines and beyond a non-grounded baseline (e.g. vanilla REINFORCE on
terminal reward without the diffusion proposer).


| id                       | dataset          | diff_lm   | hooks                                       | primary metric                         | status  |
| ------------------------ | ---------------- | --------- | ------------------------------------------- | -------------------------------------- | ------- |
| E-v2-policy-digit        | D-toy-digitshift | gaussian  | §4.1 only                                   | training-curve `R_final`, test-time em | pending |
| E-v2-policy-gsm8k        | D-gsm8k          | gaussian  | §4.1 only                                   | test em gain vs LoRA-frozen            | pending |
| E-v2-policy+critic-gsm8k | D-gsm8k          | gaussian  | §4.1 + §4.3                                 | variance of update, test em gain       | pending |
| E-v2-full-gsm8k          | D-gsm8k          | trainable | §4.1 + §4.2 + §4.3                          | test em gain                           | pending |
| E-v2-baseline-vanilla-rl | D-gsm8k          | n/a       | terminal-reward REINFORCE on whole sequence | test em (baseline to beat)             | pending |
| E-v2-baseline-textgrad   | D-gsm8k          | n/a       | actual TextGrad package on same LM          | test em (external baseline)            | pending |


### 3.4 Stretch


| id                 | what                                                           | status  |
| ------------------ | -------------------------------------------------------------- | ------- |
| E-str-bbh          | run best v2 config on D-bbh                                    | pending |
| E-str-humaneval    | port + run on D-humaneval (requires sandboxed exec for reward) | pending |
| E-str-judgerewrite | run best v2 config on D-judgerewrite with LLM judge            | pending |


## 4. Ablations

Each ablation is orthogonal to the experiment axis above; cross-product
would be too large, so we commit up-front to running each ablation only at
one "canonical" operating point (written in parens).


| id             | axis                                   | values                                                                 | canonical base          | status               |
| -------------- | -------------------------------------- | ---------------------------------------------------------------------- | ----------------------- | -------------------- |
| A-K            | Langevin steps                         | 0, 5, 10, 20, 40                                                       | E-v1-gauss-digit        | pending              |
| A-step_size    | Langevin step size                     | 0.001, 0.005, 0.01, 0.05, 0.1                                          | E-v1-gauss-digit        | pending              |
| A-tangent      | tangent projection on noise            | {true, false}                                                          | E-v1-gauss-digit        | pending              |
| A-numspans     | spans per trial                        | 1, 2, 4, 8                                                             | E-v1-gauss-digit        | pending              |
| A-selector     | span selector kind                     | answer, random_sentence, lowest_logprob, judge                         | E-v1-gauss-digit        | pending              |
| A-accept       | acceptance rule                        | `R_new >= R_orig` (hill climb), `R_new > R_orig`, Metropolis with temp | E-v1-gauss-digit        | pending              |
| A-reward       | reward shape                           | exact `{0, 1}`, difflib ratio, edit distance                           | E-v1-gauss-digit        | pending              |
| A-basesize     | base LM size                           | 0.5B, 1.5B, 3B, 7B                                                     | E-v1-gauss-digit        | pending (cost-gated) |
| A-dtype        | base LM dtype                          | bf16, fp16, fp32                                                       | E-v1-gauss-digit        | pending              |
| A-difflm       | DiffLM kind at fixed other axes        | gaussian, trainable, bert_reconstructor                                | E-v15-trainable-digit   | pending              |
| A-bridge       | bridge dim for pretrained adapter      | 256, 768, 1024                                                         | E-v15-bert-digit        | pending              |
| A-pretraindata | span pool source for denoiser pretrain | toy rollouts, gsm8k rollouts, mixed                                    | E-v15-pretrain-denoiser | pending              |


## 5. Baselines and competitors

- **B-null** — frozen BaseLM generation, no edits. `R_final = R_orig`.
Every v1/v1.5/v2 result must beat this on `mean_R_final`.
- **B-random-edit** — identical pipeline but `DiffLM.score == 0` and noise
scaled so the per-step arc length is matched; this IS
`GaussianNoiseDiffLM`, so it is both a baseline and a run in its own
right.
- **B-self-refine** — prompt the BaseLM with its own output and ask for a
corrected version. External baseline.
- **B-textgrad** — real TextGrad library on the same LM + task. External
baseline.
- **B-vanilla-rl** — REINFORCE on whole-sequence reward, no span proposer.
External baseline to v2.

## 6. Success criteria

A run is "healthy" and reportable if all of the following hold on its
`summary.json`; otherwise it gets re-run with a bug fix.

- `0.1 < mean(accept_rate) < 0.95` (real signal, not pure no-ops).
- `mean(cos(z_0, z_K)) in [0.2, 0.95]` (local perturbation, not uniform
sampling).
- `mean_delta_R` (over *attempts*) is non-zero on at least 20% of trials.
- K=0 run gives `R_final == R_orig` exactly.

Research claims are considered supported if:

- **H1** — `mean_delta_R > 0` on E-v1-gauss-digit and E-v1-gauss-gsm8k
with p < 0.01 (paired over tasks).
- **H2** — decoded spans re-tokenize to a valid ID sequence ≥95% of the
time on E-v1-gauss-digit, and `cos(z_0, z_K)` CI lies strictly above 0.
- **H3** — E-v2-policy-gsm8k test em beats both B-null and B-vanilla-rl
with p < 0.05.

## 7. Compute budget (rough)


| phase                           | GPU-hours                                       | basis                                                   |
| ------------------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| v1 total (sanity + toy + gsm8k) | ~10                                             | 1.5B model, 200 tasks per run, ~15 min each, 20-30 runs |
| v1.5 pretrain denoiser          | ~5                                              | 2k steps, batch 16                                      |
| v1.5 total                      | ~20                                             | 3 variants × 3 tasks, plus bert adapter                 |
| v2 total                        | ~80                                             | 2k outer steps, LoRA, incl. baselines                   |
| stretch                         | ~80                                             | BBH + HumanEval + judge                                 |
| **total target**                | **~200 GPU-hours** on a single A100/L40 per run |                                                         |


## 8. Open questions

- Which continuous-embedding diffusion LM has open weights worth wrapping?
Candidates: Plaid (Gulrajani), LD4LG, TESS, SEDD, SSD-LM. Need a
one-week spike to evaluate.
- Does tangent projection on noise matter empirically? Theoretically it
keeps samples on-manifold but may also kill exploration.
- How should the pretrained denoiser be supervised when we don't have
access to ground-truth denoising pairs? Current plan: denoising score
matching on BaseLM-generated spans, but we should also try a
contrastive "reward-improving direction" objective once v2 is running.
- Acceptance rule: strict `>` vs. `>=` vs. Metropolis — the current `>=`
admits no-op ties, which inflates `accept_rate`. A-accept will
characterize this.
- Token-boundary drift on DecodeSpan: how often does cosine argmax produce
subword pieces that re-tokenize to different ids? If often, the
trajectory text becomes unstable under a round-trip. Measure during
E-v1-gauss-digit.

## 9. Deliverables

- `outputs/<run_id>/{results.jsonl, summary.json, run.log}` per run,
immutable.
- `scripts/aggregate_sweep.py` output tables for every sweep.
- A results notebook (or static markdown report) that loads aggregated
summaries and renders the H1/H2/H3 figures. Add as
`scripts/report.py` when the first full sweep lands.

