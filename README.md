# DiffTG — Diffusion TextGrad

Reward-grounded, span-local embedding-space alternative to TextGrad.
See [spec.md](spec.md) for motivation and algorithm.

## Install

```bash
python -m venv llm
source llm/bin/activate
pip install -r requirements.txt
```

## Quick run (v1: inference-only, toy task)

Local / CPU:

```bash
python -m src.difftg.main configs/toy_inference.yaml outputs/toy_inference
```

GPU via Slurm (adapts `sample_sbatch.sh`):

```bash
sbatch sbatch/run_difftg.sh configs/toy_inference.yaml outputs/toy_inference
```

## Modes

- `configs/toy_inference.yaml` — v1: frozen BaseLM + `GaussianNoiseDiffLM` + random span select.
- `configs/toy_train.yaml` — v2: same as above + REINFORCE with LoRA on BaseLM.
- `configs/gsm8k_inference.yaml` — stretch: GSM8K with exact-match reward.

## v1 validation sweep

```bash
python scripts/make_sweep.py       # writes configs/_sweep/*.yaml and sbatch/_sweep_submit.sh
bash sbatch/_sweep_submit.sh       # submits one Slurm job per variant
# ... wait for jobs to finish ...
python scripts/aggregate_sweep.py outputs/toy_inference_sweep
```

The K=0 run is a sanity check: Langevin runs zero steps, so `R_final == R_orig`
up to decoding artifacts.

## Layout

- `src/difftg/base_lm.py` — HuggingFace causal LM wrapper; `generate`, `encode_span`, `decode_span`, `logprob_span`.
- `src/difftg/diff_lm.py` — `GaussianNoiseDiffLM`, `TrainableEmbedDenoiser`, `PretrainedDiffLMAdapter`.
- `src/difftg/langevin.py` — K-step on-sphere Langevin.
- `src/difftg/span_select.py` — random / lowest-logprob / judge span selectors.
- `src/difftg/reward.py` — `ToyStringReward`, `ExactMatchReward`.
- `src/difftg/difftg_step.py` — v1 inference-time DiffTextGradStep.
- `src/difftg/trainer.py` — v2 REINFORCE loop.
- `src/difftg/tasks/toy.py`, `src/difftg/tasks/gsm8k.py` — task generators.
- `src/difftg/main.py` — CLI.

## Design rules

- Fail fast and loudly. No defensive programming.
- No test scripts; validate via real runs.
- Inference-time pipeline (v1) has no parameter updates.
