Project: Diffusion Text Grad (DiffTG)

Purpose: Replacing Textual Gradients with Reward‑Grounded Representation Diffusion
TextGrad and related textual‑gradient methods treat natural‑language critiques as analogues of numerical gradients within a PyTorch‑like optimization loop, using LLM‑generated feedback to rewrite prompts or other text variables. While empirically effective, this mechanism is fundamentally heuristic: gradients are represented as free‑form prose, are prone to hallucinated failure modes, and induce global, non‑local edits whose causal relationship to the underlying scalar objective is difficult to analyze or control. Our goal is to replace this textual gradient metaphor with a reward‑grounded, representation‑space alternative that preserves TextGrad’s “plug‑and‑play” optimization ergonomics but bases updates on actual changes in task reward under small, localized perturbations.

Concretely, instead of asking a backward LLM to explain how a span should change, we encode that span into a continuous embedding space, apply a few steps of diffusion/Langevin dynamics restricted to this span, decode the perturbed embeddings back to text, and accept the edit only if it improves the scalar reward. This yields a local stochastic policy‑improvement step in representation space that can be dropped into the same computation‑graph abstraction as TextGrad, but with three advantages: (i) updates are grounded in measured reward differences rather than verbal speculation; (ii) perturbations stay close to the model’s learned manifold, reducing catastrophic prompt drift; and (iii) the mechanism admits a direct connection to gradient‑estimation and Langevin‑style theory in continuous spaces, opening the door to principled bias–variance analysis that textual gradients currently lack.
***

## Intuition blurb

Use a diffusion LM (or continuous denoiser) as a **local proposal operator in embedding space**: given a problematic span, encode it to embeddings, take a small number of corruption–denoise steps on those embeddings (on the unit sphere), decode back to text, and observe the reward change. This gives you a **finite-difference estimate of a local gradient direction in representation space**, analogous to a textual gradient but grounded in actual reward differences rather than verbal critique.

***

## Motivation blurb

TextGrad-style methods give you semantic “gradient directions” in natural language, but they are heuristic, can hallucinate failure modes, and make global, non-local edits. A span-local diffusion operator in embedding space:

- stays close to the model’s learned manifold,
- yields perturbations with measurable causal effect on reward,
- can be used as a stochastic policy improvement operator (or as a TextGrad drop-in) without requiring explicit backprop through the entire LLM stack.

By running **small, local Langevin-like moves** in the span’s embedding space and accepting only reward-improving denoised versions, you approximate gradient ascent on \( \mathbb{E}[R(z)] \) with respect to that span.

***

## Rough algorithm sketch (pseudocode)

Assume:

- `BaseLM`: main AR policy / agent that produces trajectories.
- `DiffLM`: continuous-span diffusion model operating on normalized embeddings.
- `EncodeSpan`, `DecodeSpan`: deterministic encoder/decoder between text spans and embeddings.
- `SelectSpan`: external module that returns spans needing credit (abstracted away).
- `R(trajectory)`: scalar reward / judge score, differentiable or not.
- All embeddings are L2-normalized; cosine similarity is the intrinsic metric.

### High-level training loop

```python
for iteration in range(num_outer_steps):

    # 1. Sample task and generate base trajectory
    x_task = sample_task()
    traj_text = BaseLM.generate(x_task)          # full text / CoT / tool trace

    # 2. Evaluate original reward
    R_orig = R(traj_text)

    # 3. Select one or more spans for local optimization
    spans = SelectSpan(traj_text, R_orig)        # [(start, end), ...]

    for span_idx, (s, e) in enumerate(spans):

        # 3a. Encode span into continuous embeddings on unit sphere
        span_tokens = traj_text[s:e]
        z0 = EncodeSpan(span_tokens)             # shape [L_span, d]
        z0 = normalize(z0)                       # per-token L2 norm = 1

        # 3b. Run K-step local diffusion / Langevin on embeddings
        z = z0.clone()
        for t in range(K):
            eps = sample_gaussian_noise(z.shape)         # N(0, I)
            eps = project_to_tangent(eps, z)             # keep on sphere if desired

            # one-step denoising update; step_size, noise_schedule chosen externally
            z = z + step_size * DiffLM.score(z, t) + sqrt(2 * step_size) * eps
            z = normalize(z)                             # re-project to unit sphere

        z_perturbed = z

        # 3c. Decode perturbed span back to text
        span_tokens_new = DecodeSpan(z_perturbed)        # local edit candidate
        traj_text_new = replace_span(traj_text, s, e, span_tokens_new)

        # 3d. Evaluate new reward
        R_new = R(traj_text_new)

        # 3e. Compute local improvement signal
        delta_R = R_new - R_orig

        # 4. Use delta_R as a local credit / gradient estimate

        # 4.1 Policy improvement for BaseLM (REINFORCE-style on span logits)
        if delta_R > 0 or accept(delta_R, temperature):
            # log_prob_span is log pi_theta(span_tokens | context) from BaseLM
            log_prob_span = BaseLM.logprob_span(traj_text, span_tokens, s, e)
            loss_policy = -delta_R * log_prob_span
            loss_policy.backward()              # update BaseLM parameters

        # 4.2 Optional: fine-tune DiffLM as a local optimizer
        # Encourage DiffLM to move z0 toward z_perturbed when delta_R > 0
        if delta_R > 0:
            loss_diff = cosine_dist(DiffLM.sample(z0), z_perturbed)
            loss_diff.backward()                # makes DiffLM propose similar moves

        # 4.3 Optional: train a span-level critic in embedding space
        # Critic(z0, context) ~ E[Delta R | z0]
        loss_critic = (Critic(z0, context) - delta_R)**2
        loss_critic.backward()

    # 5. Optimizer step for all updated modules
    optimizer.step()
    optimizer.zero_grad()
```

### Inference-time “TextGrad alternative” (no parameter updates)

```python
def DiffTextGradStep(BaseLM, DiffLM, x_task, num_spans=1, K=small):

    traj_text = BaseLM.generate(x_task)
    R_orig = R(traj_text)

    spans = SelectSpan(traj_text, R_orig)[:num_spans]

    for (s, e) in spans:
        z0 = normalize(EncodeSpan(traj_text[s:e]))

        z = z0.clone()
        for t in range(K):
            eps = sample_gaussian_noise(z.shape)
            eps = project_to_tangent(eps, z)
            z = z + step_size * DiffLM.score(z, t) + sqrt(2 * step_size) * eps
            z = normalize(z)

        span_tokens_new = DecodeSpan(z)
        traj_candidate = replace_span(traj_text, s, e, span_tokens_new)
        R_new = R(traj_candidate)

        if R_new >= R_orig:      # simple hill-climb acceptance
            traj_text, R_orig = traj_candidate, R_new

    return traj_text, R_orig
```

***

### Key points relative to your constraints

- The decoder is fully abstracted into \(R(\cdot)\): all gradients/credit are defined in terms of changes in reward under local embedding perturbations.
- Geometry is handled via normalization and (optionally) tangent-space projection, so diffusion/Langevin steps are “in cosine space.”
- Locality is enforced by restricting diffusion to the selected span’s embeddings only.
- This gives you a **TextGrad-like outer loop** (evaluate → local update → accept/improve) but replaces textual gradients with **reward-grounded continuous perturbations** in representation space.




1. When do we train anything?
Think in three phases:

Prototype / ablation phase (no new training)

Use an existing base LM (frozen).

Use an off‑the‑shelf diffusion LM in embedding space (or a continuous denoiser) trained for generic text reconstruction / generation.

Implement the local span‑diffusion loop and treat it purely as a test‑time optimizer: given a prompt/trajectory, locally diffuse span embeddings a few steps and accept if reward improves.

This phase is analogous to TextGrad using GPT‑4o as backward engine: you don’t train the engines, you just use them.

Optimizer‑tuning phase (light training)
Once the mechanism works, you can start training auxiliary modules that make it better:

Fine‑tune the span diffusion model (DiffLM) to propose edits that tend to improve reward.

Train a span‑level critic in embedding space to predict expected Δreward for a span perturbation.

Optionally fine‑tune the base LM with REINFORCE on accepted edits (like RLHF but localized).

Full system training (if you want a TextGrad‑replacement library)

You’d package: a base LM, a span selector, a span denoiser, and a critic, all pre‑trained and tuned for a set of tasks.

Users then plug in their losses and you only run inference‑time optimization, just as TextGrad users don’t train the backward engine themselves.