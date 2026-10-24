# New proposals — round 2 (Jul 17, 2026)

> Translated from the Italian original. This is the planning document written
> *before* the round-2 experiments; the expected numbers below are hypotheses,
> and the measured outcomes are in [lab_log.md](lab_log.md). In short: Idea 1
> was falsified in its proposed form, but its autopsy led to the full-predicate
> re-rank that became stage 2 of the final system; Idea 2 stopped at its Gram
> diagnosis; Ideas 3, 4 and 5 were not run in the development campaign (a
> sparse-autoencoder analysis, related to Idea 4, was later added to the report
> notebook as an appendix).

Five ideas, ordered by (expected headroom × probability of surviving the
kill-test). All lightweight, all with a death gate declared before starting,
all tuned only on the mirror. The numbers quoted are those already measured in
the lab log; the expected numbers are hypotheses, marked as such.

**Common thread:** the accumulated negative results all point in the same
direction. The GT rule is a **conjunction** (all constraints satisfied) +
**identity locality** (Hamming ≤2), while every method tried so far compresses
the query into **a single vector** and ranks by cosine — that is, a weighted
sum, not an AND. The 1/√k dilution (P4), the diffuse negation that InfoNCE does
not activate (P1), lipstick at 0.000 and the fact that the learned negative
directions are not the opposite of the positive ones are all symptoms of the
same structural limit. Ideas 1–2 attack that limit; the others work around it.

---

## Idea 1 — Conjunctive verification with probes (retrieve-then-verify)

**Thesis:** stop synthesizing the perfect query; verify the constraints on the
candidate. DirectionFusion retrieves the top-M, then re-rank with:

```
S(x) = cos(q_DF, x) + Σ_{c∈C} w_sign(c) · log σ(±probe_c(x))
```

using the probes already calibrated in the veto experiment (benchmark mean AUC 0.968).

**Why it can beat 0.296:**
- The veto is a special case of this family (negated conditions only, hinge
  form): its oracle ceiling **+0.027 is a lower bound** on the ceiling of this
  formulation, which also rewards the positives.
- It removes diffuse negation *without training*: the Young probe (0.912)
  directly penalizes young faces for `-Young`. The veto failed because of
  learning (λ≈0 with uniform InfoNCE), not because of the probes — here the 2–3
  scalars (w⁺, w⁻, M) are chosen with a grid on the mirror, no InfoNCE.
- It removes dilution *by construction*: each condition has its own term,
  independent of k. No vector sum to dilute.
- **It may unlock lipstick**: in additive arithmetic `+Lipstick −Makeup` the two
  correlated directions cancel (consistent with 0.000 under every method, even
  the oracle veto). Two separate probes read the conjunction if the information
  is linearly present — and the P3 diagnosis says it is.

**Variants to try (one at a time):**
- (a) identity on the complement — cos(Pv_ref, Px) with P projecting out
  span{w_c}: the source violates *all* constraints by construction, so the raw
  cosine works against us exactly on the attributes to flip. Note: B4/H2 tried
  the erasure on the query side only; here it is two-sided and the constraints
  go through the probes.
- (b) z-normalization of the probe logits over the gallery (transductive
  calibration, in the same legitimate spirit as QB-Norm).

**Kill-test:** grid {w⁺, w⁻} × M ∈ {50,100,200} on the mirror, ~1h. Death: macro
≤ 0.296 over the whole grid. Mechanistic check: the violator rate in the top-10
on the negative-dominant queries must drop below the measured 25.5%.

**Protocol compatibility:** R@K depends only on the ranking, and the batched
assert-equivalent harness accepts a score matrix (the veto already did). The
guidelines explicitly list *cross-attention networks*, i.e. candidate-conditioned
scoring: the generalized form is sanctioned by the assignment.

**Expected (hypothesis):** mirror 0.31–0.33.

## Idea 2 — Minimum-norm edit with target margins (solve, don't add)

**Thesis:** addition superimposes directions that interfere: if
cos(d_lipstick, d_makeup) is high, `+d_lip − d_makeup ≈ 0` — the edit destroys
itself. Replace the sum with the constrained problem:

```
e* = argmin ‖e‖²  s.t.  ⟨w_c, v_ref + e⟩ = τ_c  ∀c ∈ C
```

Closed form with a k×k solve on the Gram matrix of the directions (k ≤ 3):
`e = W_Cᵀ(W_C W_Cᵀ)⁻¹(τ − W_C v_ref)`. A single hyperparameter τ (in units of
probe margin) on the mirror. Base: the paired directions of P2 (0.257
training-free, the best).

**Why it can beat 0.257 training-free:** it reaches *every* margin exactly even
with correlated directions (it computes the orthogonalized components), and the
margins do not dilute with k. It attacks interference + dilution while staying
a single query vector — maximally safe with respect to the harness if the
instructor required pure `query_features`.

**Preliminary diagnosis (10 min, before anything else):** Gram matrix of the
paired directions over the 12 benchmark attributes. Prediction:
|cos|(Lipstick, Heavy_Makeup) ≥ 0.6 and |cos|(Male, Mustache) high (it would
also explain `-Male,-Mustache`). Death of the thesis: all co-occurring pairs
|cos| ≤ 0.3 → interference explains nothing, only the dilution fix remains.

**Expected (hypothesis):** 0.27–0.29 training-free; as an operator inside
DirectionFusion probably absorbed (lesson of P2), but the training-free row is
worth reporting on its own.

## Idea 3 — Counterfactual negatives (near-miss curriculum)

**Thesis:** the current sampler has mixed but untargeted negatives; diffuse
InfoNCE is blind to weak signals (measured: λ≈0 on -Young/-Heavy_Makeup despite
oracle headroom +0.065/+0.050). Oversample negatives that **violate exactly the
negated attribute** and are otherwise close to the target profile — mined once
from the train labels with the 40-bit group machinery already written for P2.

**Implementation:** only the sampler changes; same architecture, same 3000
steps, sweep {p_counterfactual ∈ {0.25, 0.5}} × lr on the mirror. Downstream
mechanistic check: retraining the veto on top, λ_Young and λ_Heavy_Makeup must
become >0 (the thesis predicts that the gradient now activates them).

**Kill-test:** mirror ≤ 0.296 over the whole grid. Cost: ~6 short runs.

**Expected (hypothesis):** 0.30–0.315 — captures through the loss part of the
+0.027 that the learned veto did not take, and unlike the veto it also moves
the synthesis.

## Idea 4 — Lightweight sparse dictionary: negation as ablation (SAE)

**Thesis:** lipstick/makeup entanglement and non-antipodal negatives (cos ≤ 0.53)
are symptoms of superposition in the 512-d space. An SAE 512→2048 (TopK, on
train, minutes on a 4070) separates the atoms; then **negation = zeroing the
latents of the attribute** (an operation that *is not* a direction — consistent
with the finding), positive = boost. The edit goes through the decoder, v_ref
stays intact: `q = normalize(v_ref + Dec(Δz))`.

**Very hard death gate, before any ablation (30 min):** for the 12 benchmark
attributes, AUC of the best single latent vs the probe. Death: mean < 0.8·probe
AUC, or no latent with AUC_lipstick ≥ 0.8 ∧ AUC_makeup ≤ 0.6 (no
lipstick-specific atom). P3 style: if it dies, it dies in half an hour.

**Why keep it:** it is the only approach in the list cited *explicitly* in the
guidelines (SAEs), the risk is high but the cost of finding out is minimal, and
even the negative ("superposition persists in the sparse basis at this scale")
is report material.

## Idea 5 — Rank fusion (RRF) of the complementary methods

**Thesis:** the per-query complementarity is already measured: paired-B3 wins
the hat triple (0.854 vs 0.793) and lipstick (0.057 vs 0.000), DirectionFusion
wins the positive-dominant queries. Reciprocal Rank Fusion between DF and
paired-B3 (and the Idea 1 scorer once it exists), **a single global weight** on
the mirror — no 13 per-query choices, which would be fitting the benchmark.

**Kill-test:** ≤ 0.296. Cost: minutes, zero training. Expected: +0.005–0.015.
Little glory, but it is the glue of the final system.

---

## Summary and order of attack

| # | Idea | Bottleneck attacked | Cost | Death gate | Mirror potential (hypothesis) |
|---|---|---|---|---|---|
| 1 | Conjunctive verification with probes | diffuse negation + dilution + lipstick | ~1h grid | ≤0.296 everywhere | 0.31–0.33 |
| 2 | Margin solve (minimum norm) | interference between conditions + dilution | ~30 min | flat Gram / ≤0.257 | 0.27–0.29 TF |
| 3 | Counterfactual negatives | violators in the top-10, via the loss | ~6 runs | ≤0.296 | 0.30–0.315 |
| 4 | SAE: negation as ablation | superposition/entanglement | ~1h | latent AUC < 0.8·probe | high risk |
| 5 | RRF | per-query complementarity | minutes | ≤0.296 | +0.005–0.015 |

**Order:** Gram diagnosis of idea 2 (10 min, it also informs the lipstick story
of idea 1) → idea 1 → idea 3 → idea 5 → idea 4 if time allows. Endgame if more
than one survives: DF retrained with counterfactuals → top-M → conjunctive
verification → RRF with paired-B3. If the chain holds, mirror 0.32+ is plausible.

**Thresholds declared before starting:** a component is adopted only if
Δ mirror ≥ +0.010 (below that it is noise: the lesson of the +0.002 of P2); the
**second touch of the test** — the first was spent on DirectionFusion 0.300 —
happens only once, on the complete final system, and only if Δ mirror ≥ +0.02.
Team decision pending: the course protocol wants the test touched once; to be
assessed whether "once for the new final model" is defensible or should be
asked to the instructor.

**Framing for the report:** "composition as verification, not synthesis". With
that thesis, the five falsified proposals and H1–H5 stop being a list of dead
attempts and become the systematic evidence that motivated the change of
paradigm.
