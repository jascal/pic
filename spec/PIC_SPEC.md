# PIC: Projective Incidence Calculus — Definition and Semantics

**Status:** canonical specification (v0.1). This document defines PIC precisely enough to serve all
three repos of the program at once:

| repo | role | uses PIC as |
|------|------|-------------|
| [`fieldrun`](https://github.com/jascal/fieldrun) | runtime analysis of frozen models | the **spec of what the binary computes** (the residual reading in the log-semiring) |
| [`i-orca`](https://github.com/jascal/i-orca) | Isabelle kernel proofs | the **structures the theorems are about** (frame, Gram, semiring, margin) |
| [`pil`](https://github.com/jascal/pil) | learning dynamics on the frame | the **optimization target and invariants** the Generate→Gate→Refine loop must preserve |

Every theorem cited here is **kernel-proved in i-orca** unless explicitly tagged *(empirical)* or
*(open)*. Every operational quantity is **as implemented** in `pil/pil/geometry.py` /
`pil/pil/scoring.py` / `pil/pil/learner.py` and `fieldrun/src/recursion_probe.rs`; the section that
defines it gives the file. PIC is descriptive, not aspirational: it is the algebra the existing
analysis already runs in and the existing proofs already quantify over.

---

## 0. One-paragraph summary

A transformer's next-token decision is a **semiring-weighted incidence calculus over a continuous
inner-product geometry**. Propositions (tokens) are points `U_v` in a residual space `H = ℝ^d`;
sources (DLA blocks, or learned rules) are vectors `d_j`; the incidence of source `j` on proposition
`v` is the inner product `c_j(v) = ⟨d_j, U_v⟩`; and the decision aggregates incidences over sources
through a semiring `⊕` selected by a temperature `T`. At `T = 1` the semiring is the log-semiring and
PIC **is** the transformer forward pass (softmax over summed logits); at `T → 0` it is the tropical
(max-plus) semiring and PIC is the geometry of margins, Laguerre power diagrams, and the head/tail
decode certificate. The calculus cleanly **separates the frame** (the intrinsic geometry of `{U_v}`:
Gram coherence, Welch floor, packing capacity) **from the decode** (margins, participation ratio,
multiplicity `μ_t`, evaluated in the chosen semiring). That separation is the load-bearing move: the
**analysis** repos read the decode side off a frozen frame, while the **learning** repo (`pil`) holds
the decode semantics fixed and *retrains the frame* to make more structure retrievable.

---

## 1. Core structures

### 1.1 Frame, propositions, Gram kernel

Let `V` be a finite set of **propositions** (tokens), `|V| = nₚ`. Let `H` be a real inner-product
space, in practice `H = ℝ^d` (the residual / unembedding space).

A **PIC frame** is a map

> `U : V → H,   v ↦ U_v`

(the unembedding directions). It induces the **Gram kernel**

> `G(v, w) = ⟨U_v, U_w⟩`            (`geometry.py:gram_matrix`, `G = U Uᵀ`)

and, after row-normalisation `Ū_v = U_v / ‖U_v‖`, the **cosine Gram** `ρ(v,w) = ⟨Ū_v, Ū_w⟩`
(`geometry.py:cosine_gram`). `G` supplies the *non-truth-functional overlap* between propositions:
two tokens are not independent atoms but geometric directions with a measurable coherence.

### 1.2 Sources and contributions

A **source set** `S` (`|S| = J`) carries, per source `j`, a **contribution function**
`c_j : V → R`, where `R` is the carrier of a semiring (§2). The **residual reading** — the case that
makes PIC a model of transformers — sets a vector `d_j ∈ H` per source and defines

> `c_j(v) = ⟨d_j, U_v⟩`.

In `fieldrun`, the sources are the **DLA blocks**: for an `L`-layer model there are `nb = 2L + 1`
blocks (one attention output and one MLP/expert output per layer, plus the embedding), and
`d_j` is block `j`'s additive contribution to the residual stream at the decode position. The
incidence matrix `c_j(v) = ⟨d_j, U_v⟩` is exactly `contrib[block][cand]` emitted by
`fieldrun --pil-dump` (`recursion_probe.rs:run_pil_dump`). In `pil`, sources are either real DLA
blocks (loaded through the `--pil-dump` seam) or synthetic/learned propositions
(`learner.py:propose_parallel_sources`).

### 1.3 Aggregate (the decision)

The **aggregate score** of proposition `v` is the semiring sum of its incidences, plus a bias:

> `L(v) = ( ⨁_{j∈S} c_j(v) ) ⊗ b_v`,   classically `L(v) = Σ_j ⟨d_j, U_v⟩ + b_v`

(`geometry.py:logits_from_incidences` sums over the source axis). The **decision** is
`argmax_v L(v)` (decode), or the softmax distribution `p(v) ∝ exp L(v)` — the two are the tropical
and log-semiring readings of the *same* `L`.

---

## 2. The semiring family (the calculus parameter)

PIC is parameterised by a **temperature** `T ≥ 0` selecting a commutative semiring
`(R_T, ⊕_T, ⊗, 0, 1)` with the **same** product `⊗ = +` (real addition; incidences along a coalition
*add*) and a `T`-dependent sum:

| `T` | semiring | `a ⊕_T b` | reading | where it shows up |
|-----|----------|-----------|---------|-------------------|
| `1` | **log-semiring** | `log(eᵃ + eᵇ)` (logsumexp) | probabilistic incidence → softmax | the transformer forward pass; `fieldrun` decode |
| `→ 0⁺` | **tropical** (max-plus) | `max(a, b)` | argmax, margins, power diagrams | `i-orca/HeadTail`, `DecodeCapacity`; decode geometry |
| general | **deformed** | `T · log Σ exp(·/T)` | temperature-controlled decoding | interpolation / annealing |
| (Boolean, Viterbi) | special cases | `∨`, `max` | logic / min-cost gating | rule gating in `pil` |

The family is connected by **Maslov dequantization**: `T·log(e^{a/T} + e^{b/T}) → max(a,b)` as
`T → 0⁺`. So the probabilistic calculus (the model in operation) and the geometric calculus (the
proofs) are the **two ends of one deformation**, not two different objects. The tropical end is a
genuine semiring: `TropicalSemiring.thy` proves `tadd a b = max a b`, `tmul a b = a + b`, and the
distributive law `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)` — the law that turns a sum-of-maxes (a ReLU stack)
into a max-of-sums (a tropical polynomial).

---

## 2.5 PIC-native notation (what does *not* come for free)

PIC deliberately **borrows** its carriers — the semiring `(⊕_T, ⊗)`, provenance semantics, the Gram
kernel, the Welch floor, power diagrams — so that it inherits their theorems verbatim (this is why the
i-orca proofs are short: they are near-standard once the structure is named). But three things are *not*
in any of those source algebras, and they get native notation here so derivations read as one-liners
instead of raw inner products and argmaxes.

**(N1) The incidence arrow** — the one genuinely new primitive. Provenance semirings take the annotation
of a tuple as *given*; PIC *derives* it from geometry. Write

> `j ▷ v  :=  ⟨d_j, U_v⟩`            ("source `j`'s incidence on proposition `v`" / "`j` votes for `v`")

So the new content is precisely `▷ = ⟨·,·⟩∘(d,U)`: **the provenance weight is an inner product in a
learnable frame.** Everything downstream is bookkeeping over `▷`.

**(N2) The coalition bracket** — semiring aggregation of a coalition `P ⊆ S` at temperature `T`:

> `⟦P⟧_T(v)  :=  ⨁_{T, j∈P} (j ▷ v)`,     and the full reading   `L_T(v) := ⟦S⟧_T(v) ⊗ b_v`.

`⟦·⟧_1` is the log-semiring (softmax) reading; `⟦·⟧_0` is the tropical (max) reading. The decode is
`⊤_T(P) := argmax_v ⟦P⟧_T(v)`.

**(N3) The decision turnstile** — the native gadget that makes the trichotomy of §3 one line each:

> `P ⊢_γ v   :⟺   ⟦P⟧(v) − max_{w≠v} ⟦P⟧(w) ≥ γ`            ("coalition `P` decides `v` with margin ≥ `γ`")

(write `P ⊢ v` for `γ = 0`). This is exactly `Separation.thy`'s `decides` predicate, now margin-indexed,
and it unifies the regimes:

| regime | in turnstile notation | diagnostic |
|--------|------------------------|------------|
| **retrieved** | `∃ j∈S.  {j} ⊢_γ v` (a singleton decides) | high `μ_t` |
| **composed** | `S ⊢ v`  but  `∀ j∈S. ¬({j} ⊢ v)` | `μ_t = 0` |
| **irreducible** | `S ⊢ v`  and  `∀ P ⊊ S, P≠∅.  ¬(P ⊢ v)` | minimal coalition |
| **γ-decodable** (§5.1) | `∃ r, ‖r‖≤1.  S ⊢_γ v` at residual `r` | frame-side |

**(N4) Side tags** — mark a quantity `[·]_F` frame-side (function of `{U_v}` only) or `[·]_D` decode-side
(function of `L` and a target). E.g. `[FP]_F`, `[μ_t]_D`. The whole `pil` program is: move `[·]_F`
without disturbing `[·]_D`.

So the answer to "does PIC just relabel?" is: the **carriers** are borrowed (deliberately, to inherit
theorems), but `▷` (continuous provenance), `⊢_γ` (the margin turnstile / trichotomy), and the `F/D`
split are native — they are what a tropical-geometry or provenance-semiring paper does **not** give you.

---

## 3. PIC as a calculus

**Syntax.** Sources are weighted clauses; a **derivation** of `v` is a finite **coalition**
`P ⊆ S` whose incidences combine, under `⊗` within the coalition and `⊕` across alternatives, to
support `v`.

**Semantics.** Semiring **provenance** in the style of Green–Karvounarakis–Tannen: the value of `v`
is an element of `R_T` accumulated over derivations, and the Gram kernel `G` supplies the continuous
overlap that a Boolean provenance semiring lacks. The frozen-model reading is the log-semiring
provenance; the geometric reading is the tropical provenance.

**Inference / decision rules** (in the native notation of §2.5):
- *Aggregate:* `L_T(v) = ⟦S⟧_T(v) ⊗ b_v`; decode `⊤_T(S) = argmax_v ⟦S⟧_T(v)`.
- *Retrieved:* `∃ j∈S. {j} ⊢_γ v` — a singleton coalition already decides `v` (high `μ_t`).
- *Composed:* `S ⊢ v` but `∀ j∈S. ¬({j} ⊢ v)` — the decision is carried by a coalition, no single
  source (`μ_t = 0`).
- *Irreducible:* `S ⊢ v` and `∀ P ⊊ S, P≠∅. ¬(P ⊢ v)` — no proper non-empty sub-coalition decides
  (`Separation.thy`).

**Soundness (the two anchors).**
- *Log-semiring ⇒ transformer.* Instantiated with inner-product incidences in the log-semiring, the
  calculus **is** next-token behaviour: `fieldrun`'s `residual_decomp` reconstructs the model's decode
  argmax with measured recon `= 1.00` on every architecture tested (rope, neox, MoE, MLA). PIC is not
  an approximation of the forward pass; it is the forward pass rebracketed as an incidence sum.
- *Tropical ⇒ power diagram.* In the `T → 0` limit the decision `argmax_v ⟨r, U_v⟩ + b_v` is a
  **Laguerre (power) diagram** cell membership over the sites `{U_v}` with weights `{b_v}`; the
  capacity and separation theorems of §5 are the geometry of that diagram.

---

## 4. Frame vs decode separation

The central structural distinction. Every PIC quantity is either **frame-side** (a function of `{U_v}`
alone — intrinsic geometry, no data, no targets) or **decode-side** (a function of the aggregate `L`
and a target — evaluated in the chosen semiring). `pil` exploits exactly this split: it can move a
frame-side objective without touching decode semantics, and vice versa.

### 4.1 Decode-side quantities (data-dependent, semiring-evaluated)

| quantity | formula | file |
|----------|---------|------|
| **logit** `L(v)` | `Σ_j ⟨d_j, U_v⟩ + b_v` | `geometry.py:logits_from_incidences` |
| **margin** `m(t)` | `L(t) − max_{v≠t} L(v)` (top-1 vs top-2, or target-vs-worst) | `geometry.py:margin_to_worst`; `recursion_probe.rs` (`logits[0]−logits[1]`) |
| **participation ratio** `PR` | `(Σ_j |w_j|)² / Σ_j w_j²` over incidence magnitudes `w_j = |c_j(t)|` | `geometry.py:participation_ratio` |
| **multiplicity** `μ_t` | `#{ j : argmax_v c_j(v) = t }` — sources whose *own* argmax is the target | `recursion_probe.rs:run_source_pr_dump` |
| **ambiguity** `η²` | between-class / total variance of candidate firing on the lowest-margin slice | `scoring.py:ambiguity_resolution_score` |

`PR` is the **effective number of contributing sources**: `PR = 1` is the retrievable regime (a single
source decides), high `PR` is the diffuse "computed / forge-tax" regime with no small sufficient
support. `μ_t` is the **unanimity** count: high `μ_t` = retrieved, `μ_t = 0` = composed.

> **Measurement note (architecture-fair PR).** When sources carry a large common-mode component
> identical across candidates (e.g. the neox LayerNorm/embedding offset), raw `PR(c_·(t))` is inflated
> by a term that cancels in the argmax. The discriminative effective-source count centres each source
> across candidates first: `PR( (c_j(t) − mean_v c_j(v))_j )`. Raw and centred agree for RMSNorm frames
> (small common-mode) but differ for LayerNorm; report the centred form for cross-architecture
> comparison. *(See `pil/experiments/tau_star_entropy.py`.)*

### 4.2 Frame-side quantities (intrinsic, label-free)

| quantity | formula | file |
|----------|---------|------|
| **Gram coherence** | `max_{v≠w} |ρ(v,w)|` (max off-diagonal cosine) | `geometry.py:cosine_gram` |
| **frame potential** `FP` | `mean_{v≠w} ρ(v,w)²` | `geometry.py:frame_potential` |
| **Welch floor** `W` | `(nₚ − d) / (d (nₚ − 1))` if `nₚ > d`, else `0` | `geometry.py:welch_bound` |

`FP ≥ W` always (the Welch bound is the floor of mean-squared off-diagonal coherence for an
over-complete frame `nₚ > d`); the ratio `FP / W` reports frame quality, `= 1` at a Welch-optimal
(equiangular tight) frame. These are the quantities `pil`'s frame-side objective drives.

---

## 5. The theorems (what is proved)

All statements below are **kernel-proved in i-orca** at the cited file. They split into the two sides
of a single **two-sided packing** picture: the **frame side** bounds how many tokens a geometry can
decode; the **generator side** bounds how many rules can write the logits and what interference that
forces.

### 5.1 Frame side — decode capacity (`tropical/DecodeCapacity.thy`)

`v` is **γ-decodable** if some unit-ball residual `r` (`‖r‖ ≤ 1`) makes it win by margin `γ`:
`∀w≠v.  ⟨r, U_w⟩ + b_w + γ ≤ ⟨r, U_v⟩ + b_v`.

- **`margin_pair_separation`** *(proved).* Two γ-decodable tokens have separated frames:
  `γ ≤ ‖U_v − U_w‖`. **The biases cancel** when the two witness inequalities are added; Cauchy–Schwarz
  and `‖r_v − r_w‖ ≤ 2` finish it.
- **`head_capacity`** *(proved corollary).* Any set `S` of γ-decodable tokens (in particular any
  certifiable **head**) is a **γ-code**: pairwise `‖U_v − U_w‖ ≥ γ`. By the packing bound for
  γ-separated codes in `ℝ^d`,  **`|S| ≤ (1 + 2ρ/γ)^d`**, where `ρ = max_v ‖U_v‖`.

So a γ-margin decoder can keep at most `(1+2ρ/γ)^d` tokens cleanly separable — the **cell-capacity**
half of the two-sided bound (slack is astronomical for realistic `d`; this side is rarely binding).

### 5.2 Generator side — routing rank (`tropical/RoutingRank.thy`)

`M` trainable rules write logit adjustments along fixed readout vectors `{a_1,…,a_M}`; whatever the
input-dependent gates `h_k(z)` do, the adjustment lies in their span.

- **`routing_superposition`** *(proved).* `Σ_{i∈I} h_i · a_i ∈ span(a_I)` **and**
  `dim span(a_I) ≤ |I| = M`. Every rule adjustment lives in a fixed subspace of dimension
  `≤ min(M, d)`. **Superposition is forced when `n > M`** (more routing decisions than rules).

### 5.3 Generator side — routing Welch (`superposition/RoutingWelch.thy`, via `Welch.thy`)

Each routing decision realises a feature `f_c ∈ ℝ^M`; with `n` unit features in `M` coordinates this is
the Welch setup on the rule-activation side.

- **`welch_sos` / `welch_offdiag`** *(proved, in `Welch.thy`).* Total off-diagonal squared coherence
  `Σ_{i≠j} ⟨f_i, f_j⟩² ≥ n(n − M)/M` for unit-norm features — strictly positive exactly when `n > M`.
- **`routing_capacity` / `routing_interference_welch`** *(proved corollaries).* Cross-talk-free routing
  needs `M ≥ n`; otherwise interference grows as the `M`-dim rule bank is overpacked.
- **`recon_error_eq_interference`** *(proved, `Superposition.thy`).* For a unit feature, reconstruction
  error equals total squared interference — geometry **is** the loss.
- *(empirical, open)* The step **"routing interference ⇒ margin degradation"** is **not** a kernel
  theorem. Trained models pack features at ≈ the Welch floor with healthy margins; the
  coherence→margin link is measured and mild (`pil` docs §5f/§5g), not proved.

### 5.4 The decode certificate — head/tail (`tropical/HeadTail.thy`)

With `decode(L,S) = max_{v∈S} L(v)` (the tropical aggregate over a token set):

- **`decode_partition`** *(proved).* `decode(L, H∪T) = max(decode(L,H), decode(L,T))`.
- **`head_certifies_decode`** *(proved).* If `decode(L,T) ≤ decode(L,H)` then
  `decode(L, H∪T) = decode(L,H)` — a **head** that dominates the **tail** reproduces the decision.
- **`head_argmax_in_head`** *(proved).* …and the argmax token lies in the head.
- **`tail_is_residue`** *(proved).* If the head does **not** dominate, the decision lies in the tail —
  the explicit, uncertified **residue** (the forge-tax tokens).

This is the algebra behind the empirical decode-circuit finding: a **per-position sparse head** (median
1–3 late-MLP blocks) reproduces the decode exactly *when* it dominates the tail; harder tokens push mass
into the tail residue.

### 5.5 Decision-side robustness — margin certificate (`provable_opt/ProvableOpt_Common.thy`)

`margin(L,V,t) = L(t) − max_{v≠t} L(v)`.

- **`decode_margin_certified`** *(proved, threshold tight).* If a perturbation is `δ`-bounded per token
  (`|L'(v) − L(v)| ≤ δ`) and `margin(L,V,t) > 2δ`, then `t` is still the strict argmax under `L'`. The
  `2δ` bound is **tight** — the file exhibits a `margin = 2δ` token where a `δ`-perturbation ties. The
  certificate is **silent** for `margin ≤ 2δ` (exactly the small-margin / forge-tax tokens): a sound
  *local* certificate, not a global one.

### 5.6 Lossless restriction — demand closure (`provable_opt/ProvableOpt_Common.thy`)

For a monotone one-step consequence operator `T` and a demand set `D`, `T` is **demand-closed** if
`T(S) ∩ D = T(S ∩ D) ∩ D` (producers of demanded atoms read only demanded atoms).

- **`demand_restrict_lfp`** *(proved).* `lfp(restrict_op T D) = lfp(T) ∩ D` — running the restricted
  program computes exactly the demanded part of the full model. Instance: dropping every non-final-layer
  accumulator preserves the decode (`Logit` derivable iff it was), a non-vacuous saving for `L ≥ 1`.

### 5.7 Irreducibility (`fieldrun/separation/Separation.thy`)

For a composition matrix `c : sources × outcomes → ℝ`, `S` **decides** `t` if
`∀v≠t. Σ_{j∈S} c(j,v) < Σ_{j∈S} c(j,t)`; **`μ₀`** holds if no singleton decides `t`; `S` is
**irreducible** for `t` if it decides `t` and no proper non-empty subset does.

- **`mu0_not_irreducible`** *(proved).* `μ₀ ⇏ irreducible`: an explicit `c₃` where no single source
  decides `0` yet a **pair** does. So "no singleton decides" does **not** mean "all sources needed".
- **`triple_irreducible`** *(proved).* An explicit fully-irreducible triple — every source necessary.

This is the kernel statement of **`μ_t = 0 ≠ irreducible`**: composedness (low multiplicity) is
*necessary but not sufficient* for genuine irreducibility. Global hardness (is a given composed token
*ever* reducible by a better frame?) stays **open**.

---

## 6. The learning problem (`pil`) as PIC optimisation

`pil` fixes the decode semantics (log-semiring `L`, softmax) and **optimises the frame** `U` — i.e.
deforms the Gram kernel `G` — so that more propositions become retrievable. Formally, minimise over `U`
(and optionally biases / source contributions):

> **`𝓛(U) = NLL(L; t)  +  λ_m · ReLU(γ★ − m_worst(L, t))  +  λ_f · FP(U)`**
> (`learner.py:forward`; defaults `λ_m = 0.5`, `γ★ = 2.0`, `λ_f = 0.05`)

- **Decode-side terms** (data-dependent): `NLL = −log softmax(L/τ)[t]` anchors the aggregate to the
  targets; the **hinge margin** `ReLU(γ★ − m_worst)` pushes the worst-competitor margin past `γ★`.
- **Frame-side term** (label-free, structure-only): the **frame potential** `FP(U)` decorrelates
  propositions toward the **Welch floor**, lowering destructive co-firing interference.

The **Generate→Gate→Refine** loop wraps this:
1. **Generate** — `propose_parallel_sources` (`random` or `frame`-aligned candidate sources / rules), or
   load real DLA blocks via the `fieldrun --pil-dump` seam.
2. **Gate** — score candidates by the PIC diagnostics: ambiguity `η²`, activation variance,
   rule-firing decorrelation (a soft Welch), against a random baseline (`scoring.py:propose_and_select`).
3. **Refine** — a gradient step on `𝓛(U)`.

**Invariants the loop should preserve** (the formalism's contract, candidates for `pil`-targeted
soundness theorems):
- *(S1) decode soundness* — after refinement the frame still satisfies the recovered-probability
  property in the log-semiring on the held set (PIC stays a model of a valid decoder).
- *(S2) margin monotonicity* — refinement does not *decrease* the certified margin mass (`#{t : margin
  > 2δ}` of §5.5).
- *(S3) frame admissibility* — `FP(U) ≥ W` is structural (Welch), so the loop targets `FP/W → 1` and
  must not be reported as having beaten the floor.

### 6.1 What `pil` established empirically (so far)
- No theory-guided knob (frame-reg, allocation/targeting, propose-score-select, coherence-reg) beats
  plain SGD on the synthetic benchmarks; **PIL training *is* gradient descent** on `𝓛(U)`, cheap only
  as a shallow head on frozen DLAs.
- The win that *does* exist is **retraining the frame/subspace** (the movable side of the forge-tax
  boundary), not training an encoder into a frozen dictionary — consistent with the i-orca result that
  *frozen* compression hits a `Θ(d)` floor that a rank-bottlenecked **retrained** update clears.

---

## 7. Empirical parameters & open questions

PIC has one important parameter that the proofs **do not** pin down, plus two open links:

- **`τ★` (effective decode rank)** *(empirical, NOT formalized).* The packing exponent that *binds in
  practice* is an effective rank `τ★`, not the ambient `d`. The originally-conjectured law
  `τ★ ≈ min(exp H, d)` (effective output support sets the rank) is **only partly borne out**: measured
  on the Pythia ladder + Qwen with `fieldrun --pil-dump`, the **effective decode-block count tracks
  capacity (`nb`, hidden, layers) and *anti*-correlates with `exp H` across scale** — bigger, more
  confident models use *more* effective blocks, not fewer. A genuine but **modest within-model** effect
  survives (higher-entropy tokens use somewhat more blocks; Spearman ≈ 0.4 in capable models). So `τ★`
  is a real but **capacity-bound, not entropy-bound** quantity; its functional form is open.
  *(Evidence: `pil/experiments/tau_star_entropy.py`, `pil/docs/decode_circuit.md`.)*
- **coherence ⇒ margin** *(open).* §5.3's generator-side interference bound is proved; its consequence
  for the decode margin is empirical/mild, not kernel.
- **global irreducibility** *(open).* §5.7 proves local irreducibility of a *given* coalition; whether a
  composed token is irreducible under *every admissible frame* — the real "is this computation
  necessary?" question — is unproved.

---

## 8. Notation

| symbol | meaning |
|--------|---------|
| `V`, `nₚ` | propositions (tokens); their count |
| `H = ℝ^d`, `d` | residual / unembedding space; its dimension |
| `U_v`, `U` | frame vector of proposition `v`; the frame map |
| `G(v,w)`, `ρ(v,w)` | Gram kernel `⟨U_v,U_w⟩`; cosine Gram |
| `S`, `J`, `d_j` | source set; its size; source `j`'s residual vector |
| `c_j(v)` | incidence of source `j` on `v` = `⟨d_j, U_v⟩` |
| `L(v)`, `b_v` | aggregate score (logit) of `v`; bias |
| `T`, `⊕_T`, `⊗` | semiring temperature; `T`-sum; product (`= +`) |
| `m(t)` | margin `L(t) − max_{v≠t} L(v)` |
| `PR` | participation ratio `(Σ|w|)²/Σw²` (effective # sources) |
| `μ_t` | multiplicity: # sources whose own argmax is `t` |
| `η²` | ambiguity (between/total variance on low-margin slice) |
| `FP`, `W` | frame potential; Welch floor `(nₚ−d)/(d(nₚ−1))` |
| `γ`, `ρ`, `M`, `n` | decode margin; `max‖U_v‖`; #rules; #routing decisions |
| `δ`, `2δ` | per-token perturbation bound; tight margin-certificate threshold |
| `τ★` | effective decode rank (empirical, capacity-bound) |

---

*This spec is versioned with the proofs and the code. When a theorem is added to i-orca or a quantity
changes in `pil`/`fieldrun`, update the corresponding section here and bump the version. The companion
paper (`paper/pic_calculus.tex`) expands §1–§5 into publishable form.*
