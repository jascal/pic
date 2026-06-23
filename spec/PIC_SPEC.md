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
`v` is the inner product `c_j(v) = ⟨d_j, U_v⟩`. Sources of one token combine by `⊗` (= real `+`) into a
**monomial** `L(v) = Σ_j c_j(v) + b_v`; the **tokens** then combine by a temperature-`T` semiring sum
`⊕_T` into the decode **polynomial** `Z_T = ⊕_{T,v} L(v)`. At `T = 1` `⊕_T` is logsumexp and PIC **is**
the transformer forward pass (softmax over the summed logits); at `T → 0` it is `max` and PIC is the
geometry of margins, Laguerre power diagrams, and the head/tail decode certificate — and (§3.1) the
decoded token is the **same** at every `T`. The calculus cleanly **separates the frame** (the intrinsic geometry of `{U_v}`:
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

### 1.3 Monomials, polynomial, decision

The semiring has **two** operations and they play **different roles** — getting this right is the whole
point of the algebra:

- **`⊗` (= real `+`) combines sources *within* a token.** A token `v`'s score is the `⊗`-product of its
  incidences — a **monomial**:
  > `L(v) = ( ⊗_{j∈S} c_j(v) ) ⊗ b_v = Σ_{j∈S} ⟨d_j, U_v⟩ + b_v`
  (`geometry.py:logits_from_incidences` sums over the source axis). This is `T`-independent: sources just
  add.
- **`⊕_T` combines *across* tokens (the alternatives).** The whole decode is the `⊕_T`-sum of the token
  monomials — a **`(⊕_T,⊗)` polynomial** whose monomials are the tokens and whose variables are the
  sources/residual:
  > `Z_T = ⊕_{T,\ v∈V} L(v) = ⊕_{T,\ v∈V} ⊗_{j∈S} c_j(v) ⊗ b_v`.

The **decision** reads off this polynomial. At `T → 0` (tropical) `Z_0 = max_v L(v)` is the decode value
and `argmax_v L(v)` the decoded token; at `T = 1` (log) `Z_1 = logsumexp_v L(v)` is the log-partition and
`p(v) = exp(L(v) − Z_1)` the softmax. **The monomials `L(v)` are shared; only the cross-token aggregation
`⊕_T` changes with `T`.** (This is exactly the tropical-polynomial picture of a ReLU network, with tokens
as monomials — see §2 and the temperature-invariance lemma §3.1.)

---

## 2. The semiring family (the calculus parameter)

PIC is parameterised by a **temperature** `T ≥ 0` selecting a commutative semiring
`R_T = (R, ⊕_T, ⊗, 𝟘, 𝟙)`. The full tuple (the complete semiring, not just the operations):

| component | value | role |
|-----------|-------|------|
| carrier `R` | `ℝ ∪ {−∞}` (log-domain scores) | incidence / score values |
| sum `⊕_T` | `a ⊕_T b = T·log(e^{a/T} + e^{b/T})`, with `⊕_0 := max` | combine **alternatives** (tokens) |
| product `⊗` | `a ⊗ b = a + b` (real addition), **`T`-independent** | combine **sources** within a token |
| sum identity `𝟘` | `−∞` | "no alternative" (`a ⊕_T −∞ = a`) |
| product identity `𝟙` | `0` | "empty coalition" (`a ⊗ 0 = a`) |

**Semiring axioms** (all hold for every `T ≥ 0`; the tropical case `T = 0` is the one machine-checked in
`TropicalSemiring.thy`):
1. `(R, ⊕_T, 𝟘)` is a commutative monoid — `⊕_T` associative, commutative, identity `−∞`. *(proved, T=0)*
2. `(R, ⊗, 𝟙)` is a commutative monoid — `⊗ = +` associative, commutative, identity `0`. *(proved, T=0)*
3. **Distributivity** `a ⊗ (b ⊕_T c) = (a ⊗ b) ⊕_T (a ⊗ c)` — addition distributes over the `T`-sum.
   *(proved, T=0: `tmul_tadd_distrib_left`; for general `T` it is the log-domain identity
   `a + T\log(e^{b/T}+e^{c/T}) = T\log(e^{(a+b)/T}+e^{(a+c)/T})`.)* This is the load-bearing law: it
   rewrites a sum-of-maxes (a ReLU stack) as a max-of-sums (a tropical polynomial).
4. **Annihilation** `𝟘 ⊗ a = 𝟘` — `−∞ + a = −∞` (an absent alternative stays absent under combination).

`R_T` is **commutative** and (at `T=0`) **idempotent** (`a ⊕_0 a = a`); idempotency is lost for `T>0`
(`a ⊕_1 a = a + log 2`), which is exactly the difference between geometry (`T→0`) and probability (`T=1`).
The three named instances:

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

**(N2) The coalition bracket** — the `⊗`-monomial of a coalition `P ⊆ S` on a token `v` (sources add;
**`T`-independent**):

> `⟦P⟧(v)  :=  ⊗_{j∈P} (j ▷ v)  =  Σ_{j∈P} ⟨d_j, U_v⟩`,     full token score   `L(v) := ⟦S⟧(v) ⊗ b_v`.

The **`T` lives across tokens, not inside the bracket:** the decode polynomial is
`Z_T := ⊕_{T, v∈V} L(v)`, and the decoded token is `⊤(S) := argmax_v ⟦S⟧(v) ⊗ b_v` — which, by §3.1, is
the **same for every `T`**. (So there is no `⟦·⟧_T`; the bracket is one object, read out by the
`T`-family.)

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
- *Aggregate:* `L(v) = ⟦S⟧(v) ⊗ b_v`; decode `⊤(S) = argmax_v L(v)` (`T`-invariant, §3.1); the
  `T`-family only sets how the cross-token partition `Z_T = ⊕_{T,v} L(v)` normalises confidence.
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

### 3.1 Decode temperature-invariance *(native; what ties the two anchors together)*

The two soundness anchors are only consistent because of a fact that is **native to the `T`-family** and
sits in neither tropical geometry nor provenance semirings on its own:

> **Lemma (decode invariance).** For every `T > 0`, `⊤(S) = argmax_v L(v)` is the **same** token, equal
> to the tropical (`T → 0`) decoder. Only the *normalised confidence* — the softmax `p_T(v) = exp((L(v) −
> Z_T)/T)` and hence the entropy/partition `Z_T = ⊕_{T,v} L(v)` — varies with `T`.

*Why.* The monomials `L(v) = ⟦S⟧(v) ⊗ b_v` carry **no `T`** (sources combine by `⊗ = +`); `T` enters
only the cross-token aggregator `⊕_T`, which is **strictly order-preserving in each argument** and
satisfies `argmax = (⊕_0)`-selector. So the `argmax` of a fixed vector `(L(v))_v` is `T`-independent;
`⊕_T` reweights *how much* the winner beats the field, never *who* wins.

This is the rigorous form of the soundness bridge: the **proofs** run at `T → 0` (tropical margins,
power-diagram capacity, head/tail) and the **model** runs at `T = 1` (softmax), yet they decode the
**same token** because they share the monomials and differ only in `⊕_T`. The forge-tax / confidence
story (entropy, margins) is the `T`-dependent part; the *decision* is the `T`-invariant part. Corollary:
every margin/decode theorem of §5 proved tropically transfers verbatim to the operating model's argmax.

---

## 4. The three roles: encode, frame, decode

PIC *is* a sandwich — there are **three** roles, not two, and the frame is the filling, not a bread:

```
        ENCODE                         FRAME                        DECODE
  input ─▶ residual r = Σ_j d_j ─▶ ⟨·, U_v⟩ geometry of {U_v} ─▶ logits L(v), argmax/softmax
  (sources WRITE the residual)   (shared dictionary both meet at)  (READ the residual out)
```

- **Encode (the generator side).** How the residual gets *written*: the sources `d_j` and the rules
  that produce them. PIC does **not** model the encoder's internal computation (the attention/MLP that
  builds each `d_j`) — that is taken as **given** (the observed per-block writes; the irreducible
  forge-tax compute). What PIC *does* bound is the encoder's **output**: how many independent sources
  can write (**routing rank** §5.2, `≤ min(M,d)`) and what interference overpacking forces
  (**routing Welch** §5.3). This is the `⊗`/variable side of the polynomial and **the binding side**.
- **Frame.** The shared dictionary geometry `{U_v}` — what both encode and decode meet at. Its intrinsic
  quantities (Gram coherence, frame potential, Welch floor) and its **decode capacity** (§5.1).
- **Decode (the read-out).** The aggregate `L(v) = ⟨r, U_v⟩ + b_v`, evaluated in the chosen semiring:
  margins, participation ratio, multiplicity `μ_t`.

**So why does PIC look like only "frame vs decode"?** Because that two-way cut (below) is an
*operational* split for **analysis and learning**, not the whole architecture:
- the **encode is observed, not modelled** — in frozen analysis the `d_j` are read straight off the
  real forward pass (`fieldrun --pil-dump`), so there is nothing to *separate*: encode is an input. PIC
  only *counts* the encoder (rank/Welch), it does not *re-derive* it. This is principled, not a gap: it
  is the empirical **mass ≠ causation** result (§5 guardrail) — the read-out is sparse and faithful (so
  PIC models it cleanly), while *building* the residual is the whole causal stack (so PIC leaves it as
  the un-modelled forge-tax).
- the cut that **`pil` exploits** is then: every remaining PIC quantity is either **frame-side** (a
  function of `{U_v}` alone — intrinsic, label-free) or **decode-side** (a function of `L` and a target —
  evaluated in the semiring). `pil` moves a frame-side objective without touching decode semantics, and
  vice versa. A learner that *also* trained the encoder (the `d_j` generators) would be working the
  **encode/generator side** — exactly the routing-rank/Welch regime, and the place a future PIC could add
  an explicit encode model.

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

### 4.3 The encoder (formal model of the encode side)

The encode side has an explicit formal model — kernel-checked in i-orca `PIC_Core.thy` (`locale
pic_encoder`). The residual is **factored, not given**: on input `x`,

> `enc(x) = Σ_{k∈K} g_k(x) · a_k`,     `L(v; x) = ⟨enc(x), U_v⟩ + b_v = Σ_k g_k(x)·⟨a_k, U_v⟩ + b_v`

with **`K`** a finite set of **rules** (neurons / attention slots), **`a_k ∈ H`** their **fixed write
directions** (the output-projection columns), and **`g_k(x) ∈ ℝ`** the **input-dependent gates**
(activations). This expresses *any* transformer block: a block's residual write is `W_out · (input-
dependent vector)` — fixed output columns `a_k` gated by an input coefficient `g_k(x)` — for **MLP and
attention alike**. The gate `g_k(x)` is left **uninterpreted**: PIC models the *linear* write/read
algebra (`⟨a_k, U_v⟩` is rule `k`'s incidence, `▷`), and leaves the nonlinear gate compute as the
un-modelled forge tax. So the encoder **sharpens** the boundary rather than enlarging the calculus.

Three things are proved about it (the generator side made first-class):
- **encode→decode composition** (`Lx_gated_incidence`): the whole forward pass is one PIC term — a gated
  sum of fixed rule-incidences.
- **routing rank** (`routing_rank`, `routing_rank_dim`): for *every* input, `enc(x) ∈ span{a_k}`, of
  dimension `≤ min(|K|, d)` — superposition is forced when rules exceed the ambient dimension (§5.2/§5.3
  are now properties of this object).
- **slice = `pic`** (`encoder_slice_logit`, via `interpretation`): every fixed-input slice of an encoder
  **is** a `pic` source-model (sources `d_k = g_k(x)·a_k`), and its logit equals `L(v;x)`. So `pic` is the
  fixed-input restriction; the encoder adds the input axis and the write/gate factorisation.

These are the quantities `pil`'s frame-side objective drives; a learner that *also* trained the gates /
write directions would be working this encode side (the routing-rank/Welch regime).

---

## 5. The theorems (what is proved)

All statements below are **kernel-proved in i-orca** at the cited file. They split into the two sides
of a single **two-sided packing** picture: the **frame side** bounds how many tokens a geometry can
decode; the **generator side** — i.e. the **encode side** of §4 — bounds how many rules can write the
logits and what interference that forces. (The "encode" and "generator" sides are the same thing: the
production of the residual. PIC counts it here; it does not model its mechanism.)

### 5.0 The duality that organises them *(native schema)*

The split is not two unrelated bound-collections — it is the **two arities of the decode polynomial**
`Z_T = ⊕_{T,v∈V} ⊗_{j∈S} c_j(v)` (§1.3). A `(⊕,⊗)` polynomial has two sizes, and PIC bounds each:

| | algebraic role | what is bounded | by | theorem |
|---|---|---|---|---|
| **monomials** (tokens `v`) | the `⊕`-arity | how many can be cleanly **decoded** (γ-separated) | frame geometry: `\|S\| ≤ (1+2ρ/γ)^d` | §5.1 |
| **variables** (sources `j`) | the `⊗`-arity | how many are **independent** writers | generator rank: `dim ≤ min(M,d)` | §5.2 |
| **packing excess** | `#variables > rank` | forced **interference** | Welch: `Σ_{i≠j}⟨f_i,f_j⟩² ≥ n(n−M)/M` | §5.3 |

So **"two-sided packing" = a simultaneous bound on a polynomial's monomial-count and its variable-rank**,
with Welch as the obstruction that appears exactly when the `⊗`-side (sources) is overpacked relative to
its rank. The frame side (`§5.1`, slack ~`10^59` for real `d`) is rarely binding; the **generator side
(`§5.2`–`§5.3`) is the side that binds** — superposition is forced on the writers, not on the readouts.
The head/tail certificate (§5.4) is then the statement that a *sub-polynomial* (a head of monomials)
reproduces the decode when it dominates; the margin certificate (§5.5) bounds how much the *coefficients*
may be perturbed without changing the argmax. Every §5 theorem is one of: bound a monomial-count, bound a
variable-rank, or certify a sub-polynomial / a coefficient perturbation.

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
- **`encoder_superposition`** *(proved, `PIC_Core.thy`).* On the explicit encoder rule bank (§4.3), more
  rules than the ambient dimension forces the write directions to be linearly **dependent**
  (`DIM < card(a_K) ⟹ ¬ independent(a_K)`) — the *qualitative* generator-side packing, now a property
  of the encoder and the companion to `routing_rank`.
- *(empirical, open)* The step **"routing interference ⇒ margin degradation"** is **not** a kernel
  theorem — the **one** thing left open on this side. The *count/rank/dependence* facts are all proved
  (`routing_rank`, `encoder_superposition`); the quantitative interference floor `≥ n(n−d)/d` is
  `Welch.thy`; only the implication from coherence to a *margin penalty* is unproved. Trained models pack
  features at ≈ the Welch floor with healthy margins; the coherence→margin link is measured and mild
  (`pil` docs §5f/§5g), not proved.

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

### 5.5 Decision-side robustness — margin certificate (`PIC_Core` `PIC_Logic.thy`; orig. `provable_opt/ProvableOpt_Common.thy`)

`margin(L,V,t) = L(t) − max_{v≠t} L(v)`. *(Now also kernel-checked self-contained in `PIC_Logic.thy` as
the LP clause-weight-drift bound — the decision-side companion to demand-closure.)*

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

## 6.5 PIC as a logic program (the weighted-Datalog reading)

PIC is, structurally, a **semiring-weighted logic program**, and this is not a stretched analogy — the
pieces already exist across the program and only need the algebra of §2 to name them. The turnstile
`P ⊢_γ v` (§2.5) is literally a **weighted inference rule**: a coalition (the body) derives a proposition
(the head) with margin `γ`. Reading the calculus as a logic program:

| logic programming | PIC | already in the program |
|---|---|---|
| rules / clauses (the program) | rule incidences `⟨a_k, U_v⟩` — `head ⟸ body` over the **fixed** rule bank | `pic_encoder` (§4.3) |
| ground facts on input `x` (the EDB) | the **gated rule-fires** `g_k(x)·a_k` of the encoder | `pic_encoder.enc` (§4.3); `fieldrun --pil-dump` `contrib` |
| atoms / fact weights | incidences `j ▷ v` valued in `R_T` | the `contrib` matrix |
| coalition / derivation | `P ⊢ v` (a `⊗`-monomial over body sources) | the turnstile §2.5 |
| answer / model | the lfp of the immediate-consequence operator `T_P` | i-orca `ProvableOpt_Common` |
| query | the decode `⊤(S) = ⊕_T`-argmax over propositions | `fieldrun` decode |
| magic-sets / demand transform | **demand-closure** `lfp(restrict T D) = lfp T ∩ D` *(proved)* | §5.6, `ProvableOpt.thy` |
| program emission | a recursive Soufflé/Datalog program | `fieldrun --datalog` (`LOGIC_EXPORT`) |

**The encoder (§4.3) is the fact-generating (EDB) layer** — this is why the LP formalism needs it. The
fixed rule bank `{a_k}` (with incidences `⟨a_k, U_v⟩`) is the **program** (the clauses, input-independent);
the input `x` fires the gates `g_k(x)`, producing the **weighted ground facts** `g_k(x)·a_k` that the
decode (query) then resolves. So the split is clean: *clauses are the frame-side geometry of the rule bank*
(static), *the EDB is the encoder's per-input gated fires* (dynamic), *the query is the decode*. Without an
explicit encoder there is no place for the input-to-EDB step — exactly the gap an executable `pic`-LP would
have to fill (`PIC_Core.encoder_slice_logit` is the formal statement that each input grounds the program to
a `pic` model).

The **semiring chooses the logic**: `T = 0` (tropical) is shortest-path / Viterbi / min-cost logic
programming (the geometry/optimization reading); `T = 1` (log) is **probabilistic** logic programming
(ProbLog-style, the model's softmax reading); Boolean is classical Datalog (pure rule gating). This is
exactly the Green–Karvounarakis–Tannen provenance-semiring foundation, with PIC's one native move (§2.5
N1): the fact weights are not given but **derived from geometry**, `j ▷ v = ⟨d_j, U_v⟩`.

So PIC *does* lend itself to a logic-programming language, and the **type discipline is the frame/decode
separation** (§4): a PIC program is *frame-side* clauses (the geometry of `{U_v}` — what overlaps what)
plus *decode-side* queries (margins, multiplicity, evaluated in `R_T`). The proved demand-closure theorem
(§5.6) is precisely the soundness of the standard **magic-sets / demand** optimization for such a
language, and the margin certificate (§5.5) bounds how much a clause weight may drift before an answer
flips. A concrete `pic`-LP would be a semiring-parameterized weighted Datalog whose immediate-consequence
operator is `T_P(I) = { v : ∃ P ⊆ I, P ⊢ v }`, evaluated in `R_T`, with the frame as the (learnable)
fact-weight oracle — the analysis (`fieldrun --datalog`), the semantics (i-orca `lfp`/demand-closure),
and the learnable weights (`pil`) are the three components, already built; `pic` is their shared grammar.
*(This is the `LOGIC_EXPORT` / `PROVABLE_OPT` thread, given its algebraic spine.)*

> **Full treatment: [`PIC_LP.md`](./PIC_LP.md)** — the language spec (syntax grounded in the actual
> `fieldrun export --logic` emission, the `T_P`/lfp semantics, the semiring modes, the two recursion
> sources, demand-closure = magic-sets, and a DCG-style grammar surface for the recursive fragment),
> with the kernel theory in i-orca `examples/pic_core/PIC_Logic.thy` (`Tp`/`answer`/`reach_all`/
> `demand_restrict_lfp`/`decode_magic_sets_lossless`, all kernel-checked, 0 sorry).

---

## 7. Empirical parameters & open questions

PIC has one important parameter that the proofs **do not** pin down, plus two open links:

- **`τ★` (effective decode rank)** *(empirical, NOT formalized).* The packing exponent that *binds in
  practice* is an effective rank `τ★`, not the ambient `d`. The originally-conjectured law
  `τ★ ≈ min(exp H, d)` (effective output support sets the rank) is **refuted in the cross-model
  direction**: measured on a 7-model sweep (Pythia 70m→1.4b + Qwen 0.5B/1.5B) via `fieldrun --pil-dump`,
  the architecture-fair effective decode-block count `r_eff` **anti-correlates with `exp H`** (Pearson
  −0.71) and **tracks block count `nb = 2L+1`** (Pearson +0.75, the best predictor), depth and hidden
  width — bigger, more confident models use *more* effective blocks, not fewer. The clean dissociation:
  **pythia-1b (16 layers, *more* params) uses fewer effective blocks (14.1) than pythia-410m (24 layers,
  *fewer* params, 23.6)** — so it is **depth/`nb`, not parameter count and not entropy** that sets
  `r_eff ≈ (0.4–0.5)·nb` within an architecture. A genuine but **modest within-model** effect survives
  (per token, higher-entropy positions use somewhat more blocks; Spearman ≈ 0.3–0.4 in capable models,
  ≈ 0 in the two tiniest). So `τ★` is real but **capacity/depth-bound, not entropy-bound**; its functional
  form is open. The earlier "≈12 scale-invariant blocks" reading was a Qwen *raw*-PR artifact — the
  common-mode-centred count grows with `nb`.
  *(Evidence: `pil/experiments/tau_star_entropy.py`, `pil/results/tau_star_entropy.txt`.)*
- **coherence ⇒ margin** *(matched-filter form proved; optimal form open).* Now pinned precisely
  (`PIC_Interference.thy`): under the **matched-filter** decoder (steer by the target feature) the margin
  is `1 − (max cross-coherence)`, so coherence subtracts *directly* (`mfmargin_le`, `mfmargin_lt_one`).
  But the **unconditional** claim is *false* — the achievable (best-steering) margin is `≥ 1 − ρ`, so the
  decoder can recover (the measured "cope at the Welch floor"). The genuinely open part is a geometric
  lower bound on the **optimal** margin in the Welch regime `n > M`. Everything else up to it is proved:
  rank (`routing_rank`), superposition (`encoder_superposition`), the interference floor `≥ n(n−d)/d`
  (`Welch.thy`).
- **global irreducibility** *(RESOLVED — the pure-behavioral notion is **vacuous**).* The question "is a
  composed token irreducible under *every* admissible frame?" collapses: with free gates `g_j(x)` and a
  free frame, the residual ranges over `span{a_j}`, so any realizable behavior can be re-geometrized to
  make *any* nominated decision single-source (place one generator inside its decode cell — an open,
  full-dimensional cone — the rest of the bank still reaches every other cell with free gates). So there
  is **no frame-invariant behavioral certificate** that a decision must be composed; the forge tax is
  **not** a behavioral invariant. (The `M=1` case shows the rank bound is real, but it does not localize
  to a specific token; Welch counting forces *some* decisions composed when `|X|>M`, never a *named* one.)
  *The well-posed version fixes the gates to the model's measured activations* — see below.
- **the forge-tax certificate is activation-relative, not behavior-relative** *(the well-posed
  reformulation — open, now tractable).* Fix `g_j(x)` to the model's *actual* per-input activations (which
  `fieldrun --pil-dump` already emits as `d_j(x) = g_j(x)·a_j`). Then "is `t_{x₀}` single-source / low-
  coalition realizable?" becomes a nontrivial linear-realizability question over the *fixed* activation
  matrix, with genuine sign-rank / margin-complexity certificates and (finite instances) decidability via
  real-closed-field / SMT. This is the regime fieldrun *already operates in* — so the negative result
  **vindicates** the native-frame local-irreducibility approach (§5.7) as the correct notion and points
  the forge-tax certificate at the fixed-activation problem. *(Resolution via an adversarial-collaborator
  constructive argument, 2026-06.)*
- **`τ★` functional form** *(open — empirical, awaiting data).* The cross-model law is refuted
  (capacity/depth-bound, above); the precise functional form `r_eff = f(nb, depth, H)` awaits the
  large-N (14m→72B) sweep on bigger hardware. Not a kernel question.
- ~~**`lfp(layer program) = model decode`**~~ *(now **proved**, `PIC_Forward.thy`).* The forward pass is
  the least fixpoint of a layered Datalog program (atom `(ℓ,x)` = "residual at layer ℓ is `x`"; the
  embedding is the fact `(0,r₀)`; each layer is a clause `(ℓ+1, step ℓ x) ⟸ (ℓ, x)`). `lfp_is_trajectory`
  proves `answer(Player) = {(ℓ, Rℓ) | ℓ}` **exactly** (⊇ by induction, ⊆ via `answer_least` on the
  trajectory-as-model), so the lfp *is* the forward pass; `lfp_determines` pins the final-layer residual
  uniquely; and `lfp_decode_logits` (in `pic_frame`) shows the model logits are the frame read-out of the
  lfp's final fact — so the lfp determines the decode. `step` is abstract; the encoder is the instance.

**What this round closed** (so the open set is honest): the generator-side **superposition** bound is now
proved on the explicit encoder (`encoder_superposition`), the **margin certificate** is self-contained in
`PIC_Logic`, the **LP metatheory** (operator, least model, magic-sets, unbounded recursion) is
kernel-checked (`PIC_Logic.thy`), and the **forward pass = lfp** end-to-end theorem is proved
(`PIC_Forward.thy`). `pic_core` now covers the full §5 theorem set except the quantitative Welch floor
(left to `Welch.thy`). The only remaining opens are the three above (coherence⇒margin's last implication,
global irreducibility, τ★'s functional form) — two empirical/data-bound, one possibly genuinely hard.

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

---

## Appendix A. A worked micro-example (the calculus on real numbers)

The whole calculus on a tiny instance, exercising the native notation and the
composed-vs-irreducible distinction of §5.7. Take sources `S = {a, b, c}` and read incidences
`j ▷ v = c_j(v)` off a fixed table (these are the proved instances of `Separation.thy`, transcribed). A
coalition's score is the bracket `⟦P⟧(v) = ⊗_{j∈P}(j ▷ v) = Σ_{j∈P} c_j(v)`; it **decides** `v` (`P ⊢ v`)
when `v` is its strict argmax.

**A.1 An irreducible coalition** (`triple_irreducible`). Target `t`, competitors `x, y, z`:

| `j ▷ ·` | `t` | `x` | `y` | `z` |
|---|---|---|---|---|
| `a` | 3 | 8 | 0 | 0 |
| `b` | 3 | 0 | 8 | 0 |
| `c` | 3 | 0 | 0 | 8 |

- Full coalition: `⟦{a,b,c}⟧(t) = 9`, `⟦{a,b,c}⟧(x)=⟦{a,b,c}⟧(y)=⟦{a,b,c}⟧(z)=8`, so `{a,b,c} ⊢_1 t`
  (decides `t` with margin 1).
- Every singleton fails: `{a} ▷`: `t=3` but `x=8`, so `¬({a} ⊢ t)` (and likewise `b, c`) — **`μ_t = 0`**.
- Every proper pair fails: `⟦{a,b}⟧(t)=6` but `⟦{a,b}⟧(x)=8`; symmetrically for `{a,c}, {b,c}`.
- No proper non-empty subset decides `t`, yet `{a,b,c}` does ⟹ **`{a,b,c}` is irreducible for `t`**.
  Each source is *necessary*: the decision is the whole coalition or nothing.

**A.2 Composed but reducible** (`mu0_not_irreducible`) — the distinction that matters. Same `t`,
competitors `x, y`:

| `j ▷ ·` | `t` | `x` | `y` |
|---|---|---|---|
| `a` | 3 | 4 | 0 |
| `b` | 3 | 0 | 4 |
| `c` | 3 | 4 | 4 |

- Full: `⟦{a,b,c}⟧(t)=9 > ⟦·⟧(x)=⟦·⟧(y)=8`, so `{a,b,c} ⊢_1 t`.
- Every singleton fails (`{a}`: `t=3, x=4`; `{c}`: `t=3, x=4=y`) — again **`μ_t = 0`**, so `t` is
  **composed**.
- **But the pair `{a,b}` decides:** `⟦{a,b}⟧(t)=6 > ⟦{a,b}⟧(x)=4 = ⟦{a,b}⟧(y)`, i.e. `{a,b} ⊢_2 t`.
- So `t` is composed (`μ_t = 0`) yet **not** irreducible — a proper subset already suffices.

**The lesson.** A.1 and A.2 have the **same** full-coalition decision and the **same** `μ_t = 0`, but
differ in irreducibility. So `μ_t = 0` (low multiplicity / "no single source decides") is *necessary but
not sufficient* for irreducibility — the kernel content of `Separation.thy`'s `mu0_not_irreducible`.
Reading `μ_t = 0` as "this computation is irreducible" is exactly the overclaim the proof rules out; the
turnstile `⊢` over **sub-coalitions**, not the multiplicity count, is what certifies irreducibility.
*(Arithmetic checked; both tables are the proved instances of `Separation.thy`.)*
