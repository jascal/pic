# PIC-LP: the logic-programming face of Projective Incidence Calculus

**Status:** language spec (v0.1), companion to [`PIC_SPEC.md`](./PIC_SPEC.md) §6.5. PIC-LP is **not a new
language to be invented** — it is the formal semantics of a semiring-weighted Datalog that **already
exists and runs**: `fieldrun export --logic` emits it, Soufflé executes it, and i-orca's `ProvableOpt`
corpus proves its optimization-soundness. This doc gives that program a precise reading and pins down
exactly which parts are kernel-proved.

The one-line claim: **a transformer's next-token decision is a semiring-Datalog program; the encoder is
its EDB, the frame is its clause weights, the decode is its query, and the temperature `T` selects the
logic** (tropical = min-cost, log = ProbLog, Boolean = classical Datalog).

---

## 1. The program already emitted (ground truth)

`fieldrun export --logic` (`fieldrun/src/logic.rs`) renders one next-token decision as runnable Soufflé.
The core relations and clauses, verbatim in spirit:

```prolog
.decl candidate(t:number)                 // the candidate tokens (the queried propositions)
.decl contrib(block:symbol, t:number, w:float)   // the facts: per-block incidence  w = ⟨d_block, U_t⟩
.decl logit(t:number, s:float)
.decl decide(t:number)
.decl retrieved(t:number)
.decl induction_copy(t:number)            // the recursive rule's premise
.decl ngram_succ(key:symbol, t:number)

logit(T, S)   :- candidate(T), S = sum W : { contrib(_, T, W) }.   // ⊗ over blocks  (log-semiring +)
decide(T)     :- logit(T, S), S = max S2 : { logit(_, S2) }.       // ⊕ = max        (T → 0, argmax)
retrieved(T)  :- induction_copy(T).        // the clean RECURSIVE rule: copy the token after the matched prefix
```

Everything below is the formal reading of *this*. The mapping to PIC (`PIC_SPEC.md`):

| PIC-LP | PIC | semiring role |
|--------|-----|---------------|
| fact `contrib(b, t, w)` | incidence `b ▷ t = ⟨d_b, U_t⟩` valued in `R_T` | the atoms' weights |
| `logit(T,S) :- …S = sum W…` | the `⊗`-monomial `L(t) = ⊗_b (b ▷ t)` | **`⊗` aggregates a body** |
| `decide(T) :- …S = max S2…` | the decode `⊤ = ⊕_T`-argmax over tokens | **`⊕_T` aggregates alternatives** |
| `retrieved`, `induction_copy` | the recursive macro (induction head) | the genuinely recursive clause |

So the database-style `sum`/`max` aggregates in the emitted Soufflé are **literally** PIC's `⊗` and `⊕_T`
(§1.3 of `PIC_SPEC.md`): a body is a `⊗`-monomial, the query head is the `⊕_T` over alternatives.

## 2. Syntax

- **Atoms / relations.** A finite signature of relations; an *atom* is a ground relation instance.
  `contrib(b,t,w)`, `logit(t,s)`, `decide(t)`, `retrieved(t)`, `induction_copy(t)`, `ngram_succ(k,t)`.
- **Facts (the EDB).** Ground atoms given for an input — here the `contrib(b,t,w)` rows, **produced by the
  encoder** (§7): on input `x`, rule `b` fires with gate `g_b(x)` and contributes
  `w = g_b(x)·⟨a_b, U_t⟩`. The EDB is input-dependent; the rules are not.
- **Clauses (the IDB rules).** `head ⟸ body₁, …, bodyₙ` with semiring aggregates in the body
  (`sum`/`max`). The frame supplies the clause weights `⟨a_b, U_t⟩` (input-independent).
- **Program** `P` = facts ∪ clauses. **Query** = a relation to solve for — here `decide`.

## 3. Semantics

Let `B` be the Herbrand base (all ground atoms). A program `P` induces an **immediate-consequence
operator** `T_P : 𝒫(B) → 𝒫(B)`,

> `T_P(I) = I ∪ { head(c) : c ∈ P, body(c) ⊆ I }`,

which is **monotone**, so by Knaster–Tarski it has a **least fixpoint** `lfp(T_P)` — the program's
**least model** (its answer). The query is read off the answer: the decoded token is the unique `t` with
`decide(t) ∈ lfp(T_P)`.

**Semiring-weighted (provenance) evaluation.** Atoms carry weights in `R_T` (§2 of `PIC_SPEC.md`); a
clause combines its body by `⊗` and merges alternative derivations of the same head by `⊕_T`. This is
Green–Karvounarakis–Tannen provenance with the carrier `R_T`. PIC's one native move: the fact weights
are **not given but derived from geometry**, `b ▷ t = ⟨d_b, U_t⟩`. The two emitted aggregates are exactly
the two semiring operations:
- `logit(T,S) :- … S = sum W : {contrib(_,T,W)}` — the body is a `⊗`-product (`= +` in the log domain).
- `decide(T) :- … S = max S2 : {logit(_,S2)}` — alternatives merge by `⊕_T` (`max` at `T → 0`).

**The semiring chooses the logic.**

| `T` | `⊕_T` | PIC-LP is | query semantics |
|-----|-------|-----------|-----------------|
| `0` (tropical) | `max` | **min-cost / Viterbi** logic programming | the single best derivation (the argmax decode) |
| `1` (log) | logsumexp | **probabilistic** logic programming (ProbLog-style) | the normalized answer distribution (softmax) |
| Boolean | `∨` | **classical Datalog** | derivable-or-not (pure rule gating) |

The **decode temperature-invariance** lemma (`PIC_SPEC.md` §3.1) has a clean LP reading: the *winning
derivation* `decide(t)` is the **same** for every `T` — only its *certainty* (the provenance weight)
changes. So the tropical program and the probabilistic program compute the **same answer**, differing
only in the confidence annotation.

## 4. Where the recursion comes from

A single forward pass is **stratified** (acyclic): `contrib` → `logit` → `decide` are layered, so
`lfp(T_P)` is reached by finite bottom-up iteration — Datalog with no real recursion. The genuine
recursion has two sources, both already in the emitted program / the program's structure:
1. **The induction macro** — `retrieved(T) :- induction_copy(T)` (copy the token after the matched
   prefix). This is the one *clean recursive clause*: it derives the next token by reference to an earlier
   match, the LP face of an induction head. Generation feeds `decide` back as a new fact → an
   autoregressive recursion whose `lfp` is the generated sequence.
2. **The layer recurrence** — the residual at layer `ℓ+1` is derived from layer `ℓ` (the encoder applied
   once per layer): a stratified program whose strata are the layers. Its `lfp` is the full residual
   stack; the decode reads the final stratum.

## 5. Optimization is kernel-proved (magic-sets = demand-closure)

The standard logic-programming optimizations are **already proved sound** in i-orca's `ProvableOpt`:
- **Demand transform / magic-sets** — restrict the program to the atoms a query demands. With
  `restrict_op T D = (λI. T(I) ∩ D)` and `demand_closed T D ⟺ ∀I. T(I)∩D = T(I∩D)∩D`, the theorem
  `demand_restrict_lfp` *(proved)* gives `lfp(restrict_op T D) = lfp(T) ∩ D`: **running the
  demand-restricted program computes exactly the demanded part of the answer.** Instance
  (`ProvableOpt.thy`): dropping every non-final-layer accumulator preserves `decide` — the LP face of "the
  decode only needs the late strata" (and exactly the `mass ≠ causation` boundary, stated as a program
  transform).
- **Bounded clause-weight perturbation** — the margin certificate (`PIC_SPEC.md` §5.5,
  `decode_margin_certified` *proved*, threshold tight) bounds how far a clause weight may drift before the
  answer `decide(t)` flips.

These are the language's metatheory, machine-checked, not asserted.

## 6. The three roles, in LP terms

`PIC_SPEC.md` §4's encode → frame → decode sandwich is the LP pipeline:
- **Encoder = the fact generator (EDB).** Input `x` fires gates `g_b(x)`, grounding the program to the
  `contrib(b,t,·)` facts. `PIC_Core.encoder_slice_logit` *(proved)* is the statement that each input
  grounds the program to a `pic` model.
- **Frame = the clause weights (the program proper).** The fixed rule incidences `⟨a_b, U_t⟩` are the
  input-independent clauses — the *program*, not the data.
- **Decode = the query.** `decide` over the candidate propositions.

This is the type discipline: *frame-side* clauses (static geometry) + *encode-side* EDB (per-input
fires) + *decode-side* query.

## 7. Worked reading of the emitted program

For one decision with candidates `{t₀, t₁}` and blocks `{b₁, b₂}`, the encoder grounds
`contrib(b₁,t₀,w₁₀), contrib(b₂,t₀,w₂₀), contrib(b₁,t₁,w₁₁), contrib(b₂,t₁,w₁₁)`. Then
`logit(t,·)` fires `⊗` (`w₁ₜ + w₂ₜ`), `decide` fires `⊕_T` (`max` at `T=0`) → `decide(t*)` for
`t* = argmax`. If `induction_copy(t₁)` holds (an induction head matched), the recursive clause adds
`retrieved(t₁)` regardless of the numeric margin — the *macro* fires structurally. The answer
`lfp(T_P) ⊇ {decide(t*), retrieved(t₁)}` is the program's least model; the same `decide(t*)` for every
`T`, with the provenance weight (margin / softmax mass) the only `T`-dependent annotation.

## 7.5 A grammar surface (DCG-style)?

Worth asking, because the empirical work finds the model's core *is* a grammar (a compact, corpus-invariant,
closed-class "generic-grammar head", with recursive syntax in the composition). Could a Definite-Clause-
Grammar surface make the recursive PIC-LP rules readable?

- **Pure Datalog has no native DCGs.** Classic Prolog DCGs thread the input as *difference lists*, which
  need function terms (compound structures); Datalog is function-free, so DCGs don't lift verbatim, and
  Soufflé (bottom-up) evaluates the wrong direction for Prolog-style DCG parsing.
- **But the DCG *idea* maps cleanly — via position indices.** A grammar rule `A → B C` becomes the Datalog
  clause `A(i,k) :- B(i,j), C(j,k)` over string **positions** `i,j,k` (chart / CYK / Earley parsing —
  *the* classic demonstration of Datalog's expressiveness). This is fully supported, and is exactly the
  shape PIC-LP already has once you add the position argument; Soufflé also offers records/components if you
  want explicit term structure.
- **Where it would help PIC.** Not the flat single-position decode (a terminal/lexical lookup — no grammar
  there). The readability win is at the **sequence / recursive-compositional** level: the induction macro
  (`retrieved(T) :- induction_copy(T)`), the syntactic core, the unnamed number-mover circuit — the rules
  that span positions and recurse. Writing *those* as grammar productions
  `MOVE → MATCH COPY` and **compiling down to position-indexed PIC-LP clauses** would read far better than
  the threaded `(i,j,k)` form, and it lines up directly with the finding that the entangled core is a
  Chomsky-but-simpler grammar. So: yes — a DCG-style **surface** over the position-indexed PIC-LP, for the
  recursive fragment only. (`reach_all` in `PIC_Logic.thy` is the minimal kernel witness that the lfp
  semantics already supports the unbounded recursion such a grammar surface would compile to.)

## 8. Status

- **proved** (i-orca, `PIC_Logic.thy` + `PIC_Forward.thy`): the `T_P`/least-model semantics
  (`Tp`/`answer`); the magic-sets/demand-closure soundness (`demand_restrict_lfp`, + the decidable
  rule-level criterion `demand_closed_TpI`); the margin (clause-weight) certificate, threshold-tight
  (`decode_margin_certified`); unbounded recursion in the lfp (`reach_all`); the `μ_t=0 ≠ irreducible`
  gap (`PIC_Core`/`Separation`); each input grounds the program to a model
  (`PIC_Core.encoder_slice_logit`); **and the forward pass IS the least fixpoint** —
  `lfp_is_trajectory`: `answer(Player) = {(ℓ, Rℓ) | ℓ}` exactly, so `lfp(T_P)` of the full layer
  program equals the forward residual stack and the decode reads its final layer (`lfp_decode_logits`).
- **empirical** (fieldrun): the emitted program reproduces the model decode (`export --logic`, run in
  Soufflé); the induction rule is the one clean recursive clause measured.
- **open**: the semantics of *unbounded* recursion depth (induction composition) as a recursive PIC-LP
  program with a genuine (not finite-stratified) fixpoint over the position axis — the DCG-surface
  direction (§7.5).

---

*Next build targets (the surface): a thin `pic`-LP surface syntax so `fieldrun export --logic`
round-trips through the formal `T_P`/`lfp` semantics; and the DCG-style grammar layer (§7.5) for the
recursive fragment, compiling to position-indexed clauses.*
