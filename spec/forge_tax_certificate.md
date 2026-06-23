# The forge-tax certificate: activation-relative irreducibility

**Status:** formulation draft (v0.1), the well-posed successor to "global irreducibility" (`PIC_SPEC.md`
§7). The pure-*behavioral* global notion is **vacuous** — with free gates and a free frame any decision
re-geometrizes to single-source (the adversarial-collaborator negative result). The forge tax is therefore
**not** a behavioral invariant. It *is* an **activation-relative** one: fix the model's measured
computation and ask whether re-*representing* can turn a composed decision into retrieval. This document
states that question precisely, gives a **tractable (LP) certificate** with infeasibility duals, the
stronger (bilinear) variant, and the empirical protocol on real `fieldrun` data.

---

## 1. What is measured vs free

The negative result says the freedom must **not** include the computation. So:

**Measured (fixed), from the model via `fieldrun`:**
- a finite set of decode positions `X = {x₁,…,x_N}` (held-out contexts);
- sources `S`, `|S| = M` (the DLA blocks; or, at neuron granularity, the output-weight columns);
- per source `j` and position `x`, the **contribution vector** `d_j(x) ∈ ℝ^d` — the block's additive write
  to the residual (fieldrun's `residual_decomp`), so the residual is `r_x = Σ_{j∈S} d_j(x)`;
- the decoded token `t_x ∈ V` and a competitor set `W_x ⊆ V` (the top-K candidates, `t_x ∈ W_x`).

The `d_j(x)` are the model's **computation** — what each source actually wrote on each input. They stay
fixed (freeing them is what made the behavioral version collapse).

**Free (the representation): the decoder frame.**
- `U : V → ℝ^d` and bias `b : V → ℝ` — a candidate **output representation**. The realization decodes
  `argmax_v ⟨r_x, U_v⟩ + b_v`.

Re-choosing `(U, b)` while holding `{d_j(x)}` fixed is exactly "re-represent the output space without
changing the computation." (Neuron reading: `d_j(x) = g_j(x)·a_j` with measured gate `g_j(x)` and column
`a_j`; fixing `d_j(x)` fixes the firing pattern. A stronger variant frees the columns `a_j` too — §4.)

## 2. The predicates (all linear in the free `(U, b)`)

Fix a margin `γ > 0`.

- **Faithful** — the decoder reproduces every observed decision:
  > `∀ x∈X, ∀ v∈W_x∖{t_x}:   ⟨r_x, U_{t_x} − U_v⟩ + (b_{t_x} − b_v) ≥ γ.`
- **Coalition `P ⊆ S` decides `x`** — the partial residual `r_x^P = Σ_{j∈P} d_j(x)` alone decodes `t_x`:
  > `∀ v∈W_x∖{t_x}:   ⟨r_x^P, U_{t_x} − U_v⟩ + (b_{t_x} − b_v) ≥ γ.`
  **Single-source** is `|P| = 1` (the retrieval / `μ_t` regime).

Because `r_x` and `r_x^P` are **fixed vectors** (measured), every constraint above is **linear** in the
unknowns `(U_v, b_v)_{v∈V}`. This is the whole point of fixing `d_j(x)`: the question becomes a linear
program, not the bilinear mess of the free-computation version.

## 3. The certificate (LP, with infeasibility duals)

> **Definition.** Position `x₀` is **k-irreducible (under the model's activations)** iff there is **no**
> faithful decoder `(U, b)` under which any coalition `P` with `|P| ≤ k` decides `x₀`. If `x₀` is
> 1-irreducible it is **certified computed**: no output representation makes it single-source retrieval
> while preserving all the model's decisions.

**Decision procedure.** For a fixed candidate coalition `P`, the set
`{(U,b) : Faithful ∧ P-decides x₀}` is a polyhedron; non-emptiness is a **linear feasibility** (an LP).
So:
- `x₀` is **k-reducible** ⟺ this LP is feasible for *some* `P` with `|P| ≤ k` (for `k=1`, just the `M`
  singletons — `M` small LPs).
- `x₀` is **k-irreducible** ⟺ all those LPs are **infeasible**, and each infeasibility comes with a
  **Farkas dual certificate** — nonnegative multipliers on the faithfulness + single-source constraints
  exhibiting `0 ≥ γ`. *That dual is the forge-tax certificate*: a finite, checkable witness that the
  computation forces `x₀` to compose, in every faithful representation.

Complexity: poly-time (one LP per candidate coalition; `M` for single-source, `Σ_{i≤k} C(M,i)` for
`k`-coalitions — cap `k` small, it's the interesting regime). Decidable and *cheap*, unlike the
free-computation version (which was either vacuous or, with directions free, NP-hard — §4).

## 4. Stronger variant (free directions; bilinear)

Free not just the frame `U` but the source **write-directions** while keeping the measured **gates**:
`d_j(x) = g_j(x)·a_j` with `g_j(x)` fixed (measured) and `a_j ∈ ℝ^d` free. Now the constraints carry
products `⟨a_j, U_v⟩·g_j(x)` — **bilinear** in `(A, U)`. This certifies the stronger
*representation*-invariance (no relocation of where sources write, nor of the output frame, recovers
retrieval). It is decidable (existential theory of the reals) but **NP-hard** in general; attack with
SDP/SOS relaxations or SMT (nonlinear real arithmetic) on small instances. The **LP version (§3) is a
sound relaxation**: LP-infeasible ⟹ the frame alone cannot reduce `x₀`; the bilinear test asks whether
directions can. Report the LP certificate first; escalate to bilinear only for the positions it leaves
open.

**Cheap necessary conditions (sign-rank).** Single-source decodability of `x₀` via source `j` requires the
sign pattern of the margins `(⟨d_j(x₀), U_{t_{x₀}} − U_v⟩)_{v∈W}` be realizable by one direction; a
**margin-complexity / sign-rank** lower bound on the relevant submatrix gives a *sufficient* condition for
"no single source works" without solving the LP — a fast pre-filter.

## 5. Empirical protocol (on real `fieldrun` data)

1. **Data.** Extend the dump seam to emit the raw contribution vectors: `fieldrun --source-dump` →
   `d_j(x) ∈ ℝ^d` per position `x ∈ X` and block `j` (today `--pil-dump` emits only the incidences
   `⟨d_j(x), U_v⟩` under the model's *own* frame, which is insufficient — re-representation needs the raw
   `d_j(x)`). Modest addition: `residual_decomp` already computes these internally.
2. **Pick a composed target.** A position `x₀` with low native-frame multiplicity `μ_{t}` (no single block
   already wins — the candidate "computed" tokens).
3. **Certify.** Solve the `M` single-source LPs of §3 over the held-out `X`. If all infeasible, `x₀` is
   **1-irreducible** with Farkas duals; sweep `k` for `k`-irreducibility. Pre-filter with the sign-rank
   bound (§4).
4. **Aggregate.** The fraction of decode positions that are `k`-irreducible is a **representation-invariant
   forge-tax measure** — strictly stronger than `μ_t` (which is native-frame-relative and, per §1, not
   invariant). Compare across scale/architecture (does it track the same way the decode-circuit metrics
   do?).

## 6. Honest scope

- **LP version (free frame, fixed computation):** certifies **decoder-invariance** — *no output
  representation* turns `x₀` into retrieval. The computation (block writes) is held fixed; that is the
  point — the forge tax is a property of the computation, not the readout.
- **Bilinear version (free frame + directions, fixed gates):** certifies **representation-invariance** —
  no relocation of where sources write recovers retrieval either. Stronger, NP-hard.
- **Both fix the model's firing/writes** — which is exactly why this is well-posed where the behavioral
  version (which freed the gates) was vacuous. A certified-computed token is one whose compositionality
  survives *re-representation* but is, of course, still produced *by* the model's own computation — the
  claim is "not retrieval under any faithful representation," not "uncomputable."
- **What it does not claim:** global, model-independent hardness. It is a certificate *relative to the
  measured activations of one model on one corpus* — which is precisely the right scope for a mechanistic
  "this token was computed, not looked up" statement (and the scope `fieldrun` already lives in).

---

*Next: implement `fieldrun --source-dump` (raw `d_j(x)`); a small LP harness (`pil/experiments/`) for the
§3 certificate; and the sign-rank pre-filter. The i-orca side can kernel-check the soundness of the Farkas
certificate (LP-infeasibility ⟹ k-irreducible) as a general lemma over the fixed-contribution polyhedron.*
