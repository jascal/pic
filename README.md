# pic — Projective Incidence Calculus

The **foundational algebraic formalism** for the analyze → verify → learn program. PIC defines, once
and precisely, the algebra that three sibling repos already operate in:

- **[`fieldrun`](https://github.com/jascal/fieldrun)** *(analyze)* — a pure-Rust runtime that decompiles
  a frozen transformer's decode into per-block incidences. PIC is the **spec of what it computes**: the
  residual reading in the log-semiring (reconstruction = 1.00 across architectures).
- **[`i-orca`](https://github.com/jascal/i-orca)** *(verify)* — an Isabelle kernel-proof corpus. PIC is
  the **structures its theorems quantify over** (frame, Gram kernel, semiring, margin, Welch bound).
- **[`pil`](https://github.com/jascal/pil)** *(learn)* — Projective Incidence Learning, frame
  optimization. PIC is the **optimization target and the soundness invariants** the learning loop must
  preserve.

PIC is **descriptive, not aspirational**: it is the algebra the analysis already runs in and the proofs
already quantify over, written down once so a definition in one repo can be a hypothesis in another.

## The one-paragraph idea

A transformer's next-token decision is a *semiring-weighted incidence calculus over a continuous
inner-product geometry*. Propositions (tokens) are points `U_v` in residual space `ℝ^d`; sources
(per-layer DLA blocks, or learned rules) are vectors `d_j`; the incidence of source `j` on token `v` is
`c_j(v) = ⟨d_j, U_v⟩`; the decision aggregates incidences through a semiring chosen by a temperature `T`.
At `T = 1` (log-semiring) PIC **is** the softmax forward pass; at `T → 0` (tropical/max-plus) it is the
geometry of margins and Laguerre power diagrams — the two ends joined by Maslov dequantization. PIC
separates the **frame** (intrinsic geometry of `{U_v}`) from the **decode** (margins, participation
ratio, multiplicity), which is exactly the seam the learning repo exploits: hold the decode fixed,
retrain the frame.

## What's here

```
spec/PIC_SPEC.md        the canonical specification — definitions, semiring family, the proved
                        theorems (each tagged proved / empirical / open), the operational quantities
                        with their file-level pointers, and the learning objective. Reference this.
paper/pic_calculus.tex  the publishable theory paper expanding the spec
paper/references.bib     external references (provenance semirings, tropical geometry, Welch, power diagrams)
```

The spec is the **single source of truth**; the paper expands it. Both are versioned with the proofs
and the code: when a theorem lands in i-orca or a quantity changes in pil/fieldrun, the corresponding
section here is updated and the version bumped.

## Build the paper

```bash
cd paper && latexmk -pdf pic_calculus.tex
```

## Proof-status discipline

Every claim is tagged. **proved** = machine-checked in the i-orca Isabelle corpus; **empirical** =
measured by the fieldrun runtime / pil experiments; **open** = a flagged hypothesis. The two-sided
packing bounds (frame capacity, routing rank, routing Welch, head/tail certificate, margin certificate,
irreducibility) are *proved*; the soundness anchor (PIC = forward pass) and the `τ★` effective-rank law
are *empirical*; the coherence→margin link and global irreducibility are *open*. See `spec/PIC_SPEC.md`
§5 and §7.
