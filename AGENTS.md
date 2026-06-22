# AGENTS.md — orientation for `pic`

`pic` is the **formalism repo**: the algebraic definition of Projective Incidence Calculus that the
analyze/verify/learn repos reference. It holds **no code and no proofs** — only the spec and the paper.
The proofs live in `i-orca`, the runtime in `fieldrun`, the learner in `pil`. Read
[`README.md`](./README.md) first, then `spec/PIC_SPEC.md`.

## What this repo is for

To write down, *once and precisely*, the algebra the other three repos already operate in, so a
definition in one is a hypothesis in another. PIC is **descriptive** — it documents the calculus the
analysis runs in and the proofs quantify over. It does not propose new behaviour.

## Files

- `spec/PIC_SPEC.md` — **the single source of truth.** Definitions; the semiring family; PIC-native
  notation (§2.5: the incidence arrow `▷`, coalition bracket `⟦·⟧_T`, decision turnstile `⊢_γ`, F/D
  side tags); the proved theorems (§5, each tagged); operational quantities with file-level pointers to
  pil/fieldrun; the learning objective (§6); empirical/open parameters (§7).
- `paper/pic_calculus.tex` — the publishable paper expanding the spec. `paper/references.bib` — externals.

## The discipline that must not slip

**Every claim is tagged `proved` / `empirical` / `open`.**
- `proved` = machine-checked in i-orca. Before tagging something `proved`, the theorem must exist there;
  cite the `.thy` file and lemma name.
- `empirical` = measured by fieldrun / pil. Cite the experiment.
- `open` = a flagged hypothesis. The coherence→margin link, global irreducibility, and the `τ★`
  functional form are `open` — do not promote them.

This mirrors the workspace-wide rule: never state "irreducible / needs / required" from a method
*plateau*; that implies a first-principles proof. The `τ★` section (§7) is the canonical example — a
plausible law (`τ★ ≈ min(exp H, d)`) that the data only *partly* support, reported by what the data show
(capacity-bound, not entropy-bound), not by what would have been tidy.

## When you change this repo

- A theorem lands in i-orca → add it to `spec/PIC_SPEC.md` §5 with its file/lemma and a proof-status tag,
  mirror it in `paper/pic_calculus.tex`, bump the spec version.
- A quantity changes in pil/fieldrun → update its row in §4 (keep the `file:function` pointer exact).
- Keep the spec and paper in sync; the spec is authoritative, the paper expands it.

## What NOT to do

- Do **not** add code, tests, or Isabelle theories here — they belong in the sibling repos.
- Do **not** introduce native notation that doesn't earn its keep; §2.5 is intentionally minimal (only
  what tropical-geometry / provenance-semiring machinery does not already give: `▷`, `⊢_γ`, F/D).
- Do **not** confuse this `pic` repo's theory paper with `pil/paper/pic.tex` — that is a *separate*,
  author-owned empirical paper on the decode-circuit finding. Do not edit it from here.
