# CLAUDE.md

See [`AGENTS.md`](./AGENTS.md) for the canonical orientation. In brief: `pic` is the **formalism repo**
(spec + paper, no code/proofs) defining Projective Incidence Calculus, which `fieldrun` (analyze),
`i-orca` (verify), and `pil` (learn) reference. `spec/PIC_SPEC.md` is the single source of truth.

**The one rule that governs everything here:** every claim is tagged `proved` (machine-checked in
i-orca) / `empirical` (measured in fieldrun or pil) / `open` (flagged hypothesis). Never promote a tag
without the artifact that backs it. This repo's `paper/pic_calculus.tex` is **not** the same document as
`pil/paper/pic.tex` (a separate, author-owned empirical paper); do not edit the latter from here.
