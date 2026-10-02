# LLEMA and CISE

Both methods use crystal edits, fixed property predictors, task rewards and an island archive. LLEMA selects candidates using point predictions. CISE uses context-specific Gibbs conformal intervals and evaluates the relevant interval endpoints for reward and admission.

## Conformal inference

The predictor, positive error scale and optional quantile model are fixed before search. The standard score is `abs(y-f(x))/scale(x)`. With `quantile_offset=true`, a frozen quantile prediction is subtracted from the calibration score. Live candidates supply structure features and predictions, never measured target labels. Gibbs inference includes test-point correction and splits `alpha_total` across the task properties. Online bounded LSIF supplies a density-ratio basis column estimated from valid probes in the current context.

Probes and main candidates are requested separately. Probes never enter calibration, the archive, output counts or the main-search duplicate set. Invalid main proposals trigger repair with fresh probes. An iteration requires two valid main candidates; exhausted retry limits leave it incomplete. `cci.max_attempts` controls failed admission batches and bounds probe attempts; LLEMA uses `llema.max_attempts`.

Solver failures, nonfinite cutoffs and empty inverse sets cause abstention. Abstained candidates retain evaluation records and adopted fingerprints but cannot enter the archive, parent selection or feedback. Search stops before requesting candidates if no usable initial seed remains.

## Feedback and duplicate handling

Success/failure feedback follows task qualification (`interval_pass` for CISE, `proxy_pass` for LLEMA). Reward scores rank examples within those categories. Initial seeds can supply parents and feedback but are excluded from final output counts. Both methods share the edit-response schema; CISE adds interval feedback. Repair prompts include validation errors.

Both methods reject repeats of task-eligible initial seeds, previously adopted candidates and candidates within the current main batch. Probes may repeat. The fingerprint uses element identities, lattice matrices and fractional coordinates rounded to six decimal places, ignoring site order. It does not establish symmetry-equivalent or composition-level novelty. There is no duplicate comparison against external datasets or other runs.

## Calibration and interpretation

`cise calibrate` checks duplicate property/ID entries, caller-supplied group overlap across splits and, when CIFs are supplied, byte-identical structures across splits and declared hashes. Users remain responsible for scientific grouping, independent calibration and label provenance.

Selected outputs satisfy the point constraints (LLEMA) or interval constraints (CISE). Optional DFT measurements are recorded separately; missing measurements are unresolved. Software tests do not establish scientific coverage under adaptive search.

## Continuation

Code, configuration and assets are sealed before search. Resume requires the same output directory and matching seals. Start a new directory after changing them. Cached requests support continuation; archive resets depend on elapsed time, so a long interruption can affect the trajectory.
