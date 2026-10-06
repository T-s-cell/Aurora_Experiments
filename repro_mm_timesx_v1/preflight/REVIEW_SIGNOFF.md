# EXP-011 preflight review sign-off (M48T512)

- Date: 2026-10-07
- Reviewer: user (manual, item-by-item)
- Audit pack reviewed: `temp/Aurora_TimesX_mm_preflight_20261007.tar.gz` (48 files)

## Verdict

**Approved for the full run** with the fixed-text-budget M48T512 baseline
configuration, unchanged. Approval token: `74b08ecac1ed`
(sha256(protocol_mm.json) first 12 hex).

## Findings

1. Rear perturbation index offset **fixed and verified**: input indices
   0..125 fixed ([CLS] at 0; retained rows = content positions 1..125),
   perturbation from 126. Text-effectiveness, determinism, and
   weights-unchanged checks all pass (MM-A/B/C).
2. English-date Events scan **added**. Reviewer checked **all 162 flagged
   windows / 201 date marks** individually (`preflight/
   events_english_test_preflight_input.csv`): **no clear after-the-fact
   result leakage found**; flagged wording is schedules, effective dates, or
   forward-looking statements. **Actual publication times remain
   independently unverified** — recorded as a standing conclusion boundary.
3. Pack hashes verified; text cache and the EXP-010 comparison configuration
   are consistent with the frozen state.

## Scope statement (unchanged)

This experiment measures only the gain of the current fixed input scheme
(M48T512) relative to EXP-010 (A48); it does not represent Aurora's
capability under full text conditions. No training/fine-tuning on TimesX.
Pretraining overlap not audited; event publication timing not independently
verified.
