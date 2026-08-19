# Paper workspace (ACM MM transition)

This folder now tracks the project's transition from a finance/ML paper into an ACM MM-oriented submission on multimodal market intelligence.

Files:
- `acm_mm_execution_plan.md`: end-to-end roadmap from current repo state to a viable ACM MM submission.
- `main_draft.md`: writing skeleton for the ACM MM paper.
- `results_pack.md`: required tables, figures, and claim-evidence links.
- `reproducibility.md`: current reproducibility status plus the missing artifact pieces to add before submission.
- `submission_checklist.md`: venue-facing and project-specific submission gate.
- `figures/`: exported paper figures.
- `tables/`: exported paper tables.

Recommended order:
1. Read `acm_mm_execution_plan.md` and lock the task definition.
2. Build the missing multimodal dataset and model pipeline.
3. Populate `results_pack.md` only from frozen experiment outputs.
4. Fill `main_draft.md` Sections 3-6 before writing the abstract.
5. Use `submission_checklist.md` as the final gate, not as a wish list.

Current repo reality:
- The existing `src/tom/` pipeline is a strong price-series baseline and ablation anchor.
- `src/web_intelligence.py` is useful prior art, but it is not yet a paper-grade multimodal pipeline.
- The current reviewer suite can be reused as a downstream trading evaluation layer after the multimodal stack is added.
