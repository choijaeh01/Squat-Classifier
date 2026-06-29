# Results Artifacts Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a curated, GitHub-friendly subset of experiment results without committing the full generated `results/` tree.

**Architecture:** Keep original `results/` read-only and ignored. Copy selected CSV, Markdown, and PNG artifacts into `docs/results_artifacts/` with per-experiment README files and a top-level index. Commit only the curated artifact mirror and documentation.

**Tech Stack:** Git, Markdown, existing CSV/PNG artifacts, shell copy commands.

---

### Task 1: Create Curated Artifact Mirror

**Files:**
- Create: `docs/results_artifacts/README.md`
- Create: `docs/results_artifacts/<experiment>/README.md`
- Copy selected existing CSV/MD/PNG files from `results/` into `docs/results_artifacts/`

- [ ] **Step 1: Create destination folders**

Run:

```bash
mkdir -p docs/results_artifacts/{full_supervised_matrix_v1,literature_baseline_full_extension_v1,controlled_feature_extractor_comparison_v1,v3_component_ablation_v1,xgboost_only_completion_v1,review_bundles}
```

Expected: directories exist, original `results/` is unchanged.

- [ ] **Step 2: Copy selected lightweight artifacts**

Run copy commands for aggregate metrics, paired differences, classwise/subjectwise summaries, feature importance summaries, selected merged comparisons, and selected figures. Exclude checkpoints, raw predictions where not needed, `.npy`, `.pt`, `.pth`, `.ckpt`, and full generated result trees.

- [ ] **Step 3: Write README files**

Use `apply_patch` to create the top-level artifact index and per-experiment README files with original result path, included files, and interpretation warnings.

- [ ] **Step 4: Validate artifact size and ignored files**

Run:

```bash
du -sh docs/results_artifacts
find docs/results_artifacts -type f | sort
git status -sb
```

Expected: curated artifacts are small enough for GitHub; `results/` remains ignored.

- [ ] **Step 5: Commit and push**

Run:

```bash
git add docs/results_artifacts docs/superpowers/plans/2026-06-29-results-artifacts-curation.md
git commit -m "docs: curate shareable result artifacts"
git push
```

Expected: GitHub receives the curated result artifacts, not the full generated results.
