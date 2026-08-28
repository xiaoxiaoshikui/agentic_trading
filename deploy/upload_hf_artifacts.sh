#!/usr/bin/env bash
set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:?Set HF_REPO_ID, for example xiaoxiaoshikui/agentic-trading-results}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-.}"
HF_BIN="${HF_BIN:-hf}"
STAGING_DIR="${STAGING_DIR:-artifacts/hf_release}"

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  if [ -x .venv/bin/hf ]; then
    HF_BIN=".venv/bin/hf"
  else
    echo "Cannot find hf CLI. Install with huggingface_hub or run: .venv/bin/hf auth login" >&2
    exit 1
  fi
fi

"${HF_BIN}" auth whoami >/dev/null

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

python3 - <<'PY'
from pathlib import Path
import shutil

root = Path(".")
stage = Path("artifacts/hf_release")
patterns = [
    "README.md",
    "LICENSE",
    "docs/figures/cgcma_architecture.png",
    "paper/*.csv",
    "paper/*.json",
    "paper/*.md",
    "paper/latex/main.tex",
    "paper/latex/references.bib",
    "paper/latex/sections/*.tex",
    "paper/latex/figures/cgcma_arch_v2.drawio.png",
    "experiments/configs/icaif_*.json",
]

for pattern in patterns:
    for src in root.glob(pattern):
        if src.is_dir():
            continue
        rel = src.relative_to(root)
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

manifest = stage / "ARTIFACT_MANIFEST.md"
manifest.write_text(
    "# AgenticTrading Artifact Release\n\n"
    "This bundle contains public result tables, selected paper source files, "
    "experiment configs, and README material. Raw market/news data, API keys, "
    "large caches, private reviews, and submission workflow files are intentionally excluded.\n",
    encoding="utf-8",
)
PY

UPLOAD_ARGS=(
  upload
  "${HF_REPO_ID}"
  "${STAGING_DIR}"
  "${HF_PATH_IN_REPO}"
  "--repo-type=${HF_REPO_TYPE}"
  "--commit-message=Upload AgenticTrading public artifacts"
)

if [ "${HF_PRIVATE:-0}" = "1" ]; then
  UPLOAD_ARGS+=("--private")
fi

"${HF_BIN}" "${UPLOAD_ARGS[@]}"
