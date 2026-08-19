# Open Source Release Checklist

Use this checklist before pushing the repository to GitHub.

## Must Do Before First Push

- Keep `.env` private. Only commit `.env.example`.
- Confirm `.venv/`, `__pycache__/`, `logs/`, `tmp/`, generated datasets, data caches, and experiment output directories are ignored.
- Review `git status --short --ignored` before the first commit.
- Run a secret scan over tracked files before pushing.
- Verify the license choice. This repository currently uses MIT; replace `LICENSE` if you prefer Apache-2.0, BSD, GPL, or a research-only license.
- Check third-party files in `paper/` and templates under `paper/latex/` for their own license terms.
- Keep paper reviews, rebuttals, submission packages, and draft PDFs private unless you intentionally want them public.
- Do not publish raw market/news datasets unless their source licenses allow redistribution.

## Recommended First Commit Flow

```bash
git init
git add .
git status --short --ignored
git commit -m "Initial open source release"
```

If `git status` shows files from `.venv/`, `data/`, `logs/`, `experiments/results*`, `experiments/data_cache/`, private paper workflow files, or secret-containing files, stop and update `.gitignore` before committing.

## GitHub Repository Settings

- Enable GitHub secret scanning and push protection.
- Add a repository description that makes clear this is research/educational trading software.
- Add topics such as `trading`, `crypto`, `backtesting`, `multi-agent`, and `llm`.
- Consider using GitHub Releases, DVC, or Git LFS for large reproducibility artifacts instead of committing large generated files.

## Safety Notice

This project contains trading automation code. Keep `DRY_RUN=true` for examples and documentation. Avoid publishing examples that encourage real-money trading without explaining the risks and operational safeguards.
