# ACM MM 2026 LaTeX manuscript

This manuscript currently targets the **ACM Multimedia 2026 main technical track**.
The active source file is `paper/latex/main.tex`, which uses the ACM article template:

- `\documentclass[sigconf,review,anonymous,screen]{acmart}`

The older LNCS files under `paper/latex/lncs_official/` are legacy reference material only and are **not** the active submission template.

## Build

From the repository root:

```powershell
Set-Location "paper/latex"
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Files
- `main.tex`: entry point.
- `sections/*.tex`: paper sections.
- `references.bib`: bibliography database.

## Notes
- Default mode is anonymous for double-blind submission.
- The current manuscript is already configured with anonymous placeholder author metadata.
- ACM MM 2026 main-track length is up to 8 pages, plus up to 2 additional reference-only pages.
- Supplementary material must be uploaded separately; do not append an appendix after the references in the main PDF.
