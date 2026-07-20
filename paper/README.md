# arXiv preprint

`mushin.tex` is a short software preprint describing `mushin`, ready to submit to
[arXiv](https://arxiv.org/) (which mints a citable identifier). `paper.bib` holds
the references.

## Before submitting

- Add your **affiliation** and **ORCID** (a `\thanks{}` on the author) in
  `mushin.tex`.
- Review the claims and the related-work framing.

## Build the PDF

```bash
cd paper
pdflatex mushin
bibtex mushin
pdflatex mushin
pdflatex mushin
```

CI also builds `mushin.pdf` on any change under `paper/` (see
`.github/workflows/paper-pdf.yml`) and uploads it as an artifact.

## Submitting to arXiv

Upload `mushin.tex` and `paper.bib` (arXiv compiles LaTeX server-side; it runs
BibTeX for you). Suggested primary category: `cs.LG` (cross-list `cs.MS` /
`cs.SE` if you like). arXiv will assign an identifier you can cite; pair it with
the Zenodo DOI in `CITATION.cff` so both the software and the write-up are
citable.
