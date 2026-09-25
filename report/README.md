# Technical report

`ltpred_methods.pdf` is the detailed derivation and historical evidence snapshot (v0.7.4,
25 September 2026) for colleagues who already know GWAS, pedigree `h²` and
Falconer's threshold model. `ltpred_methods.tex`
and `efficient_inference.tex` are its source; the six tables it `\input`s from
`paper/tables/` are static LaTeX snapshots of committed benchmark CSVs. What
the note covers, and how it relates to the user documentation, is on the
documentation site's [model reference](../docs/algorithm.md#methods-report).

Rebuild (requires [Tectonic](https://tectonic-typesetting.github.io/)), then
run the evidence check, which binds the release-defining rows to the committed
artifacts and rejects a report source or included table newer than the tracked
PDF:

```bash
cd report
tectonic -X compile ltpred_methods.tex
cd ..
python scripts/check_evidence.py
```

The PDF is tracked as repository documentation and omitted from the source
distribution.
