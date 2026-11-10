# Experimental context limits transfer of microprotein sequence scores

Robin Gounder (University of Cambridge) and Samuel Schmidt (Vaionex Corporation; Cornell University).

This study asks when a microprotein sequence score remains useful outside the assay and population in which it was trained. The central result is that transfer depends on the measurement, construct context, input coverage and decision metric. Reporter prediction does not establish native protein turnover or peptide-dependent function.

## Paper, appendix and data

- [Read the paper (PDF)](paper/manuscript.pdf).
- [Read the appendix: S1 Text (PDF)](paper/S1_Text.pdf).
- [Download the frozen v0.1.1 data and code](https://github.com/Kohze/microprotein-score-transfer/releases/download/v0.1.1/microprotein_score_transfer_public_reproducibility_2026-11-10.zip).
- [Browse the S1-S18 supporting-data map](SUPPORTING_DATA_MAP.csv).
- [View all versioned releases and downloads](https://github.com/Kohze/microprotein-score-transfer/releases).

The appendix contains supplementary methods, detailed results, six supporting
figures and two supporting tables. Editable sources are `paper/manuscript.tex`
and `paper/S1_Text.tex`. The v0.1.1 frozen analysis is pinned to commit
`1a22e80f60b46fefa8f370100b2e322457f3f518`. The v0.1.2 documentation update
identifies that published release explicitly; it changes no predictions,
confidence intervals, data or scientific results. Its release also supplies
the revised paper and appendix as separate PDF downloads.

## Start here

- `paper/manuscript.pdf`: revised manuscript, five inline figures.
- `paper/S1_Text.pdf`: supplementary methods, supporting-file map, detailed results and exploratory diagnostics, six figures.
- `paper/figures/`: publication-numbered vector PDF/SVG and high-resolution PNG artwork, with frozen plotting tables.
- `analysis/stability_benchmark_2026-10-31/`: original core benchmark scripts, saved outputs and dependence/feature records.
- `analysis/paired_context_2026-11-01/`: same-peptide context-transfer extension.
- `analysis/novelty_audit_2026-11-01/`: canonical-tile transfer, paired effects, published baselines, input/exposure controls and exploratory candidate joins.

## Offline verification

Use Python 3.11 with the packages in `requirements-verification.txt`. From this directory run:

```sh
python -m pip install -r requirements-verification.txt
python verify_saved_results.py
```

The verifier checks file hashes and recalculates point metrics directly from frozen row-level outcomes/predictions. It imports no study analysis module or model, accesses no network and fits no predictor. It checks the core 261 correlations and 140 reported MAEs; same-peptide transfer; canonical-tile transfer; independent paired changes; neural paired effects; and terminal-model exposure/input strata. The JSON output reports the exact checks and tolerances. Constant training-mean baselines in the paired-context table are excluded from that point-metric check because their fold means are not saved as prediction columns.

This is saved-result reproducibility. Bootstrap confidence intervals, raw-workbook reconstruction, descriptor recomputation, pretrained-model inference, training and native experimental validation are outside this verifier's scope. Frozen interval tables and original analysis/protocol scripts are included for inspection; passing does not certify them afresh.

## Reproducing the original workflow

See `REPRODUCING.md`. Original scripts preserve their historical directory layout and protocol bytes. They require upstream sources and environments that are intentionally omitted here. Do not treat every historical `verify_*.py` as an offline entry point: many require those original workbooks or model installations. Only the root verifier is self-contained against this bundle. No complete clean-machine refit has been claimed.

## Interpretation and release status

The core benchmark is distinguished from later retrospective transfer extensions and exploratory diagnostics. These labels describe their role in the paper and do not imply preregistration or prospective confirmation. No new predictor, dataset or model fit was added for this release.

This is a research manuscript and reproducibility resource, not an SSRN
submission or a claim of peer review. The publication revision was released
as [v0.1.1](https://github.com/Kohze/microprotein-score-transfer/releases/tag/v0.1.1)
on 10 November 2026. The v0.1.2 documentation update makes its data and appendix
links explicit and incorporates no new analysis.
Robin Gounder performed the main study work and original drafting; Samuel Schmidt
contributed review and editing. Both authors declare no competing interests.
Robin was supported by a BBSRC DTP studentship (reference 2116112 in the Cambridge
thesis record). `SUPPORTING_DATA_MAP.csv` resolves every S1-S18 label to actual
bundle paths, including the added frozen native-context annotations.

Related work: [Guide selection transfers depletion effects across noncanonical
ORF screens](https://github.com/Kohze/norf-guide-selection) addresses earlier
CRISPR guide choice and later depletion estimates. Some ORFs and reference
resources are shared; the present reporter and paired-edit endpoints differ.

## Attribution, rights and citation

Original source data and predictor methods are cited in the manuscript and `provenance/UPSTREAM_SOURCE_REGISTRY.json`. Derived tables retain source identifiers and upstream provenance. Raw papers/workbooks, third-party code copies, weights and runtime libraries are excluded. Their rights remain with their respective owners. No blanket licence is granted for upstream data or models. Original study code is released under the MIT licence in `LICENSE`. See `RIGHTS_AND_ATTRIBUTION.md`.

Use `CITATION.cff` to cite this draft and identify the actual release tag/commit when available.

AI disclosure: OpenAI Codex agents contributed substantially to source discovery, literature synthesis, and mathematical exposition.

The abstract and Fig3b have also been revised; see `ABSTRACT_AND_FIGURE_REVISION.md`.
