"""Read public supplementary tables; preserve constructs and measurement context.

The workbook is never modified. Outputs are analytical intermediates, not a new
spreadsheet deliverable. Reported values are reporter-assay scores, not half-lives.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources/Guo2026_mmc4.xlsx"


def main():
    sheets = pd.read_excel(SOURCE, sheet_name=None)
    result = {"source": str(SOURCE), "source_url": "https://ars.els-cdn.com/content/image/1-s2.0-S1097276526002753-mmc4.xlsx",
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "definition": "Table S3 reporter stability scores; biological replicate columns as defined in Description",
              "endogenous_half_life": False, "sheets": {}}
    observations = []
    for name, x in sheets.items():
        if name == "Description":
            continue
        x = x.copy()
        x["source_sheet"] = name
        x["source_excel_row"] = np.arange(len(x)) + 2
        x["sequence_valid"] = x.protein_sequence.map(lambda s: isinstance(s, str) and bool(s) and set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))
        x["sequence_length"] = x.protein_sequence.str.len()
        observations.append(x)
        info = dict(rows=len(x), unique_sequences=int(x.protein_sequence.nunique()),
                    missing_sequence_rows=int(x.protein_sequence.isna().sum()),
                    valid_sequence_rows=int(x.sequence_valid.sum()),
                    unique_identifiers=int(x.protein_identifier.nunique()),
                    duplicated_identifiers=int(x.protein_identifier.duplicated().sum()),
                    group_counts=x.group.value_counts(dropna=False).to_dict())
        if "mean_stability" in x:
            info["measured_counts"] = {col: int(x[col].notna().sum()) for col in ["stability_r1", "stability_r2", "mean_stability"]}
            info["maximum_mean_reconciliation_error"] = float(np.nanmax(abs(x.mean_stability - (x.stability_r1 + x.stability_r2) / 2)))
            w = x.loc[x.group.eq("microprotein_WT")]
            info["WT_replicate_spearman"] = float(spearmanr(w.stability_r1, w.stability_r2, nan_policy="omit").statistic)
        result["sheets"][name] = info
    pd.concat(observations, ignore_index=True).to_parquet(ROOT / "Guo2026_all_assay_constructs.parquet", index=False)
    a = sheets["GFP_MP"].query("group == 'microprotein_WT'")
    targets = pd.read_csv(ROOT.parent / "publication_novelty_review_2026-10-31/Guo2026_release/Regression_train_data.csv").rename(columns={"protein sequence": "protein_sequence"})
    t = targets.merge(a, on="protein_sequence", validate="one_to_one")
    result["released_regression_target_provenance"] = dict(matched_rows=len(t), target_rows=len(targets),
        source_sheet="GFP_MP", source_group="microprotein_WT",
        maximum_absolute_difference=float(np.max(abs(t["Experimental stability"] - t.mean_stability))),
        independent_of_supplementary_GFP_MP_cohort=False)
    b = sheets["GFP_MP_mutagenesis"].query("group == 'microprotein_WT'")
    cross = a.merge(b, on=["protein_identifier", "protein_sequence"], suffixes=("_initial", "_mutagenesis"), validate="one_to_one")
    result["same_WT_construct_across_initial_and_mutagenesis_library"] = dict(matched_rows=len(cross),
        spearman=float(spearmanr(cross.mean_stability_initial, cross.mean_stability_mutagenesis).statistic),
        median_absolute_difference=float(np.median(abs(cross.mean_stability_initial-cross.mean_stability_mutagenesis))),
        q90_absolute_difference=float(np.quantile(abs(cross.mean_stability_initial-cross.mean_stability_mutagenesis), .9)),
        interpretation="Different library measurements; do not pool or replace within-library mutant baselines with the original measurements")
    cross.to_parquet(ROOT / "Guo2026_WT_cross_library_measurements.parquet", index=False)
    # Mutation pairs use the wild type recorded in the SAME library, not an
    # identically named sequence measured in a different library.
    x = sheets["GFP_MP_mutagenesis"]
    parents = x.loc[x.group.eq("microprotein_WT")].copy()
    pairs = []
    for group in ["L_to_A", "microprotein_10mer_ext", "G_to_D", "A_to_D", "V_to_D", "C_to_D", "R-3_to_K", "N_to_E", "N_to_P", "RXXG_to_KXXD", "RG_to_KD", "PG_to_DD", "GG_to_DD", "AA_to_DD"]:
        p = x.loc[x.group.eq(group)].merge(parents, on="protein_identifier", suffixes=("_mut", "_WT"), validate="one_to_one")
        p["mutation_group"] = group
        p["edit_check_rule"] = "exact global Leu-to-Ala" if group == "L_to_A" else ("exact DNYNEPKANQ extension" if group == "microprotein_10mer_ext" else "same length and nonidentical; specific substitution not verified")
        p["pair_sequence_valid"] = p.protein_sequence_mut.map(lambda s: isinstance(s, str) and bool(s) and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")) & p.protein_sequence_WT.map(lambda s: isinstance(s, str) and bool(s) and set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))
        p["mutation_verified"] = False
        for idx, row in p.loc[p.pair_sequence_valid].iterrows():
            w, m = row.protein_sequence_WT, row.protein_sequence_mut
            if group == "L_to_A":
                p.at[idx, "mutation_verified"] = m == w.replace("L", "A") and m != w
            elif group == "microprotein_10mer_ext":
                p.at[idx, "mutation_verified"] = m == w + "DNYNEPKANQ"
            else:
                # Preserve the exact edit description, without assuming a
                # terminal edit that was not actually checked.
                p.at[idx, "mutation_verified"] = len(w) == len(m) and m != w
        pairs.append(p)
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.to_parquet(ROOT / "Guo2026_within_library_mutant_pairs.parquet", index=False)
    result["within_library_pairs"] = pairs.groupby("mutation_group").agg(rows=("protein_identifier", "size"),
        valid_sequences=("pair_sequence_valid", "sum"), edit_verified=("mutation_verified", "sum")).reset_index().to_dict("records")
    result["limitations"] = ["Full STAR Methods normalization and sequencing-depth/censoring details not obtained",
        "All these conditions belong to one paper, not independent studies",
        "Two biological replicates do not justify assuming a known variance or endogenous half-life",
        "Cross-orientation measurements are different fusion constructs, not interchangeable replicate half-lives"]
    (ROOT / "GUO_SUPPLEMENT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
