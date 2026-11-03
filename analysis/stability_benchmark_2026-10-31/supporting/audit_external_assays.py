"""Audit external endpoint feasibility without model-outcome associations."""
from pathlib import Path
import hashlib
import json
import zipfile
import pandas as pd
CODONS = dict(zip((a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"),
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"))

ROOT = Path(__file__).resolve().parent


def main():
    s=ROOT / "sources"
    constructs=pd.read_excel(s/"Yang2024_mmc11.xlsx",sheet_name="sequences for pcDNA3.1",header=3)
    constructs=constructs.loc[constructs["ORF name"].notna()].copy()
    rows=[]
    for _,r in constructs.iterrows():
        dna=r["Synthsized sequence with Flag and restriction enzymes"].strip().upper()
        start=dna.find("ATG")
        coding=dna[start:]
        fusion="".join(CODONS.get(coding[i:i+3],"X") for i in range(0,len(coding)-2,3)).split("*")[0]
        has_flag=fusion.endswith("DYKDDDDK")
        native=fusion[:-8] if has_flag else fusion
        rows.append(dict(orf_name=r["ORF name"],native_sequence=native,fusion_sequence=fusion,
            flag_verified=has_flag,length_verified=len(native)==int(r["Peptide length (aa)"]),
            native_length=len(native),published_PepScore=r.PepScore,
            construct_source="Supplementary Data 8, sequences for pcDNA3.1"))
    peptides=pd.DataFrame(rows)
    assert peptides.flag_verified.all() and peptides.length_verified.all()
    measurements=[]
    for name in ["Fig 4d","Fig 4n"]:
        x=pd.read_excel(s/"Yang2024_mmc12.xlsx",sheet_name=name,header=2)
        x=x.loc[x.iloc[:,0].notna()].copy()
        for _,r in x.iterrows():
            # Read the author-provided replicate cells; do not substitute plot
            # means or treat absent conditions as zero observations.
            row=dict(orf_name=r.iloc[0],source_sheet=name)
            reps=r.iloc[1:6] if name=="Fig 4d" else r.iloc[1:5]
            row["untreated_replicates"]=[float(v) for v in reps]
            row["untreated_mean"]=sum(row["untreated_replicates"])/len(row["untreated_replicates"])
            if name=="Fig 4d":
                row["MG132_replicates"]=[float(v) for v in r.iloc[6:10]]
                row["MG132_mean"]=sum(row["MG132_replicates"])/4
            measurements.append(row)
    measured=peptides.merge(pd.DataFrame(measurements),on="orf_name",validate="one_to_one")
    assert len(measured)==37
    guo=pd.read_parquet(ROOT/"Guo2026_all_assay_constructs.parquet")
    natural=set(guo.loc[guo.group.isin(["microprotein_WT","microprotein_ms_HLA","microprotein_ms_nonHLA"]),"protein_sequence"])
    measured["exact_Guo_natural_sequence_overlap"]=measured.native_sequence.isin(natural)
    measured.to_parquet(ROOT/"Yang2024_independent_constructs_and_assays.parquet",index=False)
    with zipfile.ZipFile(s/"ADatlas2026_microprotein_master.zip") as z:
        with z.open("microprotein_master.csv") as f:
            brain=pd.read_csv(f,low_memory=False)
    n=pd.read_parquet(ROOT.parent/"thesis_reanalysis_2026-10-31/norf_catalogue_audit.parquet")
    updated=set(n.loc[n.eligible_updated,"sequence"])
    output=dict(Yang2024=dict(independent_publication=True,assay="GFP-normalized fraction of FLAG-positive cells, not endogenous half-life",
        native_constructs=len(measured),uORFs=int(measured.source_sheet.eq("Fig 4d").sum()),other_ORFs=int(measured.source_sheet.eq("Fig 4n").sum()),
        sequences_verified_from_synthesized_oligos=True,Cterminal_FLAG_verified=True,
        minimum_native_length=int(measured.native_length.min()),maximum_native_length=int(measured.native_length.max()),
        length_le30=int(measured.native_length.le(30).sum()),exact_Guo_natural_sequence_overlap=int(measured.exact_Guo_natural_sequence_overlap.sum()),
        no_model_associations_computed=True,
        limitations=["Small validation set; ORFs intentionally span PepScore levels", "C-terminal tag masks native terminal sequence",
                     "Assay score is not on the Guo GFP-fusion scale", "Related-sequence and predictor-training overlap still need audit"]),
        ADatlas2026=dict(master_rows=len(brain),unique_sequences=int(brain.sequence.nunique()),
            exact_updated_catalogue_sequences=int(brain.loc[brain.sequence.isin(updated),"sequence"].nunique()),
            exact_Guo_natural_sequences=int(brain.loc[brain.sequence.isin(natural),"sequence"].nunique()),
            no_model_associations_computed=True,
            selection_circularity="Author gold-standard filtering uses ShortStop SAM labels in has_RiboSAM and DIA_evidence; those selected subsets cannot independently validate ShortStop",
            independent_candidate_branch="DDA_evidence uses unique spectral counts independently of ShortStop; still requires spectral-quality and peptide-uniqueness audits",
            limitation="Public master contains derived aggregate values; it does not by itself establish individual matched protein/RNA values or endogenous degradation rates"))
    (ROOT/"EXTERNAL_ENDPOINT_FEASIBILITY.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
