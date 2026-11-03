"""Independent genome-to-peptide check of the exact precomputed prediction set."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "full_database_study_2026-09-13/vendor"))
from twobitreader import TwoBitFile
CODONS = dict(zip((a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"),
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"))
COMP = str.maketrans("ACGTN", "TGCAN")


def main():
    gtf = ROOT / "ShortStop2025_code/gencode_smorfs.gtf"
    byid = defaultdict(list)
    for line in gtf.read_text().splitlines():
        p = line.split("\t")
        if len(p) != 9 or p[2] != "CDS":
            continue
        ident = re.search(r"gene_id ([^;]+);", p[8]).group(1)
        byid[ident].append((p[0], int(p[3])-1, int(p[4]), p[6]))
    genome = TwoBitFile(str(ROOT.parent / "full_database_study_2026-09-13/references/hg38.2bit"))
    predictions = pd.read_csv(ROOT / "sources/shortstop_v1.2_gencode_smorfs_classifications.csv")
    metadata = json.loads((ROOT / "sources/ShortStop_v1.2.0_release_metadata.json").read_text())
    assert metadata["published_at"][:10] <= "2026-10-31"
    constructs = pd.read_parquet(ROOT / "Guo2026_all_assay_constructs.parquet")
    rows = []
    ids = set(predictions.orf_id) & set(constructs.protein_identifier)
    for ident in sorted(ids):
        blocks = byid.get(ident, [])
        assert blocks, ident
        assert len({(x[0], x[3]) for x in blocks}) == 1
        chrom, _, _, strand = blocks[0]
        dna = "".join(genome[chrom][a:b].upper() for _, a, b, _ in sorted(blocks, key=lambda x:x[1]))
        if strand == "-":
            dna = dna.translate(COMP)[::-1]
        raw = "".join(CODONS.get(dna[i:i+3], "X") for i in range(0, len(dna)-2, 3))
        seq = raw.rstrip("*")
        rows.append(dict(protein_identifier=ident, reference_sequence=seq,
            shortstop_implementation_sequence=raw.replace("*", ""), cds_nt_length=len(dna),
            divisible_by_three=len(dna)%3 == 0, internal_stop="*" in seq,
            chrom=chrom, strand=strand, blocks=json.dumps([[a,b] for _,a,b,_ in sorted(blocks)])))
    mapping = pd.DataFrame(rows)
    checked = constructs.merge(mapping, on="protein_identifier", validate="many_to_one").merge(predictions,
        left_on="protein_identifier", right_on="orf_id", validate="many_to_one")
    checked["exact_reference_sequence"] = checked.protein_sequence.eq(checked.reference_sequence)
    checked["exact_shortstop_sequence"] = checked.protein_sequence.eq(checked.shortstop_implementation_sequence)
    checked["initiator_only_difference"] = checked.apply(lambda x: isinstance(x.protein_sequence,str) and len(x.protein_sequence)==len(x.reference_sequence)
        and x.protein_sequence[1:]==x.reference_sequence[1:] and x.protein_sequence[:1]!=x.reference_sequence[:1],axis=1)
    checked.to_parquet(ROOT / "ShortStop_Guo_sequence_checked_mapping.parquet", index=False)
    summaries = []
    for name, x in checked.loc[checked.group.eq("microprotein_WT")].groupby("source_sheet"):
        summaries.append(dict(source_sheet=name, matched_ids=len(x), exact_peptides=int(x.exact_reference_sequence.sum()),
            initiator_only_difference=int(x.initiator_only_difference.sum()),
            exact_implementation_peptides=int(x.exact_shortstop_sequence.sum()),
            internal_stop_rows=int(x.internal_stop.sum()),
            other_differences=int((~x.exact_reference_sequence & ~x.initiator_only_difference).sum())))
    out = dict(release="v1.2.0", released_utc=metadata["published_at"],
        predictions_sha256=hashlib.sha256((ROOT/"sources/shortstop_v1.2_gencode_smorfs_classifications.csv").read_bytes()).hexdigest(),
        gtf_sha256=hashlib.sha256(gtf.read_bytes()).hexdigest(),
        genome="Same hg38.2bit source as verified reanalysis reproduction_inputs.json",
        primary_analysis_rule="Exact peptide sequence, valid whole-codon length, no internal stop", WT_mapping=summaries,
        model_assay_associations_computed=False)
    (ROOT / "SHORTSTOP_SEQUENCE_MAPPING_AUDIT.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
