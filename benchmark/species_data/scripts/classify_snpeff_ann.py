#!/usr/bin/env python3
"""Split a SnpEff-annotated VCF stream into synonymous and missense site lists."""

import argparse
import sys


REF_MISMATCH = "WARNING_REF_DOES_NOT_MATCH_GENOME"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read a SnpEff VCF on stdin and write site lists for "
            "synonymous_variant and missense_variant."
        )
    )
    parser.add_argument("--syn-sites", required=True)
    parser.add_argument("--nonsyn-sites", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--chrom-prefix",
        default="",
        help="Prefix restored onto chromosome names when writing sites "
        "that are queried against the original VCF.",
    )
    parser.add_argument(
        "--max-mismatch-fraction",
        type=float,
        default=0.01,
        help="Fail if this fraction of coding calls have a reference mismatch.",
    )
    return parser.parse_args()


def effects_for_alt(ann, alt):
    effects = []
    mismatch = False
    for record in ann.split(","):
        fields = record.split("|")
        if len(fields) < 2:
            continue
        allele = fields[0]
        if allele not in {alt, ""}:
            continue
        effects.extend(part for part in fields[1].split("&") if part)
        if REF_MISMATCH in fields[-1]:
            mismatch = True
    return effects, mismatch


def classify(effects):
    if "missense_variant" in effects:
        return "nonsyn"
    if "synonymous_variant" in effects:
        return "syn"
    return None


def main():
    args = parse_args()
    counts = {
        "variants": 0,
        "syn": 0,
        "nonsyn": 0,
        "ref_mismatch": 0,
        "coding_calls": 0,
    }
    with (
        open(args.syn_sites, "w") as syn_handle,
        open(args.nonsyn_sites, "w") as nonsyn_handle,
    ):
        for line in sys.stdin:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, _vid, ref, alt, _qual, _flt, info = fields[:8]
            if "," in alt or len(ref) != 1 or len(alt) != 1:
                continue
            counts["variants"] += 1
            ann = None
            for item in info.split(";"):
                if item.startswith("ANN="):
                    ann = item[4:]
                    break
            if not ann:
                continue
            effects, mismatch = effects_for_alt(ann, alt)
            label = classify(effects)
            if label is None:
                continue
            counts["coding_calls"] += 1
            if mismatch:
                counts["ref_mismatch"] += 1
                continue
            out_chrom = f"{args.chrom_prefix}{chrom}"
            handle = syn_handle if label == "syn" else nonsyn_handle
            handle.write(f"{out_chrom}\t{pos}\n")
            counts[label] += 1

    coding = counts["coding_calls"]
    mismatch_fraction = (counts["ref_mismatch"] / coding) if coding else 0.0
    with open(args.summary, "w") as handle:
        handle.write(
            "\t".join(
                [
                    "syn_snps",
                    "nonsyn_snps",
                    "ref_mismatch",
                    "coding_calls",
                    "variants_seen",
                ]
            )
            + "\n"
        )
        handle.write(
            "\t".join(
                str(counts[key])
                for key in (
                    "syn",
                    "nonsyn",
                    "ref_mismatch",
                    "coding_calls",
                    "variants",
                )
            )
            + "\n"
        )
    print(
        f"syn={counts['syn']} nonsyn={counts['nonsyn']} "
        f"ref_mismatch={counts['ref_mismatch']}/{coding}",
        file=sys.stderr,
    )
    if mismatch_fraction > args.max_mismatch_fraction:
        raise SystemExit(
            f"Reference mismatch fraction {mismatch_fraction:.4f} exceeds "
            f"{args.max_mismatch_fraction:.4f}. Annotation was not kept."
        )


if __name__ == "__main__":
    main()
