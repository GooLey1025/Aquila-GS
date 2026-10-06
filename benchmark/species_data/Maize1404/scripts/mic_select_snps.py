#!/usr/bin/env python3
"""Select SNPs by maximal information coefficient (MIC) against phenotypes.

Scores each SNP with minepy ``cstats`` (default estimator ``mic_approx``,
``alpha=0.6``, ``c=15``). Samples are matched to the phenotype ``LINE`` column
after stripping a ``CUBIC_`` / ``0_`` prefix. Missing trait values are dropped
per trait. A SNP's score is the maximum MIC across traits. The top ``--n-snps``
markers are written to a bgzipped VCF, in original coordinate order.
"""

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True, help="Imputed SNP VCF (.vcf.gz)")
    parser.add_argument("--pheno", required=True, help="Phenotype TSV with a LINE column")
    parser.add_argument("--vcf-out", required=True, help="Output VCF of selected SNPs (.vcf.gz)")
    parser.add_argument("--scores-out", required=True, help="TSV of selected SNPs and MIC scores")
    parser.add_argument("--n-snps", type=int, default=10000, help="Number of SNPs to keep")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--c", type=float, default=15)
    parser.add_argument("--est", default="mic_approx", choices=("mic_approx", "mic_e"))
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Skip a trait with fewer non-missing aligned samples than this",
    )
    parser.add_argument(
        "--sample-map",
        choices=("maize", "soybean", "tomato"),
        default="maize",
        help="How VCF sample IDs map onto phenotype LINE IDs",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory of per-chunk MIC scores; reused on restart",
    )
    return parser.parse_args()


_SL_ID = re.compile(r"^SL(\d+)$")


def to_pheno_id(sample_id: str, sample_map: str) -> str:
    if sample_map == "maize":
        name = sample_id
        if name.startswith("0_"):
            name = name[2:]
        if name.startswith("CUBIC_"):
            name = name[len("CUBIC_") :]
        return name
    if sample_map == "soybean":
        return sample_id.split("_", 1)[0]
    match = _SL_ID.match(sample_id)
    if match is None:
        return sample_id
    return f"TS-{int(match.group(1))}"


def load_traits(pheno_path, vcf_samples, min_samples, sample_map):
    pheno = pd.read_csv(pheno_path, sep="\t")
    if "LINE" not in pheno.columns:
        raise SystemExit(f"Missing LINE column in {pheno_path}")
    pheno["LINE"] = pheno["LINE"].astype(str)
    pheno = pheno.drop_duplicates("LINE", keep="first").set_index("LINE")
    trait_names = [c for c in pheno.columns if c != "LINE"]
    if not trait_names:
        raise SystemExit(f"No trait columns in {pheno_path}")

    aligned = []
    seen = set()
    for sample in vcf_samples:
        key = to_pheno_id(sample, sample_map)
        if key not in pheno.index:
            continue
        if key in seen:
            raise SystemExit(f"Duplicate phenotype ID after sample-map {sample_map}: {sample} -> {key}")
        seen.add(key)
        aligned.append(sample)
    if not aligned:
        raise SystemExit(
            f"No VCF samples match phenotype LINE IDs with --sample-map {sample_map}."
        )

    pheno = pheno.loc[[to_pheno_id(s, sample_map) for s in aligned]]
    traits = []
    masks = []
    used = []
    for name in trait_names:
        values = pd.to_numeric(pheno[name], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(values)
        if int(mask.sum()) < min_samples:
            print(
                f"[WARN] skip {name}: {int(mask.sum())} non-missing samples",
                file=sys.stderr,
            )
            continue
        if np.nanstd(values) == 0:
            print(f"[WARN] skip {name}: constant phenotype", file=sys.stderr)
            continue
        traits.append(values)
        masks.append(mask)
        used.append(name)
    if not used:
        raise SystemExit("No traits left after missing-value and sample-size filters.")
    print(
        f"[INFO] aligned samples: {len(aligned)} / {len(vcf_samples)}; traits: {len(used)}",
        file=sys.stderr,
    )
    return aligned, traits, masks, used


def iter_dosage_chunks(vcf_path, samples, chunk_size):
    import cyvcf2

    vcf = cyvcf2.VCF(vcf_path, samples=samples)
    n = len(samples)
    buf = np.empty((chunk_size, n), dtype=np.float64)
    meta = []
    i = 0
    seen = 0
    for variant in vcf:
        gt = np.asarray(variant.gt_types)
        dosage = np.full(gt.shape[0], np.nan, dtype=np.float64)
        dosage[gt == 0] = 0.0
        dosage[gt == 1] = 1.0
        dosage[gt == 3] = 2.0
        buf[i] = dosage
        vid = variant.ID if variant.ID not in (None, ".") else f"{variant.CHROM}:{variant.POS}"
        meta.append((variant.CHROM, int(variant.POS), vid))
        i += 1
        seen += 1
        if i == chunk_size:
            yield seen - i, buf.copy(), meta
            meta = []
            i = 0
    if i:
        yield seen - i, buf[:i].copy(), meta
    vcf.close()


def _prepare_geno(geno, mask):
    x = geno[:, mask]
    if np.isnan(x).any():
        x = x.copy()
        with np.errstate(all="ignore"):
            row_mean = np.nanmean(x, axis=1)
        row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
        rows, cols = np.where(np.isnan(x))
        x[rows, cols] = row_mean[rows]
    return x


def _mic_max(geno, traits, masks, alpha, c, est):
    from minepy import cstats

    scores = np.zeros(geno.shape[0], dtype=np.float64)
    geno = np.ascontiguousarray(geno, dtype=np.float64)
    groups = {}
    for trait_i, mask in enumerate(masks):
        groups.setdefault(mask.tobytes(), []).append(trait_i)
    for trait_ids in groups.values():
        mask = masks[trait_ids[0]]
        x = _prepare_geno(geno, mask)
        keep = x.std(axis=1) > 0
        if not np.any(keep):
            continue
        y_rows = []
        for trait_i in trait_ids:
            trait = np.ascontiguousarray(traits[trait_i][mask], dtype=np.float64)
            if float(trait.std()) == 0:
                continue
            y_rows.append(trait)
        if not y_rows:
            continue
        y = np.vstack(y_rows)
        mic, _tic = cstats(
            np.ascontiguousarray(x[keep]),
            np.ascontiguousarray(y),
            alpha=alpha,
            c=c,
            est=est,
        )
        mic = np.nan_to_num(mic, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.flatnonzero(keep)
        scores[idx] = np.maximum(scores[idx], mic.max(axis=1))
    return scores


def _save_chunk(path, meta, scores):
    np.savez(
        path,
        chrom=np.array([row[0] for row in meta]),
        pos=np.array([row[1] for row in meta], dtype=np.int64),
        vid=np.array([row[2] for row in meta]),
        scores=np.asarray(scores, dtype=np.float64),
    )


def _load_chunk(path):
    with np.load(path, allow_pickle=False) as stored:
        meta = list(
            zip(
                stored["chrom"].tolist(),
                stored["pos"].astype(int).tolist(),
                stored["vid"].tolist(),
            )
        )
        scores = stored["scores"].copy()
    return meta, scores


def _coord_key(row):
    chrom = str(row[0])
    chrom_key = (0, int(chrom)) if chrom.isdigit() else (1, chrom)
    return (chrom_key, row[1], row[2])


def select_top(meta, scores, n_snps):
    n = len(meta)
    if n_snps >= n:
        order = np.arange(n)
    else:
        # argpartition picks the top scores; equal MIC keeps earlier variants.
        part = np.argpartition(scores, n - n_snps)[n - n_snps :]
        order = part[np.argsort(scores[part], kind="mergesort")[::-1]]
        order = order[:n_snps]
    score_by_id = {}
    selected = []
    for i in order:
        chrom, pos, vid = meta[i]
        if vid in score_by_id:
            raise SystemExit(f"Duplicate variant ID in MIC selection: {vid}")
        score_by_id[vid] = float(scores[i])
        selected.append((chrom, pos, vid))
    selected.sort(key=_coord_key)
    selected_scores = np.array([score_by_id[row[2]] for row in selected], dtype=np.float64)
    return selected, selected_scores


def write_vcf(vcf_in, vcf_out, selected, threads):
    id_path = Path(str(vcf_out) + ".ids.txt")
    id_path.write_text("".join(f"{row[2]}\n" for row in selected))
    subprocess.check_call(
        [
            "bcftools",
            "view",
            "--threads",
            str(threads),
            "-i",
            f"ID=@{id_path}",
            "-Oz",
            "-o",
            str(vcf_out),
            str(vcf_in),
        ]
    )
    subprocess.check_call(
        ["bcftools", "index", "--threads", str(threads), "--force", str(vcf_out)]
    )
    id_path.unlink(missing_ok=True)


def main():
    args = parse_args()
    if args.n_snps < 1:
        raise SystemExit("--n-snps must be >= 1")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    import cyvcf2

    vcf_samples = cyvcf2.VCF(args.vcf).samples
    aligned, traits, masks, trait_names = load_traits(
        args.pheno, vcf_samples, args.min_samples, args.sample_map
    )
    print(f"[INFO] traits used ({len(trait_names)}): {', '.join(trait_names)}", file=sys.stderr)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = {}
    n_chunks = 0
    n_cached = 0
    started = time.time()

    def consume(done):
        for fut in done:
            chunk_id, chunk_meta = pending.pop(fut)
            _save_chunk(cache_dir / f"{chunk_id:06d}.npz", chunk_meta, fut.result())

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        for chunk_id, (start, geno, chunk_meta) in enumerate(
            iter_dosage_chunks(args.vcf, aligned, args.chunk_size)
        ):
            del start
            n_chunks = chunk_id + 1
            cache_path = cache_dir / f"{chunk_id:06d}.npz"
            if cache_path.exists():
                n_cached += 1
                continue
            fut = pool.submit(
                _mic_max, geno, traits, masks, args.alpha, args.c, args.est
            )
            pending[fut] = (chunk_id, chunk_meta)
            if len(pending) >= args.threads:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                consume(done)
            if n_chunks % 10 == 0:
                n_done = n_cached + (n_chunks - n_cached - len(pending))
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[INFO] chunks {n_chunks}; cached {n_cached}; "
                    f"scored-or-cached ~{n_done * args.chunk_size} SNPs; "
                    f"elapsed {elapsed / 60:.1f} min",
                    file=sys.stderr,
                )
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            consume(done)

    meta = []
    score_chunks = []
    for chunk_id in range(n_chunks):
        chunk_meta, chunk_scores = _load_chunk(cache_dir / f"{chunk_id:06d}.npz")
        meta.extend(chunk_meta)
        score_chunks.append(chunk_scores)
    scores = np.concatenate(score_chunks) if score_chunks else np.empty(0, dtype=np.float64)
    if not meta:
        raise SystemExit(f"No variants read from {args.vcf}")

    print(f"[INFO] scored {len(meta)} SNPs", file=sys.stderr)
    selected, selected_scores = select_top(meta, scores, args.n_snps)
    score_table = pd.DataFrame(
        {
            "CHROM": [row[0] for row in selected],
            "POS": [row[1] for row in selected],
            "ID": [row[2] for row in selected],
            "MIC": selected_scores,
        }
    )
    score_table.to_csv(args.scores_out, sep="\t", index=False)
    write_vcf(args.vcf, args.vcf_out, selected, args.threads)
    print(
        f"[INFO] kept {len(selected)} / {len(meta)} SNPs "
        f"(MIC {selected_scores.min():.4f} .. {selected_scores.max():.4f})",
        file=sys.stderr,
    )
    print(f"[INFO] wrote {args.vcf_out} and {args.scores_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
