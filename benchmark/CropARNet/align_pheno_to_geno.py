#!/usr/bin/env python3
import sys

snp_tsv = sys.argv[1]
pheno_tsv = sys.argv[2]
out_tsv = sys.argv[3]

def read_order(path):
    order = []
    with open(path, "r") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            order.append(line.rstrip("\n").split("\t")[0])
    return order

def read_pheno(path):
    with open(path, "r") as f:
        header = f.readline().rstrip("\n")
        cols = header.split("\t")
        n_pheno = len(cols) - 1
        data = {}
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            # pad short rows
            vals = parts[1:] + ["NA"] * max(0, n_pheno - (len(parts) - 1))
            data[sid] = sid + "\t" + "\t".join(vals[:n_pheno])
    return header, n_pheno, data

order = read_order(snp_tsv)
pheno_header, n_pheno, ph = read_pheno(pheno_tsv)

with open(out_tsv, "w") as out:
    out.write(pheno_header + "\n")
    na_row = "\t".join(["NA"] * n_pheno)
    for sid in order:
        out.write(ph.get(sid, sid + "\t" + na_row) + "\n")