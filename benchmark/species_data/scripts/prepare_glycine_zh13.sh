#!/usr/bin/env bash
# Build the SnpEff database for Zhonghuang 13 v2 (SoyBase Zh13.gnm2.LV9P).
# Soybean975 SNP coordinates match this assembly. Sequence names are rewritten
# to 1..20 so they match the VCF contigs.
set -euo pipefail

JAVA="${JAVA:-/usr/lib/jvm/java-21-openjdk-amd64/bin/java}"
SNPEFF_HOME="${SNPEFF_HOME:-/data4/gulei/snpEff}"
BASE="https://data.soybase.org/Glycine/max"
GDIR="${SNPEFF_HOME}/genomes/Glycine_max_ZH13v2"
DDIR="${SNPEFF_HOME}/data/Glycine_max_ZH13v2"
SRC="${SNPEFF_HOME}/genomes/soy_test"

mkdir -p "${GDIR}" "${DDIR}" "${SRC}"

if [[ ! -s "${SRC}/Zh13.gnm2.fna.gz" ]]; then
    curl -fL --retry 3 -o "${SRC}/Zh13.gnm2.fna.gz" \
        "${BASE}/genomes/Zh13.gnm2.LV9P/glyma.Zh13.gnm2.LV9P.genome_main.fna.gz"
fi
if [[ ! -s "${SRC}/Zh13.gnm2.gff3.gz" ]]; then
    curl -fL --retry 3 -o "${SRC}/Zh13.gnm2.gff3.gz" \
        "${BASE}/annotations/Zh13.gnm2.ann1.FJ3G/glyma.Zh13.gnm2.ann1.FJ3G.gene_models_main.gff3.gz"
fi
if [[ ! -s "${SRC}/Zh13.gnm2.cds.fna.gz" ]]; then
    curl -fL --retry 3 -o "${SRC}/Zh13.gnm2.cds.fna.gz" \
        "${BASE}/annotations/Zh13.gnm2.ann1.FJ3G/glyma.Zh13.gnm2.ann1.FJ3G.cds.fna.gz"
fi
if [[ ! -s "${SRC}/Zh13.gnm2.protein.faa.gz" ]]; then
    curl -fL --retry 3 -o "${SRC}/Zh13.gnm2.protein.faa.gz" \
        "${BASE}/annotations/Zh13.gnm2.ann1.FJ3G/glyma.Zh13.gnm2.ann1.FJ3G.protein.faa.gz"
fi

# glyma.Zh13.gnm2.Chr01 -> 1; scaffolds keep a short name.
rename_seq() {
    awk '
        function seqid(raw,   n) {
            n = raw
            sub(/^glyma\.Zh13\.gnm2\./, "", n)
            if (n ~ /^Chr[0-9]+$/) {
                sub(/^Chr0*/, "", n)
                return n
            }
            return n
        }
        /^>/ {
            raw = substr($1, 2)
            print ">" seqid(raw)
            next
        }
        { print }
    '
}

if [[ ! -s "${GDIR}/sequences.fa" ]]; then
    gzip -dc "${SRC}/Zh13.gnm2.fna.gz" | rename_seq > "${GDIR}/sequences.fa"
fi
if [[ ! -s "${GDIR}/genes.gff" ]]; then
    gzip -dc "${SRC}/Zh13.gnm2.gff3.gz" | awk -F '\t' '
        function seqid(raw,   n) {
            n = raw
            sub(/^glyma\.Zh13\.gnm2\./, "", n)
            if (n ~ /^Chr[0-9]+$/) {
                sub(/^Chr0*/, "", n)
                return n
            }
            return n
        }
        BEGIN { OFS = "\t" }
        /^#/ { print; next }
        NF < 8 { next }
        { $1 = seqid($1); print }
    ' > "${GDIR}/genes.gff"
fi
# SoyBase cds.fna is the spliced transcript, including UTRs. Build the
# coding sequence from the GFF so SnpEff's CDS check is meaningful.
# A GTF is what SnpEff reads reliably; the GFF's split CDS frames are not.
if [[ ! -s "${GDIR}/genes.gtf" ]]; then
    gffread "${GDIR}/genes.gff" -T -o "${GDIR}/genes.gtf"
fi
if [[ ! -s "${GDIR}/cds.fa" ]]; then
    gffread "${GDIR}/genes.gff" -g "${GDIR}/sequences.fa" \
        -x "${GDIR}/cds.fa" -y "${GDIR}/protein.gffread.fa"
fi
if [[ ! -s "${GDIR}/protein.fa" ]]; then
    gzip -dc "${SRC}/Zh13.gnm2.protein.faa.gz" > "${GDIR}/protein.fa"
fi

ln -sfn "${GDIR}/sequences.fa" "${DDIR}/sequences.fa"
rm -f "${DDIR}/genes.gff"
ln -sfn "${GDIR}/genes.gtf" "${DDIR}/genes.gtf"
ln -sfn "${GDIR}/cds.fa" "${DDIR}/cds.fa"
ln -sfn "${GDIR}/protein.fa" "${DDIR}/protein.fa"
if [[ ! -f "${GDIR}/sequences.fa.fai" ]]; then
    samtools faidx "${GDIR}/sequences.fa"
fi

if ! grep -q '^Glycine_max_ZH13v2.genome' "${SNPEFF_HOME}/snpEff.config"; then
    printf '\n# Zhonghuang 13 v2 / SoyBase Zh13.gnm2.LV9P\nGlycine_max_ZH13v2.genome : Glycine max\n' \
        >> "${SNPEFF_HOME}/snpEff.config"
fi

# About 80% of genomic CDS translations match the SoyBase proteins exactly.
# The rest differ mainly by in-frame stops that are in the genome sequence
# and absent from the protein file. Variant effects have to follow the
# genome, so the database is saved without failing that comparison.
if [[ ! -s "${DDIR}/snpEffectPredictor.bin" ]]; then
    find "${DDIR}" -maxdepth 1 -name '*.bin' -delete
    "${JAVA}" -Xmx32g -jar "${SNPEFF_HOME}/snpEff.jar" \
        build -gtf22 -v -noCheckCds -noCheckProtein \
        -c "${SNPEFF_HOME}/snpEff.config" \
        -dataDir "${SNPEFF_HOME}/data" \
        Glycine_max_ZH13v2
fi

echo "[INFO] Glycine_max_ZH13v2 ready: ${DDIR}"
