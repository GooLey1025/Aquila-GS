#!/usr/bin/env bash
# Build the SnpEff database for B73 RefGen_v3 (Ensembl Plants AGPv3.31).
# Maize1404 SNP coordinates match this assembly, not Ensembl NAM-5.0.
set -euo pipefail

JAVA="${JAVA:-/usr/lib/jvm/java-21-openjdk-amd64/bin/java}"
SNPEFF_HOME="${SNPEFF_HOME:-/data4/gulei/snpEff}"
BASE="https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-31"
GDIR="${SNPEFF_HOME}/genomes/Zea_mays_AGPv3"
DDIR="${SNPEFF_HOME}/data/Zea_mays_AGPv3"

mkdir -p "${GDIR}" "${DDIR}"

if [[ ! -s "${GDIR}/genes.gff" ]]; then
    curl -fL --retry 3 -o "${GDIR}/genes.gff.gz" \
        "${BASE}/gff3/zea_mays/Zea_mays.AGPv3.31.gff3.gz"
    gzip -dc "${GDIR}/genes.gff.gz" > "${GDIR}/genes.gff"
fi

if [[ ! -s "${GDIR}/cds.fa" ]]; then
    curl -fL --retry 3 -o "${GDIR}/cds.fa.gz" \
        "${BASE}/fasta/zea_mays/cds/Zea_mays.AGPv3.31.cds.all.fa.gz"
    gzip -dc "${GDIR}/cds.fa.gz" > "${GDIR}/cds.fa"
fi
if [[ ! -s "${GDIR}/protein.fa" ]]; then
    curl -fL --retry 3 -o "${GDIR}/protein.fa.gz" \
        "${BASE}/fasta/zea_mays/pep/Zea_mays.AGPv3.31.pep.all.fa.gz"
    gzip -dc "${GDIR}/protein.fa.gz" > "${GDIR}/protein.fa"
fi

if [[ ! -s "${GDIR}/sequences.fa" ]]; then
    : > "${GDIR}/sequences.fa"
    for i in 1 2 3 4 5 6 7 8 9 10; do
        fasta="Zea_mays.AGPv3.31.dna.chromosome.${i}.fa.gz"
        if [[ ! -s "${GDIR}/${fasta}" ]]; then
            curl -fL --retry 3 -o "${GDIR}/${fasta}" \
                "${BASE}/fasta/zea_mays/dna/${fasta}"
        fi
        gzip -dc "${GDIR}/${fasta}" >> "${GDIR}/sequences.fa"
    done
fi

# Ensembl uses feature type "transcript" and ID=transcript:NAME, while the
# CDS FASTA uses NAME. SnpEff only treats mRNA as a transcript and matches
# the FASTA header to that ID. Protein headers use a _P01 accession.
if [[ ! -s "${GDIR}/genes.snpeff.gff" ]]; then
    awk 'BEGIN{OFS="\t"}
        $3 == "transcript" { $3 = "mRNA" }
        {
            gsub(/ID=transcript:/, "ID=")
            gsub(/Parent=transcript:/, "Parent=")
            print
        }' "${GDIR}/genes.gff" > "${GDIR}/genes.snpeff.gff"
fi
if [[ ! -s "${GDIR}/protein.snpeff.fa" ]]; then
    awk '
        /^>/ {
            if (match($0, /transcript:([^[:space:]]+)/, m)) print ">" m[1]
            else print
            next
        }
        { print }
    ' "${GDIR}/protein.fa" > "${GDIR}/protein.snpeff.fa"
fi

ln -sfn "${GDIR}/sequences.fa" "${DDIR}/sequences.fa"
ln -sfn "${GDIR}/genes.snpeff.gff" "${DDIR}/genes.gff"
ln -sfn "${GDIR}/cds.fa" "${DDIR}/cds.fa"
ln -sfn "${GDIR}/protein.snpeff.fa" "${DDIR}/protein.fa"
if [[ ! -f "${GDIR}/sequences.fa.fai" ]]; then
    samtools faidx "${GDIR}/sequences.fa"
fi

if ! grep -q '^Zea_mays_AGPv3.genome' "${SNPEFF_HOME}/snpEff.config"; then
    printf '\n# B73 RefGen_v3 / Ensembl AGPv3.31\nZea_mays_AGPv3.genome : Zea_mays\n' \
        >> "${SNPEFF_HOME}/snpEff.config"
fi

if [[ ! -s "${DDIR}/snpEffectPredictor.bin" ]]; then
    find "${DDIR}" -maxdepth 1 -name '*.bin' -delete
    "${JAVA}" -Xmx32g -jar "${SNPEFF_HOME}/snpEff.jar" \
        build -gff3 -v \
        -c "${SNPEFF_HOME}/snpEff.config" \
        -dataDir "${SNPEFF_HOME}/data" \
        Zea_mays_AGPv3
fi

echo "[INFO] Zea_mays_AGPv3 ready: ${DDIR}"
