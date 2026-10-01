#!/usr/bin/env bash
# Annotate a genome-wide VCF with SnpEff and write synonymous and missense VCFs.
# Per-class LD pruning is optional; the species main.sh merges both classes first.
set -euo pipefail

usage() {
    echo "Usage: $0 --vcf FILE --genome NAME --prefix PREFIX [options]" >&2
    echo "  --fasta FILE          Reference FASTA. REF/ALT are flipped to match it before annotation." >&2
    echo "  --strip-prefix STR    Remove this chromosome prefix before SnpEff and restore it in site lists." >&2
    echo "  --skip-ld             Write class VCFs only. Do not LD-prune each class." >&2
    echo "  --ld-window 1000kb --ld-step 1 --ld-r2 0.1" >&2
    exit 2
}

VCF=""
GENOME=""
PREFIX=""
FASTA=""
STRIP_PREFIX=""
LD_WINDOW="1000kb"
LD_STEP="1"
LD_R2="0.1"
SKIP_LD=0
THREADS="${THREADS:-16}"
JAVA="${JAVA:-/usr/lib/jvm/java-21-openjdk-amd64/bin/java}"
SNPEFF_HOME="${SNPEFF_HOME:-/data4/gulei/snpEff}"
JAVA_MEM="${JAVA_MEM:-80g}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vcf) VCF="$2"; shift 2 ;;
        --genome) GENOME="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --fasta) FASTA="$2"; shift 2 ;;
        --strip-prefix) STRIP_PREFIX="$2"; shift 2 ;;
        --ld-window) LD_WINDOW="$2"; shift 2 ;;
        --ld-step) LD_STEP="$2"; shift 2 ;;
        --ld-r2) LD_R2="$2"; shift 2 ;;
        --skip-ld) SKIP_LD=1; shift ;;
        --java) JAVA="$2"; shift 2 ;;
        --snpeff-home) SNPEFF_HOME="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "${VCF}" || -z "${GENOME}" || -z "${PREFIX}" ]]; then
    usage
fi
if [[ ! -s "${VCF}" ]]; then
    echo "[ERROR] Missing VCF: ${VCF}" >&2
    exit 1
fi
if [[ ! -s "${SNPEFF_HOME}/data/${GENOME}/snpEffectPredictor.bin" ]]; then
    echo "[ERROR] SnpEff database not built: ${SNPEFF_HOME}/data/${GENOME}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(cd "$(dirname "${PREFIX}")" && pwd)"
BASE="$(basename "${PREFIX}")"
WORKDIR="${OUT_DIR}/${BASE}.work"
mkdir -p "${WORKDIR}"

SUMMARY="${PREFIX}.summary.tsv"
if [[ -s "${SUMMARY}" ]]; then
    echo "[INFO] Using existing summary: ${SUMMARY}"
    cat "${SUMMARY}"
    exit 0
fi

CHR_MAP="${WORKDIR}/chrom.map"
: > "${CHR_MAP}"
if [[ -n "${STRIP_PREFIX}" ]]; then
    bcftools view -h "${VCF}" \
        | awk -v prefix="${STRIP_PREFIX}" -F'[=,>]' '
            /^##contig=/ {
                id = $3
                if (index(id, prefix) == 1) {
                    print id "\t" substr(id, length(prefix) + 1)
                }
            }
        ' > "${CHR_MAP}"
    if [[ ! -s "${CHR_MAP}" ]]; then
        echo "[ERROR] No contigs start with prefix ${STRIP_PREFIX}" >&2
        exit 1
    fi
fi

SITES_VCF="${WORKDIR}/sites.vcf.gz"
echo "[INFO] Preparing sites for ${GENOME}"
if [[ -n "${FASTA}" ]]; then
    if [[ ! -f "${FASTA}.fai" ]]; then
        samtools faidx "${FASTA}"
    fi
    bcftools view -G -Ou --threads "${THREADS}" "${VCF}" \
        | bcftools +fixref -Ou -- -f "${FASTA}" -m flip-all \
        | bcftools view -Oz -o "${SITES_VCF}" --threads "${THREADS}"
else
    bcftools view -G -Oz -o "${SITES_VCF}" --threads "${THREADS}" "${VCF}"
fi
bcftools index -f "${SITES_VCF}"

ANNOT_IN="${SITES_VCF}"
if [[ -s "${CHR_MAP}" ]]; then
    ANNOT_IN="${WORKDIR}/sites.renamed.vcf.gz"
    bcftools annotate --rename-chrs "${CHR_MAP}" -Oz -o "${ANNOT_IN}" --threads "${THREADS}" "${SITES_VCF}"
    bcftools index -f "${ANNOT_IN}"
fi

SYN_SITES="${WORKDIR}/syn.sites"
NONSYN_SITES="${WORKDIR}/nonsyn.sites"
ANN_SUMMARY="${WORKDIR}/ann.tsv"
echo "[INFO] Running SnpEff ${GENOME}"
"${JAVA}" -Xmx"${JAVA_MEM}" -jar "${SNPEFF_HOME}/snpEff.jar" \
    -c "${SNPEFF_HOME}/snpEff.config" \
    -dataDir "${SNPEFF_HOME}/data" \
    -nodownload -noLog -noStats -noMotif -noNextProt \
    -canon -onlyProtein \
    "${GENOME}" \
    "${ANNOT_IN}" \
    2> "${WORKDIR}/snpeff.err" \
    | python3 "${SCRIPT_DIR}/classify_snpeff_ann.py" \
        --syn-sites "${SYN_SITES}" \
        --nonsyn-sites "${NONSYN_SITES}" \
        --summary "${ANN_SUMMARY}" \
        --chrom-prefix "${STRIP_PREFIX}"

prune_count() {
    local sites="$1"
    local label="$2"
    local vcf_out="${PREFIX}.${label}.vcf.gz"
    local ld_prefix="${PREFIX}.${label}.ld"
    local n
    n="$(wc -l < "${sites}")"
    if [[ "${n}" -eq 0 ]]; then
        echo 0
        return
    fi
    bcftools view -T "${sites}" -Oz -o "${vcf_out}" --threads "${THREADS}" "${VCF}"
    bcftools index -f "${vcf_out}"
    plink2 \
        --vcf "${vcf_out}" \
        --indep-pairwise "${LD_WINDOW}" "${LD_STEP}" "${LD_R2}" \
        --threads "${THREADS}" \
        --memory 100000 \
        --out "${ld_prefix}" \
        > "${ld_prefix}.stdout"
    awk 'END { print NR }' "${ld_prefix}.prune.in"
}

write_class_vcf() {
    local sites="$1"
    local label="$2"
    local vcf_out="${PREFIX}.${label}.vcf.gz"
    local n
    n="$(wc -l < "${sites}")"
    if [[ "${n}" -eq 0 ]]; then
        echo "[ERROR] No ${label} sites in ${sites}" >&2
        exit 1
    fi
    if [[ ! -s "${vcf_out}" ]]; then
        bcftools view -T "${sites}" -Oz -o "${vcf_out}" --threads "${THREADS}" "${VCF}"
        bcftools index -f "${vcf_out}"
    fi
}

if [[ "${SKIP_LD}" -eq 1 ]]; then
    write_class_vcf "${SYN_SITES}" syn
    write_class_vcf "${NONSYN_SITES}" nonsyn
    echo "[INFO] Wrote synonymous and missense VCFs without per-class LD"
    exit 0
fi

echo "[INFO] LD pruning r2=${LD_R2} window=${LD_WINDOW} step=${LD_STEP}"
SYN_LD="$(prune_count "${SYN_SITES}" syn)"
NONSYN_LD="$(prune_count "${NONSYN_SITES}" nonsyn)"

{
    printf "genome\tvcf\tld_window\tld_step\tld_r2\tsyn_snps\tnonsyn_snps\tsyn_ld\tnonsyn_ld\tref_mismatch\tcoding_calls\n"
    awk -v genome="${GENOME}" -v vcf="${VCF}" \
        -v window="${LD_WINDOW}" -v step="${LD_STEP}" -v r2="${LD_R2}" \
        -v syn_ld="${SYN_LD}" -v nonsyn_ld="${NONSYN_LD}" \
        'NR == 2 { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", genome, vcf, window, step, r2, $1, $2, syn_ld, nonsyn_ld, $3, $4 }' \
        "${ANN_SUMMARY}"
} > "${SUMMARY}"

echo "[INFO] Wrote ${SUMMARY}"
cat "${SUMMARY}"
