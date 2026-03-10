#!/usr/bin/env python3
"""
Batch generate parameter files for each phenotype.
Template: 705rice_conv_mha.aquila-vars.hpo.yaml
Output: 705rice_conv_mha.aquila-vars.{Pheno}.hpo.yaml
"""

import os
import re
import glob

# Paths
TEMPLATE_FILE = "/home/gulei/projects/Aquila-GS/benchmark/aquila/params/705rice_conv_mha.aquila-vars.hpo.yaml"
PHENOTYPES_DIR = "/home/gulei/projects/Aquila-GS/benchmark/aquila/phenotypes"
OUTPUT_DIR = "/home/gulei/projects/Aquila-GS/benchmark/aquila/params"

# Extract template filename prefix (e.g., "705rice_conv_mha.aquila-vars" from "705rice_conv_mha.aquila-vars.hpo.yaml")
template_basename = os.path.basename(TEMPLATE_FILE)
template_prefix = re.sub(r'\.hpo\.yaml$', '', template_basename)

# Read template file
with open(TEMPLATE_FILE, 'r') as f:
    template_content = f.read()

# Extract original pheno_file value from template content (handle inline comments and indentation)
pheno_match = re.search(r'^\s*pheno_file:\s*(\S+)', template_content, re.MULTILINE)
if pheno_match:
    original_pheno_file = pheno_match.group(1).strip()
else:
    raise ValueError(f"Could not find 'pheno_file' in template: {TEMPLATE_FILE}")

print(f"Original pheno_file in template: {original_pheno_file}")

# Get all phenotype files
phenotype_files = glob.glob(os.path.join(PHENOTYPES_DIR, "*.tsv"))
phenotype_files.sort()

print(f"Found {len(phenotype_files)} phenotype files")
print(f"Using template: {TEMPLATE_FILE}")
print(f"Template prefix: {template_prefix}")
print(f"Output directory: {OUTPUT_DIR}")
print("-" * 60)

# Generate parameter file for each phenotype
generated_count = 0
for pheno_file in phenotype_files:
    # Get phenotype name (e.g., GYP_BLUP from GYP_BLUP.tsv)
    pheno_name = os.path.splitext(os.path.basename(pheno_file))[0]
    
    # Create new parameter content (replace original pheno_file with phenotype-specific one)
    new_content = template_content.replace(
        f"pheno_file: {original_pheno_file}",
        f"pheno_file: phenotypes/{pheno_name}.tsv"
    )
    
    # Generate output filename using template prefix
    output_filename = f"{template_prefix}.{pheno_name}.hpo.yaml"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Write the new parameter file
    with open(output_path, 'w') as f:
        f.write(new_content)
    
    print(f"Generated: {output_filename}")
    generated_count += 1

print("-" * 60)
print(f"Successfully generated {generated_count} parameter files!")
