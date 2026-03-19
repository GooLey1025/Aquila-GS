import pandas as pd
from pathlib import Path

# 输入文件
input_file = "../1171rice.pheno"

# 输出目录
output_dir = Path("1171rice_phenotype_split")
output_dir.mkdir(exist_ok=True)

# 读取文件
df = pd.read_csv(input_file, sep="\t")

# 第一列作为样本列
sample_col = df.columns[0]

# 其余列都视为表型列
trait_cols = df.columns[1:]

for trait in trait_cols:
    out_df = pd.DataFrame({
        "samples": df[sample_col],
        trait: df[trait]
    })
    out_df.to_csv(output_dir / f"{trait}.tsv", sep="\t", index=False)

print(f"已输出 {len(trait_cols)} 个表型文件到: {output_dir}")
