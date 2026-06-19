for config in params/*.yaml; do
    name=$(basename "$config" .yaml)
    echo "Running $config ..."
    aquila_train_multi.py \
        --config "$config" \
        --n-folds 5 \
        --n-seeds 5 \
        -o "$name"
done

params/705rice_conv_mha.aquila-snp.genotype-class_encoding.yaml
