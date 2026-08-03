# MENET Benchmark Adaptation

The benchmark adaptation preserves MENET's two-stage trait-specific encoder, relatedness representation, and VE-RepGeno fusion. The changes provide a leakage-safe and reproducible evaluation under the nested cross-validation protocol shared by all benchmark models.

| Aspect | Original MENET workflow | Benchmark adaptation | Purpose |
|---|---|---|---|
| Data splitting | One predefined train/validation/test split | Fixed outer and inner cross-validation folds shared with all benchmark models | Ensure directly comparable evaluation |
| Hyperparameter optimization | Uses a fixed configuration without a systematic search in the provided workflow | Performs a 32-point grid search within the inner CV, covering encoder learning rate and embedding dimension plus MENET learning rate, batch size, and dropout | Tune both training stages without accessing the outer test fold |
| Hyperparameter selection criterion | Not applicable to the fixed-configuration workflow | Selects the candidate with the highest mean validation Pearson correlation across inner folds | Align model tuning with the primary genomic-prediction metric |
| Best MENET epoch | Selects the epoch with the highest validation R² on one validation split | Selects the highest validation Pearson epoch independently in each inner fold | Use the benchmark's primary metric and keep epoch selection fold-local |
| Final outer-training epoch | Not defined through nested-CV aggregation | Uses the half-up rounded median of the selected candidate's best MENET epochs across inner folds | Determine training duration without using outer-test performance |
| Final encoder epoch | Selects the lowest validation triplet-loss checkpoint on one validation split | Uses the half-up rounded median of the best encoder epochs across inner folds | Refit the encoder on complete outer-training data without an outer-test stopping signal |
| Final model fitting | Uses the model selected from the original split | Retrains both the trait-specific encoder and MENET from scratch on the complete outer-training fold | Keep the final test evaluation independent |
| Trait-specific encoder | Trained once for each phenotype | Trained independently within every inner-training fold and refitted on each outer-training fold | Prevent validation or test phenotypes from influencing the learned embedding |
| Relatedness reference | Constructed as a full matrix containing train, validation, and test individuals | Validation and test individuals are compared only with a training-only reference bank | Prevent held-out individuals from defining model inputs |
| Relatedness normalization | Maximum distance is calculated from the full cohort | Distance scale is estimated from training embeddings only and reused for held-out samples | Prevent held-out genotypes from influencing feature normalization |
| Triplet construction | Candidate samples are randomly resampled during data access | Triplets are fixed for each fold and random seed | Improve reproducibility and stabilize encoder selection |
| Missing phenotypes | No explicit split-safe handling | Individuals missing the selected trait are removed independently from each split | Ensure missing-value sentinels never enter training or evaluation |
| Phenotype preprocessing | Managed separately from the model workflow | Reuses transformations fitted only on the corresponding inner- or outer-training samples | Prevent preprocessing leakage |
| Multiple traits | Separate manual run for each trait | Accepts multiple trait names but trains an independent MENET pipeline for each trait | Preserve MENET's single-trait assumption while standardizing execution |
| Test evaluation | Reports results after the predefined training procedure | Evaluates each outer-test fold once, after all tuning and refitting are complete | Provide an unbiased generalization estimate |
| Outputs | Trait encoder and relatedness are saved, but the final MENET model is not fully packaged | Saves both model stages, training-only reference metadata, selected configuration, predictions, metrics, and sample audits | Improve reproducibility and traceability |
| Integrated Gradients | Computed during every training epoch | Removed from model selection and training; it can be performed after training if required | Avoid unnecessary cost and prevent test interpretation from guiding model development |
