#!/usr/bin/env Rscript

# rrBLUP Cross-Population Prediction Script
# Loads the full training set, trains rrBLUP, predicts on the 120-inbred-line population
# Then saves predictions for correlation analysis

suppressPackageStartupMessages({
  library(data.table)
  library(vcfR)
  library(rrBLUP)
  library(optparse)
})

parse_args <- function() {
  option_list <- list(
    make_option(c("--train-vcf"), type = "character",
                default = "705rice_0.03.full.all.impute.biallelic.vcf.gz",
                help = "Training VCF file (gzipped)", metavar = "character"),
    make_option(c("--test-vcf"), type = "character",
                default = "120_inbred_line.snp.indel.sv.impute.biallelic.vcf.gz",
                help = "Test (120 inbred lines) VCF file", metavar = "character"),
    make_option(c("--pheno"), type = "character",
                default = "GSTP008.pheno.tsv",
                help = "Training phenotype TSV file", metavar = "character"),
    make_option(c("--split"), type = "character",
                default = "data_split.tsv",
                help = "Data split TSV (used to get training sample IDs)", metavar = "character"),
    make_option(c("--output-dir"), type = "character",
                default = "rrblup_output",
                help = "Output directory [default=%default]", metavar = "character"),
    make_option(c("--seed"), type = "integer", default = 42,
                help = "Random seed [default=%default]", metavar = "integer")
  )

  opt_parser <- OptionParser(option_list = option_list,
                            description = "rrBLUP cross-population prediction on 120 inbred lines")
  opt <- optparse::parse_args(opt_parser)

  if (!file.exists(opt$`train-vcf`))
    stop(sprintf("Training VCF not found: %s", opt$`train-vcf`))
  if (!file.exists(opt$`test-vcf`))
    stop(sprintf("Test VCF not found: %s", opt$`test-vcf`))
  if (!file.exists(opt$pheno))
    stop(sprintf("Phenotype file not found: %s", opt$pheno))

  return(opt)
}

load_vcf_to_matrix <- function(vcf_file, restrict_samples = NULL) {
  cat(sprintf("[VCF] Loading: %s\n", vcf_file))
  vcf <- read.vcfR(vcf_file, verbose = FALSE)
  cat(sprintf("  %d variants x %d samples\n", nrow(vcf@fix), ncol(vcf@gt) - 1))

  # Keep only SNP variants (ID column starts with "SNP")
  snp_mask <- grepl("^SNP", vcf@fix[, "ID"])
  n_snp <- sum(snp_mask)
  cat(sprintf("  Filtering to SNP-only: %d / %d variants retained\n",
              n_snp, nrow(vcf@fix)))
  vcf <- vcf[snp_mask, ]

  if (any(duplicated(vcf@fix[, "ID"]))) {
    vcf@fix[, "ID"] <- paste0("v", seq_len(nrow(vcf@fix)))
  }

  geno <- extract.gt(vcf, element = "GT")
  colnames(geno) <- colnames(vcf@gt)[-1]
  rownames(geno) <- paste0("v", seq_len(nrow(geno)))

  cat("  Converting genotypes (-1/0/1)...\n")
  geno[geno %in% c("0/0", "0|0")] <- -1
  geno[geno %in% c("0/1", "1/0", "0|1", "1|0")] <- 0
  geno[geno %in% c("1/1", "1|1")] <- 1
  geno[geno %in% c("./.", ".|.")] <- NA
  storage.mode(geno) <- "numeric"

  n_miss <- sum(is.na(geno))
  if (n_miss > 0) {
    cat(sprintf("  Imputing %d NAs with row means\n", n_miss))
    rmeans <- rowMeans(geno, na.rm = TRUE)
    for (i in seq_len(nrow(geno)))
      geno[i, is.na(geno[i, ])] <- rmeans[i]
  }

  if (!is.null(restrict_samples)) {
    matched <- match(restrict_samples, colnames(geno))
    geno <- geno[, matched, drop = FALSE]
    colnames(geno) <- restrict_samples
  }

  return(t(geno))  # samples x variants
}

main <- function() {
  args <- parse_args()
  set.seed(args$seed)

  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("rrBLUP Cross-Population Prediction\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")

  output_dir <- args$`output-dir`
  if (!dir.exists(output_dir))
    dir.create(output_dir, recursive = TRUE)

  # -- Get training sample IDs from split file --------------------
  cat("[SPLIT] Reading data_split.tsv...\n")
  split_df <- fread(args$split, sep = "\t", header = TRUE, data.table = FALSE)
  train_ids <- split_df$Sample_ID[split_df$Split == "train"]
  cat(sprintf("  Training samples: %d\n\n", length(train_ids)))

  # -- Load training genotype --------------------------------------
  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("Loading Training Genotype\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")
  X_train_raw <- load_vcf_to_matrix(args$`train-vcf`)

  # Keep only training samples present in VCF
  train_ids_in_vcf <- intersect(train_ids, rownames(X_train_raw))
  cat(sprintf("  Training samples found in VCF: %d\n", length(train_ids_in_vcf)))

  X_train_full <- X_train_raw[train_ids_in_vcf, , drop = FALSE]
  cat(sprintf("  Training matrix: %d samples x %d variants\n\n",
              nrow(X_train_full), ncol(X_train_full)))

  # -- Load training phenotype ------------------------------------
  cat("[PHENO] Loading training phenotypes...\n")
  pheno_raw <- fread(args$pheno, sep = "\t", header = TRUE, data.table = FALSE)
  colnames(pheno_raw)[1] <- "Sample_ID"

  # Target BLUP traits
  target_traits <- c("PH_BLUP", "HD_BLUP", "GYP_BLUP")
  blup_traits <- intersect(target_traits, colnames(pheno_raw))
  if (length(blup_traits) == 0)
    stop("None of PH_BLUP, HD_BLUP, GYP_BLUP found in phenotype file")
  cat(sprintf("  BLUP target traits: %s\n\n", paste(blup_traits, collapse = ", ")))

  # Align phenotype to genotype samples
  pheno_aligned <- pheno_raw[match(train_ids_in_vcf, pheno_raw$Sample_ID), ,
                             drop = FALSE]
  rownames(pheno_aligned) <- pheno_aligned$Sample_ID

  # -- Load test genotype -----------------------------------------
  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("Loading Test Genotype (120 Inbred Lines)\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")
  X_test_raw <- load_vcf_to_matrix(args$`test-vcf`)
  cat(sprintf("  Test matrix: %d samples x %d variants\n",
              nrow(X_test_raw), ncol(X_test_raw)))

  # -- Align test genotype to training genotype (shared variants) --
  common_vars <- intersect(colnames(X_train_full), colnames(X_test_raw))
  cat(sprintf("  Common variants: %d / train:%d / test:%d\n",
              length(common_vars),
              ncol(X_train_full),
              ncol(X_test_raw)))

  if (length(common_vars) == 0)
    stop("No common variants between training and test VCF!")

  X_train <- X_train_full[, common_vars, drop = FALSE]
  X_test  <- X_test_raw[,  common_vars, drop = FALSE]

  # Make sure test sample order matches
  test_sample_ids <- rownames(X_test)
  X_test <- X_test[test_sample_ids, , drop = FALSE]

  cat(sprintf("\n  Final: train %d x %d | test %d x %d\n\n",
              nrow(X_train), ncol(X_train),
              nrow(X_test),  ncol(X_test)))

  # -- Train and predict for each BLUP trait ----------------------
  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("Training and Predicting\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")

  all_preds <- data.frame(Sample_ID = test_sample_ids)

  for (trait in blup_traits) {
    cat(sprintf("[%s] Training...\n", trait))

    y_all <- pheno_aligned[, trait]
    msks <- !is.na(y_all)

    if (sum(msks) < 10) {
      cat(sprintf("  SKIP: only %d non-NA training samples\n", sum(msks)))
      next
    }

    X_tr <- as.matrix(X_train[msks, , drop = FALSE])
    y_tr <- as.numeric(y_all[msks])

    # Train model
    method <- "REML"
    model <- tryCatch(
      mixed.solve(y = y_tr, Z = X_tr, method = method),
      error = function(e) {
        cat(sprintf("  REML failed, retrying with ML...\n"))
        mixed.solve(y = y_tr, Z = X_tr, method = "ML")
      }
    )

    u <- as.numeric(model$u)

    # Predict on test
    X_te <- as.matrix(X_test)
    preds <- as.vector(X_te %*% u)

    # Predict on training to check
    tr_preds <- as.vector(X_tr %*% u)
    tr_r <- suppressWarnings(cor(y_tr, tr_preds))

    cat(sprintf("  Train R = %.4f (n=%d)\n", tr_r, sum(msks)))
    cat(sprintf("  Test  predictions: mean=%.3f  sd=%.3f  range=[%.3f, %.3f]\n",
                mean(preds), sd(preds), min(preds), max(preds)))

    col_name <- switch(trait,
                       "PH_BLUP" = "PlantHeight",
                       "HD_BLUP" = "HeadingDate",
                       "GYP_BLUP" = "YieldPerPlant")
    all_preds[[col_name]] <- preds

    # Save marker effects
    eff_file <- file.path(output_dir, sprintf("marker_effects_%s.tsv", trait))
    eff_df <- data.frame(variant = common_vars, effect = u)
    write.table(eff_df, eff_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat(sprintf("  Saved marker effects: %s\n\n", eff_file))
  }

  # -- Save predictions ------------------------------------------
  preds_file <- file.path(output_dir, "preds.tsv")
  write.table(all_preds, preds_file, sep = "\t", row.names = FALSE, quote = FALSE)
  cat(sprintf("\nSaved predictions: %s\n", preds_file))
  cat(sprintf("  Rows: %d  Cols: %s\n", nrow(all_preds),
              paste(colnames(all_preds)[-1], collapse = ", ")))

  cat("\nDone!\n")
}

if (!interactive())
  main()
