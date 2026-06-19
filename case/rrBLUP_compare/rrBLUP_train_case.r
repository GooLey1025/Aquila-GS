#!/usr/bin/env Rscript

# rrBLUP Training Script for Cross-Population Genomic Selection
# Uses data_split.tsv to partition the 705-rice population
# Supports multi-phenotype training with REML/ML method selection

suppressPackageStartupMessages({
  library(data.table)
  library(vcfR)
  library(rrBLUP)
  library(optparse)
  library(parallel)
})

parse_args <- function() {
  option_list <- list(
    make_option(c("--vcf"), type = "character",
                default = "705rice_0.03.full.all.impute.biallelic.vcf.gz",
                help = "Population VCF file (gzipped)", metavar = "character"),
    make_option(c("--pheno"), type = "character",
                default = "GSTP008.pheno.tsv",
                help = "Phenotype TSV file", metavar = "character"),
    make_option(c("--split"), type = "character",
                default = "data_split.tsv",
                help = "Data split TSV file (Sample_ID \\t Split)", metavar = "character"),
    make_option(c("--blup-ref"), type = "character",
                default = "blup_phenotype.tsv",
                help = "BLUP reference file for cross-population prediction",
                metavar = "character"),
    make_option(c("--output-dir"), type = "character",
                default = "rrblup_output",
                help = "Output directory [default=%default]", metavar = "character"),
    make_option(c("--n-cores"), type = "integer", default = NULL,
                help = "Number of cores [default=detectCores()-1]", metavar = "integer"),
    make_option(c("--seed"), type = "integer", default = 42,
                help = "Random seed [default=%default]", metavar = "integer"),
    make_option(c("--test-vcf"), type = "character",
                default = "120_inbred_line.snp.indel.sv.impute.biallelic.vcf.gz",
                help = "Cross-population test VCF for prediction", metavar = "character")
  )

  opt_parser <- OptionParser(option_list = option_list,
                            description = "rrBLUP training with data_split.tsv partition")
  opt <- optparse::parse_args(opt_parser)

  if (!file.exists(opt$vcf))
    stop(sprintf("Training VCF not found: %s", opt$vcf))
  if (!file.exists(opt$pheno))
    stop(sprintf("Phenotype file not found: %s", opt$pheno))
  if (!file.exists(opt$split))
    stop(sprintf("Data split file not found: %s", opt$split))

  return(opt)
}

load_vcf_to_matrix <- function(vcf_file, sample_ids = NULL) {
  cat(sprintf("[VCF] Loading: %s\n", vcf_file))

  if (!file.exists(vcf_file))
    stop(sprintf("VCF file not found: %s", vcf_file))

  vcf <- read.vcfR(vcf_file, verbose = FALSE)
  cat(sprintf("  VCF loaded: %d variants x %d samples\n", nrow(vcf@fix), ncol(vcf@gt) - 1))

  # Keep only SNP variants (ID column starts with "SNP")
  snp_mask <- grepl("^SNP", vcf@fix[, "ID"])
  n_snp <- sum(snp_mask)
  cat(sprintf("  Filtering to SNP-only: %d / %d variants retained\n",
              n_snp, nrow(vcf@fix)))
  vcf <- vcf[snp_mask, ]

  if (any(duplicated(vcf@fix[, "ID"]))) {
    cat(sprintf("  Fixing %d duplicate variant IDs\n",
                sum(duplicated(vcf@fix[, "ID"]))))
    vcf@fix[, "ID"] <- paste0("v", seq_len(nrow(vcf@fix)))
  }

  geno <- extract.gt(vcf, element = "GT")
  colnames(geno) <- colnames(vcf@gt)[-1]
  rownames(geno) <- paste0("v", seq_len(nrow(geno)))

  cat("  Converting genotypes to numeric (-1/0/1)...\n")
  geno[geno %in% c("0/0", "0|0")] <- -1
  geno[geno %in% c("0/1", "1/0", "0|1", "1|0")] <- 0
  geno[geno %in% c("1/1", "1|1")] <- 1
  geno[geno %in% c("./.", ".|.")] <- NA
  storage.mode(geno) <- "numeric"

  n_miss <- sum(is.na(geno))
  if (n_miss > 0) {
    cat(sprintf("  Imputing %d NAs (%.2f%%) with row means\n",
                n_miss, 100 * n_miss / length(geno)))
    rmeans <- rowMeans(geno, na.rm = TRUE)
    for (i in seq_len(nrow(geno)))
      geno[i, is.na(geno[i, ])] <- rmeans[i]
  }

  if (!is.null(sample_ids)) {
    matched <- match(sample_ids, colnames(geno))
    missing_s <- sample_ids[is.na(matched)]
    if (length(missing_s) > 0)
      cat(sprintf("  WARNING: %d sample IDs not found in VCF\n", length(missing_s)))
    geno <- geno[, matched, drop = FALSE]
    colnames(geno) <- sample_ids
  }

  geno_t <- t(geno)
  cat(sprintf("  Final matrix: %d samples x %d variants\n\n", nrow(geno_t), ncol(geno_t)))
  return(geno_t)
}

load_phenotypes <- function(pheno_file) {
  cat(sprintf("[PHENO] Loading: %s\n", pheno_file))
  pheno <- fread(pheno_file, sep = "\t", header = TRUE, data.table = FALSE)
  cat(sprintf("  Loaded %d samples x %d columns\n", nrow(pheno), ncol(pheno)))
  return(pheno)
}

train_rrblup <- function(X_train, y_train, X_valid, y_valid, method = "REML") {
  X_tr <- as.matrix(X_train)
  X_va <- as.matrix(X_valid)
  y_tr <- as.numeric(y_train)
  y_va <- as.numeric(y_valid)

  model <- mixed.solve(y = y_tr, Z = X_tr, method = method)
  u <- as.numeric(model$u)

  pred_tr <- as.vector(X_tr %*% u)
  pred_va <- as.vector(X_va %*% u)

  train_r <- suppressWarnings(cor(y_tr, pred_tr))
  valid_r <- suppressWarnings(cor(y_va, pred_va))

  if (!is.finite(train_r)) train_r <- NA
  if (!is.finite(valid_r)) valid_r <- NA

  list(marker_effects = u, pred_train = pred_tr, pred_valid = pred_va,
       train_r = train_r, valid_r = valid_r)
}

main <- function() {
  args <- parse_args()
  set.seed(args$seed)

  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("rrBLUP Cross-Population Genomic Selection\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")

  output_dir <- args$`output-dir`
  if (!dir.exists(output_dir))
    dir.create(output_dir, recursive = TRUE)

  cat(sprintf("Output dir : %s\n", output_dir))
  cat(sprintf("Seed       : %d\n\n", args$seed))

  # -- Load split ---------------------------------------------------
  cat("[SPLIT] Reading data_split.tsv...\n")
  split_df <- fread(args$split, sep = "\t", header = TRUE, data.table = FALSE)
  cat(sprintf("  Total samples in split file: %d\n", nrow(split_df)))
  cat(sprintf("  Train: %d, Valid: %d\n",
              sum(split_df$Split == "train"),
              sum(split_df$Split == "valid")))

  train_ids <- split_df$Sample_ID[split_df$Split == "train"]
  valid_ids <- split_df$Sample_ID[split_df$Split == "valid"]

  # -- Load genotype -----------------------------------------------
  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("Loading Genotypes\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")
  X_all <- load_vcf_to_matrix(args$vcf)

  # Keep only samples present in split file
  all_ids <- intersect(rownames(X_all), c(train_ids, valid_ids))
  X_all <- X_all[all_ids, , drop = FALSE]
  cat(sprintf("  Samples matched in VCF: %d\n\n", nrow(X_all)))

  # -- Load phenotype ----------------------------------------------
  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("Loading Phenotypes\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")
  pheno_raw <- load_phenotypes(args$pheno)
  colnames(pheno_raw)[1] <- "Sample_ID"
  cat(sprintf("  Phenotype columns: %d\n", ncol(pheno_raw) - 1))
  cat(sprintf("  Column names: %s\n\n",
              paste(colnames(pheno_raw)[-1], collapse = ", ")))

  # Target traits for BLUP correlation later
  target_traits <- c("PH_BLUP", "HD_BLUP", "GYP_BLUP")
  blup_available <- intersect(target_traits, colnames(pheno_raw))
  if (length(blup_available) == 0)
    stop("None of PH_BLUP, HD_BLUP, GYP_BLUP found in phenotype file")

  # Additional individual-level phenotypes (use all non-BLUP columns)
  all_pheno_cols <- colnames(pheno_raw)[-1]
  # Remove BLUP suffix columns (we keep BLUP ones as targets)
  extra_traits <- setdiff(all_pheno_cols, target_traits)
  cat(sprintf("  BLUP target traits : %s\n", paste(blup_available, collapse = ", ")))
  cat(sprintf("  Extra phenotypes   : %d\n\n", length(extra_traits)))

  # Align phenotypes with genotype samples
  pheno_matched <- pheno_raw[match(all_ids, pheno_raw$Sample_ID), , drop = FALSE]
  rownames(pheno_matched) <- pheno_matched$Sample_ID

  # -- Partition ---------------------------------------------------
  train_mask <- rownames(X_all) %in% train_ids
  valid_mask <- rownames(X_all) %in% valid_ids

  X_train <- X_all[train_mask, , drop = FALSE]
  X_valid <- X_all[valid_mask, , drop = FALSE]

  y_train_all <- pheno_matched[train_mask, , drop = FALSE]
  y_valid_all <- pheno_matched[valid_mask, , drop = FALSE]

  cat(sprintf("Train: %d samples | Valid: %d samples\n\n",
              nrow(X_train), nrow(X_valid)))

  # -- Setup parallel ----------------------------------------------
  n_cores <- args$`n-cores`
  if (is.null(n_cores))
    n_cores <- max(1, parallel::detectCores() - 1)
  cat(sprintf("Using %d cores\n\n", n_cores))

  # -- Train all phenotypes ----------------------------------------
  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("Training Models\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")

  # Collect all phenotype columns to train
  pheno_names <- setdiff(colnames(pheno_matched), "Sample_ID")
  cat(sprintf("Training %d phenotypes...\n\n", length(pheno_names)))

  `%||%` <- function(a, b) if (!is.null(a)) a else b

  train_one <- function(pname) {
    out_dir <- output_dir
    log_f <- file.path(out_dir, paste0("log_", pname, ".txt"))
    log_c <- file(log_f, open = "wt")
    sink(log_c, type = "output")

    tryCatch({
      y_tr <- y_train_all[, pname]
      y_va <- y_valid_all[, pname]

      tr_msk <- !is.na(y_tr)
      va_msk <- !is.na(y_va)

      X_tr <- X_train[tr_msk, , drop = FALSE]
      y_tr <- y_tr[tr_msk]
      X_va <- X_valid[va_msk, , drop = FALSE]
      y_va <- y_va[va_msk]

      if (sum(tr_msk) < 10 || sum(va_msk) < 2) {
        cat(sprintf("SKIP %s: insufficient non-NA samples\n", pname))
        return(data.frame(phenotype = pname, train_r = NA_real_,
                         valid_r = NA_real_, method = NA_character_,
                         n_train = NA_integer_, n_valid = NA_integer_,
                         stringsAsFactors = FALSE))
      }

      if (var(y_tr, na.rm = TRUE) == 0) {
        cat(sprintf("SKIP %s: zero variance\n", pname))
        return(data.frame(phenotype = pname, train_r = NA_real_,
                         valid_r = NA_real_, method = NA_character_,
                         n_train = NA_integer_, n_valid = NA_integer_,
                         stringsAsFactors = FALSE))
      }

      # Try REML first; fall back to ML if singular
      method <- "REML"
      result <- tryCatch(
        train_rrblup(X_tr, y_tr, X_va, y_va, method),
        error = function(e) {
          cat(sprintf("  REML failed (%s), retrying with ML...\n", e$message))
          train_rrblup(X_tr, y_tr, X_va, y_va, "ML")
        }
      )

      cat(sprintf("[DONE] %s | Train R=%.4f  Valid R=%.4f  (n_tr=%d n_va=%d)\n",
                  pname, result$train_r, result$valid_r,
                  sum(tr_msk), sum(va_msk)))

      data.frame(phenotype = pname, train_r = result$train_r,
                 valid_r = result$valid_r, method = method,
                 n_train = as.integer(sum(tr_msk)),
                 n_valid = as.integer(sum(va_msk)),
                 stringsAsFactors = FALSE)

    }, error = function(e) {
      err_f <- file.path(out_dir, paste0("err_", pname, ".txt"))
      tryCatch(write(e$message, err_f), error = function(...) {})
      cat(sprintf("ERROR %s: %s\n", pname, e$message))
      data.frame(phenotype = pname, train_r = NA_real_,
                 valid_r = NA_real_, method = NA_character_,
                 n_train = NA_integer_, n_valid = NA_integer_,
                 stringsAsFactors = FALSE)
    }, finally = {
      sink(type = "output")
      close(log_c)
    })
  }

  if (n_cores > 1) {
    results_list <- parallel::mclapply(pheno_names, train_one,
                                       mc.cores = n_cores,
                                       mc.preschedule = FALSE)
  } else {
    results_list <- lapply(pheno_names, train_one)
  }

  results <- do.call(rbind, results_list)

  # -- Save metrics ------------------------------------------------
  cat(paste0(rep("=", 70), collapse = ""), "\n")
  cat("Saving Results\n")
  cat(paste0(rep("=", 70), collapse = ""), "\n\n")

  metrics_file <- file.path(output_dir, "metrics_per_phenotype.tsv")
  write.table(results, metrics_file, sep = "\t",
             row.names = FALSE, quote = FALSE)
  cat(sprintf("Saved: %s\n", metrics_file))

  # Best model per target BLUP trait
  cat("\n--- Best results for BLUP target traits ---\n")
  for (trait in blup_available) {
    sub <- results[results$phenotype == trait, ]
    if (nrow(sub) > 0 && !is.na(sub$valid_r)) {
      cat(sprintf("  %-20s  Train R=%+.4f  Valid R=%+.4f  (n=%d)\n",
                  trait, sub$train_r, sub$valid_r, sub$n_valid))
    }
  }

  # Summary
  ok <- results[!is.na(results$valid_r), ]
  fail <- sum(is.na(results$valid_r))
  cat(sprintf("\nTotal phenotypes : %d\n", nrow(results)))
  cat(sprintf("Successful       : %d\n", nrow(ok)))
  cat(sprintf("Failed           : %d\n", fail))
  if (nrow(ok) > 0) {
    cat(sprintf("\n  Valid R  Mean=%.4f  Median=%.4f  SD=%.4f\n",
                mean(ok$valid_r), median(ok$valid_r), sd(ok$valid_r)))
    cat(sprintf("  Valid R  Min=%.4f  Max=%.4f\n",
                min(ok$valid_r), max(ok$valid_r)))
  }

  # Save validation predictions for BLUP traits
  for (trait in blup_available) {
    y_va <- y_valid_all[, trait]
    va_msk <- !is.na(y_va)
    if (sum(va_msk) < 2) next

    model_r <- results$train_r[results$phenotype == trait]
    if (is.na(model_r)) next

    X_va <- X_valid[va_msk, , drop = FALSE]
    y_va_clean <- y_va[va_msk]

    X_tr_full <- X_train
    y_tr_full <- y_train_all[, trait]
    tr_msk <- !is.na(y_tr_full)
    X_tr_full <- X_tr_full[tr_msk, , drop = FALSE]
    y_tr_full <- y_tr_full[tr_msk]

    method <- results$method[results$phenotype == trait] %||% "REML"
    m <- mixed.solve(y = y_tr_full, Z = as.matrix(X_tr_full), method = method)
    preds_va <- as.vector(as.matrix(X_va) %*% as.numeric(m$u))

    pred_df <- data.frame(
      Sample_ID = rownames(X_valid)[va_msk],
      observed = y_va_clean,
      predicted = preds_va
    )
    pred_file <- file.path(output_dir, sprintf("valid_pred_%s.tsv", trait))
    write.table(pred_df, pred_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat(sprintf("Saved validation predictions: %s\n", pred_file))
  }

  cat(sprintf("\nAll results saved to: %s\n", output_dir))
  cat("\nDone!\n")
}

if (!interactive())
  main()
