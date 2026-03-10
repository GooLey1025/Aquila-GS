#!/usr/bin/env Rscript

# BayesCpi Training Script for Genomic Selection Benchmark
# Supports multi-phenotype parallel training
# Uses hibayes package for BayesCpi model
# Note: BayesCpi uses MCMC with fixed parameters, no HPO needed

suppressPackageStartupMessages({
  library(data.table)
  library(vcfR)
  library(hibayes)
  library(yaml)
  library(optparse)
  library(parallel)
})

# Parse command line arguments
parse_args <- function() {
  option_list <- list(
    make_option(c("--train-vcf"), type="character", default=NULL,
                help="Path to training VCF file", metavar="character"),
    make_option(c("--valid-vcf"), type="character", default=NULL,
                help="Path to validation VCF file", metavar="character"),
    make_option(c("--train-pheno"), type="character", default=NULL,
                help="Path to training phenotype TSV file", metavar="character"),
    make_option(c("--valid-pheno"), type="character", default=NULL,
                help="Path to validation phenotype TSV file", metavar="character"),
    make_option(c("--output-dir"), type="character", default="./bayescpi_output",
                help="Output directory for results [default=%default]", metavar="character"),
    make_option(c("--config"), type="character", default=NULL,
                help="Path to YAML config file (optional)", metavar="character"),
    make_option(c("--n-cores"), type="integer", default=NULL,
                help="Number of cores for parallel training [default=detectCores()-1]", metavar="integer"),
    make_option(c("--seed"), type="integer", default=42,
                help="Random seed [default=%default]", metavar="integer")
  )
  
  opt_parser <- OptionParser(option_list=option_list)
  opt <- optparse::parse_args(opt_parser)
  
  # Validate required arguments
  if (is.null(opt$`train-vcf`)) {
    stop("--train-vcf is required")
  }
  if (is.null(opt$`valid-vcf`)) {
    stop("--valid-vcf is required")
  }
  if (is.null(opt$`train-pheno`)) {
    stop("--train-pheno is required")
  }
  if (is.null(opt$`valid-pheno`)) {
    stop("--valid-pheno is required")
  }
  
  return(opt)
}

# Load VCF and convert to numeric matrix
load_vcf_to_matrix <- function(vcf_file) {
  cat(sprintf("Loading VCF file: %s\n", vcf_file))
  
  # Check if file exists
  if (!file.exists(vcf_file)) {
    stop(sprintf("Error: VCF file not found: %s", vcf_file))
  }
  
  # Load VCF with error handling
  tryCatch({
    vcf <- read.vcfR(vcf_file, verbose = FALSE)
    cat(sprintf("  VCF file read successfully\n"))
    
    # Fix duplicate IDs in VCF object (if ID column contains duplicates like "SNP")
    # This prevents extract.gt from failing
    if (any(duplicated(vcf@fix[, "ID"]))) {
      cat(sprintf("  Fixing duplicate IDs in VCF (found %d duplicates)...\n", 
                  sum(duplicated(vcf@fix[, "ID"]))))
      vcf@fix[, "ID"] <- paste0("variant_", 1:nrow(vcf@fix))
    }
  }, error = function(e) {
    stop(sprintf("Error reading VCF file %s: %s", vcf_file, e$message))
  })
  
  # Extract genotype matrix
  tryCatch({
    # Extract genotypes
    geno <- extract.gt(vcf, element = "GT")
    colnames(geno) <- colnames(vcf@gt)[-1]  # Sample IDs
    
    # Use row indices as variant IDs (simple and guaranteed unique)
    rownames(geno) <- paste0("variant_", 1:nrow(geno))
    
    cat(sprintf("  Extracted genotype matrix: %d variants x %d samples\n", nrow(geno), ncol(geno)))
  }, error = function(e) {
    stop(sprintf("Error extracting genotypes: %s", e$message))
  })
  
  # Convert genotypes to numeric: 0/0 or 0|0 -> 0, 0/1 or 1/0 or 0|1 or 1|0 -> 1, 1/1 or 1|1 -> 2
  # hibayes requires 0, 1, 2 format (number of ALT alleles)
  cat("  Converting genotypes to numeric format (0/1/2 for hibayes)...\n")
  tryCatch({
    # Handle both slash (0/0) and pipe (0|0) formats
    geno[geno == "0/0" | geno == "0|0"] <- 0
    geno[geno == "0/1" | geno == "1/0" | geno == "0|1" | geno == "1|0"] <- 1
    geno[geno == "1/1" | geno == "1|1"] <- 2
    # Handle missing values (./. or .|.)
    geno[geno == "./." | geno == ".|." | is.na(geno)] <- NA
    
    # Convert to numeric matrix (more efficient than apply)
    # Row names are already set above, so don't overwrite them
    storage.mode(geno) <- "numeric"
    cat("  Genotype conversion completed\n")
  }, error = function(e) {
    stop(sprintf("Error converting genotypes: %s", e$message))
  })
  
  # Check for missing values and impute with column means
  n_missing <- sum(is.na(geno))
  if (n_missing > 0) {
    cat(sprintf("  Found %d missing values (%.2f%%)\n", 
                n_missing, 100 * n_missing / length(geno)))
    cat("  Imputing missing values with column means...\n")
    # Calculate column means (for each SNP, across samples)
    col_means <- rowMeans(geno, na.rm = TRUE)
    # Replace NA with column mean (round to nearest integer for 0/1/2 encoding)
    for (i in seq_len(nrow(geno))) {
      geno[i, is.na(geno[i, ])] <- round(col_means[i])
    }
    cat("  Imputation completed\n")
  } else {
    cat("  No missing values found\n")
  }
  
  # Transpose: rows = samples, columns = variants
  cat("  Transposing matrix...\n")
  tryCatch({
    geno_t <- t(geno)
    cat(sprintf("  Loaded %d samples x %d variants\n", nrow(geno_t), ncol(geno_t)))
  }, error = function(e) {
    stop(sprintf("Error transposing matrix: %s", e$message))
  })
  
  return(geno_t)
}

# Load phenotype data
load_phenotypes <- function(pheno_file) {
  cat(sprintf("Loading phenotype file: %s\n", pheno_file))
  pheno <- fread(pheno_file, sep="\t", header=TRUE, data.table=FALSE)
  rownames(pheno) <- pheno$Sample_ID
  
  # Remove Sample_ID column
  pheno_values <- pheno[, -1, drop=FALSE]
  
  cat(sprintf("  Loaded %d samples x %d phenotypes\n", nrow(pheno_values), ncol(pheno_values)))
  return(list(sample_ids = pheno$Sample_ID, phenotypes = pheno_values))
}

# Load configuration from YAML
load_config <- function(config_file) {
  if (!is.null(config_file) && file.exists(config_file)) {
    cat(sprintf("Loading config from: %s\n", config_file))
    config <- yaml.load_file(config_file)
    return(config)
  } else {
    # Return default config for BayesCpi
    # Note: Pi is not used for BayesCpi (it's estimated, not fixed)
    cat("Using default configuration\n")
    return(list(
      hyperparameters = list(
        Pi = 0.95,      # Not used for BayesCpi (Pi is estimated), kept for compatibility
        niter = 12000,  # Number of MCMC iterations
        nburn = 2000,   # Number of burn-in iterations
        thin = 5,       # Thinning interval
        verbose = FALSE
      )
    ))
  }
}

# Train BayesCpi model for a single phenotype
train_bayescpi_single <- function(X_train, y_train, X_valid, y_valid, params, pheno_name, output_dir) {
  # Ensure data is matrix
  X_train_mat <- as.matrix(X_train)
  X_valid_mat <- as.matrix(X_valid)
  y_train_vec <- as.numeric(y_train)
  y_valid_vec <- as.numeric(y_valid)
  
  # hibayes requires phenotype data frame with first column as sample ID
  # Create phenotype data frame for training
  train_pheno_df <- data.frame(
    id = rownames(X_train),
    y = y_train_vec,
    stringsAsFactors = FALSE
  )
  
  # Create formula for ibrm (y ~ 1 means intercept only, no fixed effects)
  formula_y <- y ~ 1
  
  # Train BayesCpi model using hibayes
  # Note: For BayesCpi, Pi is estimated (not fixed), so we don't pass Pi parameter
  bayes_model <- tryCatch({
    ibrm(
      formula = formula_y,     # Formula: y ~ 1 (intercept only)
      data = train_pheno_df,   # Phenotype data frame (first column: id, second column: y)
      M = X_train_mat,         # Genotype matrix (samples x markers), values 0/1/2
      M.id = rownames(X_train), # Vector of IDs for genotyped individuals
      method = "BayesCpi",     # Method: BayesCpi (Pi is estimated, not fixed)
      niter = params$niter,    # Number of MCMC iterations
      nburn = params$nburn,    # Number of burn-in iterations
      thin = params$thin,      # Thinning interval
      verbose = params$verbose
    )
  }, error = function(e) {
    stop(sprintf("BayesCpi training failed: %s", e$message))
  })
  
  # Extract marker effects (alpha) and intercept (mu)
  # Based on hibayes documentation and example code, effects are in $alpha
  marker_effects <- as.matrix(bayes_model$alpha)
  intercept <- bayes_model$mu
  
  # Predict on train and validation
  # Training predictions: X * alpha + mu
  pred_train <- (X_train_mat %*% marker_effects)[, 1] + intercept
  
  # For validation, compute predictions manually
  pred_valid <- (X_valid_mat %*% marker_effects)[, 1] + intercept
  
  # Robust correlation calculation with proper error handling
  train_r <- suppressWarnings(cor(y_train_vec, pred_train))
  valid_r <- suppressWarnings(cor(y_valid_vec, as.vector(pred_valid)))
  
  if (!is.finite(train_r)) train_r <- NA
  if (!is.finite(valid_r)) valid_r <- NA
  
  return(list(
    model = bayes_model,
    marker_effects = marker_effects,
    pred_train = pred_train,
    pred_valid = as.vector(pred_valid),
    train_r = train_r,
    valid_r = valid_r
  ))
}

# Note: BayesCpi does not use external HPO
# BayesCpi uses MCMC with fixed hyperparameters (Pi)
# The MCMC process itself is the optimization procedure

# Main function
main <- function() {
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("BayesCpi Genomic Selection Benchmark\n")
  cat(paste(rep("=", 80), collapse=""), "\n\n")
  
  # Parse arguments
  args <- parse_args()
  
  # Set seed
  set.seed(args$seed)
  
  # Create output directory
  if (!dir.exists(args$`output-dir`)) {
    dir.create(args$`output-dir`, recursive = TRUE)
  }
  
  # Setup logging (split output to both console and file)
  log_file <- file.path(args$`output-dir`, "training.log")
  log_con <- file(log_file, open = "wt")
  sink(log_con, type = "output", split = TRUE)
  # Don't sink messages to avoid hiding errors
  # sink(log_con, type = "message")
  
  cat(sprintf("Output directory: %s\n", args$`output-dir`))
  cat(sprintf("Random seed: %d\n\n", args$seed))
  cat("Note: BayesCpi uses MCMC and can be slow. Please be patient.\n\n")
  
  # Load configuration
  config <- load_config(args$config)
  
  # Load data
  cat("\n", paste(rep("=", 80), collapse=""), "\n")
  cat("Loading Data\n")
  cat(paste(rep("=", 80), collapse=""), "\n\n")
  
  X_train <- load_vcf_to_matrix(args$`train-vcf`)
  X_valid <- load_vcf_to_matrix(args$`valid-vcf`)
  
  train_pheno <- load_phenotypes(args$`train-pheno`)
  valid_pheno <- load_phenotypes(args$`valid-pheno`)
  
  # Ensure sample order matches
  train_samples <- rownames(X_train)
  valid_samples <- rownames(X_valid)
  
  # Reorder phenotypes to match genotypes
  y_train_all <- train_pheno$phenotypes[match(train_samples, train_pheno$sample_ids), , drop=FALSE]
  y_valid_all <- valid_pheno$phenotypes[match(valid_samples, valid_pheno$sample_ids), , drop=FALSE]
  
  phenotype_names <- colnames(y_train_all)
  cat(sprintf("\nTotal phenotypes to train: %d\n", length(phenotype_names)))
  
  # Prepare variables
  seed_value <- args$seed
  output_dir <- args$`output-dir`
  
  # Setup parallel processing
  n_cores <- args$`n-cores`
  if (is.null(n_cores)) {
    n_cores <- max(1, parallel::detectCores() - 1)
  }
  cat(sprintf("Using %d cores for parallel training\n", n_cores))
  
  # Train all phenotypes in parallel
  cat("\n", paste(rep("=", 80), collapse=""), "\n")
  cat("Training Models\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  
  # Define worker function that takes a task list with all needed data
  # This avoids closure issues in parallel execution
  train_one_phenotype <- function(task) {
    pheno_name <- task$pheno_name
    X_tr <- task$X_train
    y_tr_all <- task$y_train_all
    X_val <- task$X_valid
    y_val_all <- task$y_valid_all
    cfg <- task$config
    seed_val <- task$seed_value
    out_dir <- task$output_dir
    
    # Each worker process uses its own log file to avoid sink conflicts
    pheno_log_file <- file.path(out_dir, paste0("pheno_", pheno_name, ".log"))
    pheno_log_con <- file(pheno_log_file, open = "wt")
    sink(pheno_log_con, type = "output")
    
    tryCatch({
      cat(sprintf("\n=== Training phenotype: %s ===\n", pheno_name))
      
      # Extract phenotype values
      y_train <- y_tr_all[, pheno_name]
      y_valid <- y_val_all[, pheno_name]
      
      # Remove samples with missing values
      train_mask <- !is.na(y_train)
      valid_mask <- !is.na(y_valid)
      
      X_train_clean <- X_tr[train_mask, , drop=FALSE]
      y_train_clean <- y_train[train_mask]
      X_valid_clean <- X_val[valid_mask, , drop=FALSE]
      y_valid_clean <- y_valid[valid_mask]
      
      # Check data validity
      if (sum(train_mask) < 10) {
        stop(sprintf("Not enough training samples (only %d non-missing)", sum(train_mask)))
      }
      if (sum(valid_mask) < 2) {
        stop(sprintf("Not enough validation samples (only %d non-missing)", sum(valid_mask)))
      }
      
      # Check for constant phenotype values
      train_var <- var(y_train_clean, na.rm=TRUE)
      if (is.na(train_var) || train_var == 0) {
        stop("Training phenotype has zero variance")
      }
      
      cat(sprintf("  Train samples: %d, Valid samples: %d\n", 
                  sum(train_mask), sum(valid_mask)))
      
      # Get parameters from config
      params <- cfg$hyperparameters
      
      # Train model
      cat(sprintf("  Training BayesCpi model (niter=%d, nburn=%d, Pi=%.3f)...\n", 
                  params$niter, params$nburn, params$Pi))
      cat("  Note: This may take several minutes per phenotype...\n")
      
      result <- train_bayescpi_single(X_train_clean, y_train_clean, 
                                       X_valid_clean, y_valid_clean, 
                                       params, pheno_name, out_dir)
      
      cat(sprintf("  Train R: %.4f, Valid R: %.4f\n", result$train_r, result$valid_r))
      
      return(data.frame(
        phenotype = pheno_name,
        train_r = result$train_r,
        valid_r = result$valid_r,
        niter = params$niter,
        nburn = params$nburn,
        Pi = params$Pi,
        n_train = sum(train_mask),
        n_valid = sum(valid_mask),
        stringsAsFactors = FALSE
      ))
      
    }, error = function(e) {
      # Write error to file
      error_file <- file.path(out_dir, paste0("error_", pheno_name, ".txt"))
      tryCatch({
        write(sprintf("ERROR in phenotype %s:\n%s\n", pheno_name, e$message), error_file)
      }, error = function(e2) {})
      
      cat(sprintf("\nERROR in %s: %s\n", pheno_name, e$message))
      
      return(data.frame(
        phenotype = pheno_name,
        train_r = NA,
        valid_r = NA,
        niter = NA,
        nburn = NA,
        Pi = NA,
        n_train = NA,
        n_valid = NA,
        stringsAsFactors = FALSE
      ))
    }, finally = {
      # Restore output and close per-phenotype log file
      sink(type = "output")
      close(pheno_log_con)
    })
  }
  
  # Create task list with all data needed for each phenotype
  # This ensures data is explicitly passed rather than relying on closures
  tasks <- lapply(phenotype_names, function(pname) {
    list(
      pheno_name = pname,
      X_train = X_train,
      y_train_all = y_train_all,
      X_valid = X_valid,
      y_valid_all = y_valid_all,
      config = config,
      seed_value = seed_value,
      output_dir = output_dir
    )
  })
  
  if (n_cores > 1) {
    # Parallel execution
    results_list <- parallel::mclapply(tasks, train_one_phenotype, 
                                       mc.cores = n_cores, mc.preschedule = FALSE)
  } else {
    # Sequential execution
    results_list <- lapply(tasks, train_one_phenotype)
  }
  
  # Combine results
  results <- do.call(rbind, results_list)
  
  # Check for errors in results
  failed_phenos <- results$phenotype[is.na(results$valid_r)]
  if (length(failed_phenos) > 0) {
    cat(sprintf("\nWARNING: %d phenotypes failed to train:\n", length(failed_phenos)))
    cat(paste(failed_phenos, collapse=", "), "\n")
    cat("Check error_*.txt files in output directory for details\n")
  }
  
  # Save metrics
  cat("\n", paste(rep("=", 80), collapse=""), "\n")
  cat("Saving Results\n")
  cat(paste(rep("=", 80), collapse=""), "\n\n")
  
  metrics_file <- file.path(args$`output-dir`, "metrics_per_phenotype.tsv")
  write.table(results, metrics_file, sep="\t", row.names=FALSE, quote=FALSE)
  cat(sprintf("Saved per-phenotype metrics: %s\n", metrics_file))
  
  # Calculate and save summary statistics
  successful_results <- results[!is.na(results$valid_r), ]
  failed_count <- sum(is.na(results$valid_r))
  
  if (nrow(successful_results) > 0) {
    avg_train_r <- mean(successful_results$train_r, na.rm=TRUE)
    avg_valid_r <- mean(successful_results$valid_r, na.rm=TRUE)
    median_train_r <- median(successful_results$train_r, na.rm=TRUE)
    median_valid_r <- median(successful_results$valid_r, na.rm=TRUE)
    min_train_r <- min(successful_results$train_r, na.rm=TRUE)
    min_valid_r <- min(successful_results$valid_r, na.rm=TRUE)
    max_train_r <- max(successful_results$train_r, na.rm=TRUE)
    max_valid_r <- max(successful_results$valid_r, na.rm=TRUE)
    sd_train_r <- sd(successful_results$train_r, na.rm=TRUE)
    sd_valid_r <- sd(successful_results$valid_r, na.rm=TRUE)
  } else {
    avg_train_r <- avg_valid_r <- median_train_r <- median_valid_r <- NA
    min_train_r <- min_valid_r <- max_train_r <- max_valid_r <- NA
    sd_train_r <- sd_valid_r <- NA
  }
  
  summary_file <- file.path(args$`output-dir`, "metrics_summary.txt")
  cat("=== BayesCpi Training Summary ===\n", file=summary_file)
  cat(sprintf("Total Phenotypes: %d\n", nrow(results)), file=summary_file, append=TRUE)
  cat(sprintf("Successful: %d\n", nrow(successful_results)), file=summary_file, append=TRUE)
  cat(sprintf("Failed: %d\n", failed_count), file=summary_file, append=TRUE)
  cat("\n=== Validation Set Performance (Valid R) ===\n", file=summary_file, append=TRUE)
  cat(sprintf("Mean: %.4f\n", avg_valid_r), file=summary_file, append=TRUE)
  cat(sprintf("Median: %.4f\n", median_valid_r), file=summary_file, append=TRUE)
  cat(sprintf("SD: %.4f\n", sd_valid_r), file=summary_file, append=TRUE)
  cat(sprintf("Min: %.4f\n", min_valid_r), file=summary_file, append=TRUE)
  cat(sprintf("Max: %.4f\n", max_valid_r), file=summary_file, append=TRUE)
  cat("\n=== Training Set Performance (Train R) ===\n", file=summary_file, append=TRUE)
  cat(sprintf("Mean: %.4f\n", avg_train_r), file=summary_file, append=TRUE)
  cat(sprintf("Median: %.4f\n", median_train_r), file=summary_file, append=TRUE)
  cat(sprintf("SD: %.4f\n", sd_train_r), file=summary_file, append=TRUE)
  cat(sprintf("Min: %.4f\n", min_train_r), file=summary_file, append=TRUE)
  cat(sprintf("Max: %.4f\n", max_train_r), file=summary_file, append=TRUE)
  
  cat(sprintf("\nSummary:\n"))
  cat(sprintf("  Total Phenotypes: %d\n", nrow(results)))
  cat(sprintf("  Successful: %d\n", nrow(successful_results)))
  cat(sprintf("  Failed: %d\n", failed_count))
  if (nrow(successful_results) > 0) {
    cat(sprintf("\n  Validation Set Performance (Valid R):\n"))
    cat(sprintf("    Mean: %.4f\n", avg_valid_r))
    cat(sprintf("    Median: %.4f\n", median_valid_r))
    cat(sprintf("    SD: %.4f\n", sd_valid_r))
    cat(sprintf("    Range: [%.4f, %.4f]\n", min_valid_r, max_valid_r))
    cat(sprintf("\n  Training Set Performance (Train R):\n"))
    cat(sprintf("    Mean: %.4f\n", avg_train_r))
    cat(sprintf("    Median: %.4f\n", median_train_r))
    cat(sprintf("    SD: %.4f\n", sd_train_r))
    cat(sprintf("    Range: [%.4f, %.4f]\n", min_train_r, max_train_r))
  }
  
  cat(sprintf("\nResults saved to: %s\n", args$`output-dir`))
  
  # Close logging
  sink(type = "output")
  # sink(type = "message")  # Don't sink messages to see errors
  close(log_con)
  
  cat("\nTraining complete!\n")
}

# Run main function
if (!interactive()) {
  main()
}

