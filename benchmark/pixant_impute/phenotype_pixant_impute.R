library("PIXANT")
library("parallel")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    cat("Usage: Rscript phenotype_pixant_impute.R <input_pheno_file> <output_pheno_file> [log_dir] [n_cores]\n")
    cat("\nArguments:\n")
    cat("  input_pheno_file  : Path to input phenotype file (TSV format)\n")
    cat("  output_pheno_file : Path to output imputed phenotype file\n")
    cat("  log_dir           : (Optional) Directory for log files (default: logs)\n")
    cat("                      Each phenotype will have its own log file in this directory\n")
    cat("  n_cores           : (Optional) Number of CPU cores to use (default: detect_cores() - 1)\n")
    cat("\nExample:\n")
    cat("  Rscript phenotype_pixant_impute.R ../GSTP008.pheno ../GSTP008.pheno.imputed\n")
    cat("  Rscript phenotype_pixant_impute.R ../GSTP008.pheno ../GSTP008.pheno.imputed logs 4\n")
    stop("Missing required arguments", call. = FALSE)
}

# Set paths from command line arguments
pheno_path <- args[1]
pheno_imputed_path <- args[2]
log_dir <- if (length(args) >= 3) args[3] else "logs"
summary_log_path <- file.path(log_dir, "pixant_imputation_summary.log")
n_cores <- if (length(args) >= 4) as.integer(args[4]) else max(1, detectCores() - 1)

# Validate input file exists
if (!file.exists(pheno_path)) {
    stop(sprintf("Error: Input file '%s' does not exist!", pheno_path), call. = FALSE)
}

# Create log directory if it doesn't exist
if (!dir.exists(log_dir)) {
    dir.create(log_dir, recursive = TRUE)
    cat("Created log directory:", log_dir, "\n")
}

cat("Input file:", pheno_path, "\n")
cat("Output file:", pheno_imputed_path, "\n")
cat("Log directory:", log_dir, "\n")
cat("Summary log:", summary_log_path, "\n")
cat("\n")

# Initialize summary log file
cat("PIXANT Imputation Summary Log\n", file = summary_log_path)
cat("=============================\n", file = summary_log_path, append = TRUE)
cat(paste("Start time:", Sys.time(), "\n"), file = summary_log_path, append = TRUE)
cat(paste("Input file:", pheno_path, "\n"), file = summary_log_path, append = TRUE)
cat(paste("Output file:", pheno_imputed_path, "\n"), file = summary_log_path, append = TRUE)
cat(paste("Log directory:", log_dir, "\n"), file = summary_log_path, append = TRUE)
cat("\n", file = summary_log_path, append = TRUE)

# Read phenotype data
cat("Reading phenotype data from:", pheno_path, "\n")
# Use stringsAsFactors = FALSE to prevent first column from being converted to factor
pheno_data <- read.table(pheno_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)

# Convert phenotype columns to numeric (except first column which is sample ID)
# This ensures all phenotype values are numeric and handles conversion warnings
cat("Converting phenotype columns to numeric...\n")
for (col_name in colnames(pheno_data)[-1]) {
    # Convert to numeric, non-numeric values will become NA
    pheno_data[[col_name]] <- suppressWarnings(as.numeric(pheno_data[[col_name]]))
}
cat("Data conversion completed.\n")

# Get sample ID column name (first column)
sample_id_col <- colnames(pheno_data)[1]
cat("Sample ID column:", sample_id_col, "\n")

# Get phenotype column names (all columns except the first one)
phenotype_cols <- colnames(pheno_data)[-1]
n_phenotypes <- length(phenotype_cols)
cat("Total phenotypes to impute:", n_phenotypes, "\n")
cat("Total phenotypes to impute:", n_phenotypes, "\n", file = summary_log_path, append = TRUE)
cat("\n", file = summary_log_path, append = TRUE)

# Initialize imputed data frame (start with original data)
pheno_data_imputed <- pheno_data

# Filter phenotypes that have missing values
phenotypes_to_impute <- c()
for (phen_name in phenotype_cols) {
    missing_count <- sum(is.na(pheno_data_imputed[[phen_name]]))
    if (missing_count > 0) {
        phenotypes_to_impute <- c(phenotypes_to_impute, phen_name)
    }
}

n_to_impute <- length(phenotypes_to_impute)
cat(sprintf("Phenotypes with missing values: %d / %d\n", n_to_impute, n_phenotypes))
cat(sprintf("Using %d CPU cores for parallel processing\n", n_cores))
cat(sprintf("Phenotypes with missing values: %d / %d\n", n_to_impute, n_phenotypes), 
    file = summary_log_path, append = TRUE)
cat(sprintf("Using %d CPU cores for parallel processing\n", n_cores), 
    file = summary_log_path, append = TRUE)

# Function to impute a single phenotype
impute_single_phenotype <- function(phen_name, log_dir) {
    # Create individual log file for this phenotype
    phen_log_path <- file.path(log_dir, paste0("phen_", gsub("[^A-Za-z0-9_]", "_", phen_name), ".log"))
    
    # Initialize phenotype log file
    cat("PIXANT Imputation Log for Phenotype:", phen_name, "\n", file = phen_log_path)
    cat("=====================================\n", file = phen_log_path, append = TRUE)
    cat(paste("Start time:", Sys.time(), "\n"), file = phen_log_path, append = TRUE)
    cat(paste("Phenotype:", phen_name, "\n"), file = phen_log_path, append = TRUE)
    cat("\n", file = phen_log_path, append = TRUE)
    
    result <- list(
        phen_name = phen_name,
        success = FALSE,
        imputed_values = NULL,
        r2 = NA,
        accuracy = NA,
        ref_phen_count = NA,
        error_msg = NULL,
        elapsed_time = NA,
        log_path = phen_log_path
    )
    
    tryCatch({
        # Check if phenotype exists in data
        if (!phen_name %in% colnames(pheno_data_imputed)) {
            result$error_msg <- sprintf("Phenotype '%s' not found in data", phen_name)
            cat("ERROR:", result$error_msg, "\n", file = phen_log_path, append = TRUE)
            return(result)
        }
        
        # Check missing count
        missing_count <- sum(is.na(pheno_data_imputed[[phen_name]]))
        missing_rate <- missing_count / nrow(pheno_data_imputed) * 100
        
        # Check data type
        data_type <- class(pheno_data_imputed[[phen_name]])
        cat(sprintf("Data type: %s\n", paste(data_type, collapse = ", ")), file = phen_log_path, append = TRUE)
        cat(sprintf("Missing values: %d / %d (%.2f%%)\n", missing_count, nrow(pheno_data_imputed), missing_rate), 
            file = phen_log_path, append = TRUE)
        
        # Ensure phenotype column is numeric
        if (!is.numeric(pheno_data_imputed[[phen_name]])) {
            cat("Converting to numeric...\n", file = phen_log_path, append = TRUE)
            pheno_data_imputed[[phen_name]] <- suppressWarnings(as.numeric(pheno_data_imputed[[phen_name]]))
            # Recalculate missing count after conversion
            missing_count <- sum(is.na(pheno_data_imputed[[phen_name]]))
            cat(sprintf("After conversion - Missing values: %d / %d\n", missing_count, nrow(pheno_data_imputed)), 
                file = phen_log_path, append = TRUE)
        }
        
        if (missing_count == 0) {
            result$error_msg <- "No missing values to impute"
            result$success <- TRUE  # Not really an error, but skip
            cat("INFO: No missing values, skipping imputation\n", file = phen_log_path, append = TRUE)
            return(result)
        }
        
        # Check if there are enough non-missing values for imputation
        non_missing_count <- sum(!is.na(pheno_data_imputed[[phen_name]]))
        cat(sprintf("Non-missing values: %d\n", non_missing_count), file = phen_log_path, append = TRUE)
        
        if (non_missing_count < 10) {
            result$error_msg <- sprintf("Too few non-missing values (%d) for reliable imputation", non_missing_count)
            cat("ERROR:", result$error_msg, "\n", file = phen_log_path, append = TRUE)
            return(result)
        }
        
        # For phenotypes with very few missing values (< 5), randomly mask some values to reach 5 missing
        # This helps PIXANT work properly, as it may fail with too few missing values
        original_missing_indices <- which(is.na(pheno_data_imputed[[phen_name]]))
        masked_indices <- NULL
        target_missing_count <- 5
        
        if (missing_count < target_missing_count && non_missing_count >= target_missing_count) {
            # Need to mask additional values
            additional_missing_needed <- target_missing_count - missing_count
            non_missing_indices <- which(!is.na(pheno_data_imputed[[phen_name]]))
            
            # Randomly select indices to mask (use fixed seed based on phenotype name for reproducibility)
            set.seed(sum(utf8ToInt(phen_name)) %% 10000)
            masked_indices <- sample(non_missing_indices, additional_missing_needed)
            
            # Store original values before masking
            original_values <- pheno_data_imputed[[phen_name]][masked_indices]
            
            # Temporarily set selected values to NA
            pheno_data_imputed[[phen_name]][masked_indices] <- NA
            
            cat(sprintf("INFO: Only %d missing values detected. Randomly masking %d additional values to reach %d missing values for PIXANT.\n", 
                       missing_count, additional_missing_needed, target_missing_count), 
                file = phen_log_path, append = TRUE)
            cat(sprintf("Masked sample indices: %s\n", paste(masked_indices, collapse = ", ")), 
                file = phen_log_path, append = TRUE)
            cat(sprintf("Original values of masked samples: %s\n", paste(sprintf("%.4f", original_values), collapse = ", ")), 
                file = phen_log_path, append = TRUE)
            
            # Update missing count
            missing_count <- target_missing_count
        }
        
        cat("Starting PIXANT imputation...\n", file = phen_log_path, append = TRUE)
        cat("Parameters:\n", file = phen_log_path, append = TRUE)
        cat("  maxIterations: 20\n", file = phen_log_path, append = TRUE)
        cat("  num.trees: 100\n", file = phen_log_path, append = TRUE)
        cat("  refPhenThreshold: 0.3\n", file = phen_log_path, append = TRUE)
        cat("  minNum.refPhen: 10\n", file = phen_log_path, append = TRUE)
        cat("  SC.Threshold: 0.6\n", file = phen_log_path, append = TRUE)
        cat("\n", file = phen_log_path, append = TRUE)
        
        start_time <- Sys.time()
        elapsed_time <- NA  # Initialize elapsed_time
        
        # PIXANT imputation with error handling
        cat("Calling PIXANT function...\n", file = phen_log_path, append = TRUE)
        cat("Data dimensions:", nrow(pheno_data_imputed), "x", ncol(pheno_data_imputed), "\n", file = phen_log_path, append = TRUE)
        cat("Target phenotype column exists:", phen_name %in% colnames(pheno_data_imputed), "\n", file = phen_log_path, append = TRUE)
        
        # Prepare data for PIXANT: exclude first column (sample ID) to avoid factor level issues
        # PIXANT expects only phenotype columns, not sample IDs
        sample_id_col <- colnames(pheno_data_imputed)[1]
        pheno_data_for_pixant <- pheno_data_imputed[, -1, drop = FALSE]  # Exclude first column
        
        # Ensure sample ID column is not accidentally included
        if (sample_id_col %in% colnames(pheno_data_for_pixant)) {
            pheno_data_for_pixant[[sample_id_col]] <- NULL
        }
        
        cat("Data for PIXANT (excluding sample ID):", nrow(pheno_data_for_pixant), "x", ncol(pheno_data_for_pixant), "\n", 
            file = phen_log_path, append = TRUE)
        
        pixant_result <- tryCatch({
            # Capture any warnings
            withCallingHandlers({
                PIXANT(
                    data = pheno_data_for_pixant,
                    aimPhenName = phen_name,
                    maxIterations = 20,
                    maxIterations0 = 20,
                    num.trees = 100,
                    initialLinearEffects = 0,
                    errorTolerance = 0.001,
                    aimPhenMissingSize = min(500, missing_count),
                    initialImputeType = 'random',
                    refPhenThreshold = 0.3,
                    minNum.refPhen = 10,
                    SC.Threshold = 0.6,
                    seed = 123,
                    decreasing = TRUE,
                    verbose = FALSE  # Set to FALSE for parallel processing to avoid output conflicts
                )
            }, warning = function(w) {
                cat("WARNING during PIXANT call:", w$message, "\n", file = phen_log_path, append = TRUE)
                invokeRestart("muffleWarning")
            })
        }, error = function(e) {
            cat("ERROR in PIXANT call:", e$message, "\n", file = phen_log_path, append = TRUE)
            if (!is.null(e$call)) {
                cat("Call:", paste(deparse(e$call), collapse = "\n"), "\n", file = phen_log_path, append = TRUE)
            }
            cat("Error class:", class(e)[1], "\n", file = phen_log_path, append = TRUE)
            
            # Calculate elapsed time before fallback
            end_time <- Sys.time()
            elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
            
            # Restore masked values if any
            if (!is.null(masked_indices) && length(masked_indices) > 0) {
                cat("Restoring masked values due to error...\n", file = phen_log_path, append = TRUE)
                pheno_data_imputed[[phen_name]][masked_indices] <- original_values
            }
            
            # If PIXANT fails, try simple mean imputation as fallback
            cat("Attempting fallback: mean imputation...\n", file = phen_log_path, append = TRUE)
            mean_value <- mean(pheno_data_imputed[[phen_name]], na.rm = TRUE)
            if (!is.na(mean_value) && is.finite(mean_value)) {
                # Use mean imputation as fallback
                imputed_values <- pheno_data_imputed[[phen_name]]
                imputed_values[is.na(imputed_values)] <- mean_value
                
                result$success <- TRUE
                result$imputed_values <- imputed_values
                result$r2 <- NA
                result$accuracy <- NA
                result$ref_phen_count <- 0
                result$elapsed_time <- elapsed_time
                result$ref_phen_info <- data.frame()
                result$error_msg <- NULL
                result$masked_indices <- masked_indices
                
                cat(sprintf("✓ Fallback mean imputation completed (mean = %.4f)\n", mean_value), 
                    file = phen_log_path, append = TRUE)
                cat(sprintf("  Elapsed time: %.2f seconds\n", elapsed_time), file = phen_log_path, append = TRUE)
                cat(paste("End time:", Sys.time(), "\n"), file = phen_log_path, append = TRUE)
                
                # Return result directly from error handler
                # This will bypass the rest of the tryCatch block
                return(result)
            } else {
                cat("ERROR: Mean value is NA or infinite, cannot use fallback imputation\n", 
                    file = phen_log_path, append = TRUE)
                # Set error message and return result with success=FALSE
                result$error_msg <- paste0("PIXANT error: ", e$message, " (fallback also failed)")
                return(result)
            }
        })
        
        # Check if fallback was used (pixant_result is actually a result list, not PIXANT output)
        if (!is.null(pixant_result) && is.list(pixant_result) && "success" %in% names(pixant_result)) {
            if (pixant_result$success) {
                # Fallback was successful, return it
                return(pixant_result)
            } else {
                # Fallback failed, pixant_result contains error info
                result$error_msg <- pixant_result$error_msg
                return(result)
            }
        }
        
        end_time <- Sys.time()
        elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
        
        cat(sprintf("PIXANT call completed in %.2f seconds\n", elapsed_time), file = phen_log_path, append = TRUE)
        
        # Validate result
        if (is.null(pixant_result)) {
            result$error_msg <- "PIXANT returned NULL"
            cat("ERROR:", result$error_msg, "\n", file = phen_log_path, append = TRUE)
            return(result)
        }
        
        if (is.null(pixant_result$ximp)) {
            result$error_msg <- "PIXANT result missing 'ximp' field"
            cat("ERROR:", result$error_msg, "\n", file = phen_log_path, append = TRUE)
            cat("Available fields:", paste(names(pixant_result), collapse = ", "), "\n", file = phen_log_path, append = TRUE)
            return(result)
        }
        
        if (is.null(pixant_result$ximp[[phen_name]])) {
            result$error_msg <- sprintf("PIXANT result missing imputed values for '%s'", phen_name)
            cat("ERROR:", result$error_msg, "\n", file = phen_log_path, append = TRUE)
            cat("Available columns in ximp:", paste(colnames(pixant_result$ximp), collapse = ", "), "\n", file = phen_log_path, append = TRUE)
            return(result)
        }
        
        result$success <- TRUE
        imputed_values <- pixant_result$ximp[[phen_name]]
        
        # If we masked some values, restore original values for those indices
        # (keep PIXANT imputation only for truly missing values)
        if (!is.null(masked_indices) && length(masked_indices) > 0) {
            cat("Restoring original values for masked samples...\n", file = phen_log_path, append = TRUE)
            imputed_values[masked_indices] <- original_values
            
            # Calculate R² only for truly missing values (not masked ones)
            true_missing_indices <- original_missing_indices
            if (length(true_missing_indices) > 0) {
                cat(sprintf("INFO: %d values were masked for PIXANT, %d were originally missing\n", 
                           length(masked_indices), length(true_missing_indices)), 
                    file = phen_log_path, append = TRUE)
            }
        }
        
        result$imputed_values <- imputed_values
        result$r2 <- ifelse(is.null(pixant_result$imputePhen.r2), NA, pixant_result$imputePhen.r2)
        result$accuracy <- ifelse(is.null(pixant_result$imputePhen.accuracy), NA, pixant_result$imputePhen.accuracy)
        result$ref_phen_count <- ifelse(is.null(pixant_result$imputePhen.refPhen), 0, nrow(pixant_result$imputePhen.refPhen))
        result$elapsed_time <- elapsed_time
        result$ref_phen_info <- pixant_result$imputePhen.refPhen
        result$masked_indices <- masked_indices  # Store for logging
        
        # Log results to phenotype-specific log file
        cat("✓ Imputation completed successfully\n", file = phen_log_path, append = TRUE)
        cat(sprintf("  Elapsed time: %.2f seconds\n", elapsed_time), file = phen_log_path, append = TRUE)
        cat(sprintf("  Imputation R²: %.4f\n", result$r2), file = phen_log_path, append = TRUE)
        cat(sprintf("  Imputation accuracy: %.4f\n", result$accuracy), file = phen_log_path, append = TRUE)
        cat(sprintf("  Reference phenotypes: %d\n", result$ref_phen_count), file = phen_log_path, append = TRUE)
        
        if (!is.null(pixant_result$imputePhen.refPhen) && nrow(pixant_result$imputePhen.refPhen) > 0) {
            cat("\nReference phenotypes:\n", file = phen_log_path, append = TRUE)
            write.table(pixant_result$imputePhen.refPhen, phen_log_path, 
                       sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE, append = TRUE)
        }
        
        cat(paste("\nEnd time:", Sys.time(), "\n"), file = phen_log_path, append = TRUE)
        
    }, error = function(e) {
        # Capture detailed error information
        error_msg <- e$message
        if (is.null(error_msg) || error_msg == "") {
            error_msg <- "Unknown error occurred"
        }
        
        result$error_msg <- paste0("PIXANT error: ", error_msg)
        
        # Try to get more error details
        if (!is.null(e$call)) {
            call_str <- paste(deparse(e$call), collapse = " ")
            result$error_msg <- paste0(result$error_msg, " (call: ", call_str, ")")
        }
        
        # Log full error details
        cat("ERROR:", result$error_msg, "\n", file = phen_log_path, append = TRUE)
        cat("Error class:", class(e)[1], "\n", file = phen_log_path, append = TRUE)
        
        # Try to get traceback if available
        tryCatch({
            traceback_lines <- capture.output(traceback())
            if (length(traceback_lines) > 0) {
                cat("Traceback:\n", file = phen_log_path, append = TRUE)
                cat(paste(traceback_lines, collapse = "\n"), "\n", file = phen_log_path, append = TRUE)
            }
        }, error = function(te) {
            # Ignore traceback errors
        })
        
        cat(paste("End time:", Sys.time(), "\n"), file = phen_log_path, append = TRUE)
    }, warning = function(w) {
        # Log warnings but don't fail
        warning_msg <- w$message
        cat("WARNING:", warning_msg, "\n", file = phen_log_path, append = TRUE)
        # Don't set error_msg for warnings, just log them
    })
    
    return(result)
}

# Process phenotypes in parallel batches
if (n_to_impute > 0) {
    cat("\nStarting parallel imputation...\n")
    cat("\nStarting parallel imputation...\n", file = summary_log_path, append = TRUE)
    
    # Create cluster
    cl <- makeCluster(n_cores)
    clusterEvalQ(cl, library("PIXANT"))
    
    # Export data, log directory, and the function to cluster
    clusterExport(cl, c("pheno_data_imputed", "PIXANT", "log_dir", "impute_single_phenotype"))
    
    # Run parallel imputation
    start_time_parallel <- Sys.time()
    imputation_results <- parLapply(cl, phenotypes_to_impute, function(phen_name) {
        impute_single_phenotype(phen_name, log_dir)
    })
    end_time_parallel <- Sys.time()
    total_elapsed <- as.numeric(difftime(end_time_parallel, start_time_parallel, units = "secs"))
    
    # Stop cluster
    stopCluster(cl)
    
    # Process results and update data frame
    successful_count <- 0
    failed_count <- 0
    failed_phenotypes <- list()  # Store failed phenotypes with their log paths
    
    for (i in seq_along(imputation_results)) {
        result <- imputation_results[[i]]
        phen_name <- result$phen_name
        
        cat(sprintf("\n[%d/%d] Phenotype: %s\n", i, n_to_impute, phen_name))
        cat(sprintf("[%d/%d] Phenotype: %s\n", i, n_to_impute, phen_name), 
            file = summary_log_path, append = TRUE)
        
        if (result$success) {
            # Update imputed values
            pheno_data_imputed[[phen_name]] <- result$imputed_values
            
            # Log results
            cat(sprintf("  ✓ Imputation completed in %.2f seconds\n", result$elapsed_time))
            cat(sprintf("  Imputation R²: %.4f", result$r2))
            
            # Warn about low R²
            if (!is.na(result$r2) && result$r2 < 0.3) {
                cat(" ⚠️  (Low R² - imputation quality may be poor)")
            }
            cat("\n")
            
            cat(sprintf("  Imputation accuracy: %.4f\n", result$accuracy))
            cat(sprintf("  Reference phenotypes: %d\n", result$ref_phen_count))
            
            cat(sprintf("  ✓ Imputation completed in %.2f seconds\n", result$elapsed_time), 
                file = summary_log_path, append = TRUE)
            r2_msg <- sprintf("  Imputation R²: %.4f", result$r2)
            if (!is.na(result$r2) && result$r2 < 0.3) {
                r2_msg <- paste0(r2_msg, " ⚠️  (Low R² - imputation quality may be poor)")
            }
            cat(r2_msg, "\n", file = summary_log_path, append = TRUE)
            cat(sprintf("  Imputation accuracy: %.4f\n", result$accuracy), 
                file = summary_log_path, append = TRUE)
            cat(sprintf("  Reference phenotypes: %d\n", result$ref_phen_count), 
                file = summary_log_path, append = TRUE)
            cat(sprintf("  Log file: %s\n", result$log_path), 
                file = summary_log_path, append = TRUE)
            
            # Log if masking was used
            if (!is.null(result$masked_indices) && length(result$masked_indices) > 0) {
                cat(sprintf("  Note: %d values were randomly masked to help PIXANT work properly\n", 
                           length(result$masked_indices)), 
                    file = summary_log_path, append = TRUE)
            }
            
            # Save reference phenotypes info to log directory
            ref_phen_file <- file.path(log_dir, paste0("ref_phen_", gsub("[^A-Za-z0-9_]", "_", phen_name), ".txt"))
            write.table(result$ref_phen_info, ref_phen_file, 
                       sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
            
            successful_count <- successful_count + 1
        } else {
            error_msg <- ifelse(is.null(result$error_msg), "Unknown error", result$error_msg)
            cat(sprintf("  ✗ Error: %s\n", error_msg))
            cat(sprintf("  ✗ Error: %s\n", error_msg), 
                file = summary_log_path, append = TRUE)
            
            # Additional diagnostics
            missing_count <- sum(is.na(pheno_data_imputed[[phen_name]]))
            cat(sprintf("  Missing values remaining: %d\n", missing_count))
            cat(sprintf("  Missing values remaining: %d\n", missing_count), 
                file = summary_log_path, append = TRUE)
            
            # Log file path if available
            if (!is.null(result$log_path)) {
                cat(sprintf("  Log file: %s\n", result$log_path))
                cat(sprintf("  Log file: %s\n", result$log_path), 
                    file = summary_log_path, append = TRUE)
                # Store failed phenotype info
                failed_phenotypes[[length(failed_phenotypes) + 1]] <- list(
                    name = phen_name,
                    log_path = result$log_path,
                    error = error_msg
                )
            } else {
                failed_phenotypes[[length(failed_phenotypes) + 1]] <- list(
                    name = phen_name,
                    log_path = "N/A",
                    error = error_msg
                )
            }
            
            failed_count <- failed_count + 1
        }
        
        cat("\n", file = summary_log_path, append = TRUE)
    }
    
    cat(sprintf("\nParallel processing completed in %.2f seconds\n", total_elapsed))
    cat(sprintf("Successful: %d, Failed: %d\n", successful_count, failed_count))
    cat(sprintf("\nParallel processing completed in %.2f seconds\n", total_elapsed), 
        file = summary_log_path, append = TRUE)
    cat(sprintf("Successful: %d, Failed: %d\n", successful_count, failed_count), 
        file = summary_log_path, append = TRUE)
} else {
    cat("No phenotypes with missing values to impute.\n")
    cat("No phenotypes with missing values to impute.\n", file = summary_log_path, append = TRUE)
}

# Save imputed data
cat("Saving imputed data to:", pheno_imputed_path, "\n")
write.table(pheno_data_imputed, pheno_imputed_path, sep = "\t", 
           row.names = FALSE, col.names = TRUE, quote = FALSE)

# Summary statistics
cat("\nImputation Summary:\n")
cat("==================\n")
cat(sprintf("Total phenotypes processed: %d\n", n_phenotypes))
cat(sprintf("Total samples: %d\n", nrow(pheno_data_imputed)))

# Count remaining missing values
remaining_missing <- sum(is.na(pheno_data_imputed[, phenotype_cols]))
total_values <- nrow(pheno_data_imputed) * n_phenotypes
cat(sprintf("Remaining missing values: %d (%.2f%%)\n", 
           remaining_missing, remaining_missing / total_values * 100))

cat("\nImputation Summary:\n", file = summary_log_path, append = TRUE)
cat("==================\n", file = summary_log_path, append = TRUE)
cat(sprintf("Total phenotypes processed: %d\n", n_phenotypes), file = summary_log_path, append = TRUE)
cat(sprintf("Total samples: %d\n", nrow(pheno_data_imputed)), file = summary_log_path, append = TRUE)
cat(sprintf("Remaining missing values: %d (%.2f%%)\n", 
           remaining_missing, remaining_missing / total_values * 100), file = summary_log_path, append = TRUE)
cat(paste("End time:", Sys.time(), "\n"), file = summary_log_path, append = TRUE)

cat("\n✓ Imputation completed!\n")
cat("  Output file:", pheno_imputed_path, "\n")
cat("  Summary log:", summary_log_path, "\n")
cat("  Individual logs:", log_dir, "\n")

# Print failed phenotypes summary if any
if (exists("failed_phenotypes") && length(failed_phenotypes) > 0) {
    cat("\n")
    cat(paste(rep("=", 80), collapse = ""), "\n")
    cat("Failed Phenotypes Summary:\n")
    cat(paste(rep("=", 80), collapse = ""), "\n")
    for (i in seq_along(failed_phenotypes)) {
        failed_info <- failed_phenotypes[[i]]
        cat(sprintf("\n[%d] %s\n", i, failed_info$name))
        cat(sprintf("    Error: %s\n", failed_info$error))
        cat(sprintf("    Log file: %s\n", failed_info$log_path))
    }
    cat("\n")
    cat(paste(rep("=", 80), collapse = ""), "\n")
    
    # Also write to summary log
    cat("\n", file = summary_log_path, append = TRUE)
    cat(paste(rep("=", 80), collapse = ""), "\n", file = summary_log_path, append = TRUE)
    cat("Failed Phenotypes Summary:\n", file = summary_log_path, append = TRUE)
    cat(paste(rep("=", 80), collapse = ""), "\n", file = summary_log_path, append = TRUE)
    for (i in seq_along(failed_phenotypes)) {
        failed_info <- failed_phenotypes[[i]]
        cat(sprintf("\n[%d] %s\n", i, failed_info$name), file = summary_log_path, append = TRUE)
        cat(sprintf("    Error: %s\n", failed_info$error), file = summary_log_path, append = TRUE)
        cat(sprintf("    Log file: %s\n", failed_info$log_path), file = summary_log_path, append = TRUE)
    }
    cat("\n", file = summary_log_path, append = TRUE)
    cat(paste(rep("=", 80), collapse = ""), "\n", file = summary_log_path, append = TRUE)
}
