# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/GooLey1025/Aquila-GS

parse_worker_args <- function(arguments) {
  names <- c(
    "--train-x", "--train-y", "--predict-x", "--parameters",
    "--output-dir", "--seed"
  )
  result <- list()
  index <- 1
  while (index <= length(arguments)) {
    key <- arguments[[index]]
    if (!(key %in% names) || index == length(arguments)) {
      stop(sprintf("Invalid worker argument: %s", key))
    }
    normalized_key <- gsub("-", "_", substring(key, 3), fixed = TRUE)
    result[[normalized_key]] <- arguments[[index + 1]]
    index <- index + 2
  }
  expected <- gsub("-", "_", substring(names, 3), fixed = TRUE)
  missing <- names[!expected %in% names(result)]
  if (length(missing) > 0) {
    stop(sprintf("Missing worker arguments: %s", paste(missing, collapse = ", ")))
  }
  result$seed <- as.integer(result$seed)
  result
}

require_packages <- function(packages) {
  missing <- packages[
    !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing) > 0) {
    stop(sprintf(
      "Missing required R package(s): %s",
      paste(missing, collapse = ", ")
    ))
  }
}

read_numeric_matrix <- function(path) {
  values <- as.matrix(utils::read.table(
    path,
    header = FALSE,
    sep = "\t",
    check.names = FALSE
  ))
  storage.mode(values) <- "double"
  values
}

read_numeric_vector <- function(path) {
  as.numeric(read_numeric_matrix(path))
}

write_json <- function(value, path) {
  jsonlite::write_json(
    value,
    path,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null",
    digits = NA
  )
}

write_predictions <- function(values, output_dir) {
  utils::write.table(
    as.numeric(values),
    file.path(output_dir, "predictions.tsv"),
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE,
    sep = "\t"
  )
}

capture_warnings <- function(expression) {
  warnings <- character()
  value <- withCallingHandlers(
    expression,
    warning = function(condition) {
      warnings <<- c(warnings, conditionMessage(condition))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = unique(warnings))
}

session_lines <- function() {
  capture.output(utils::sessionInfo())
}

validate_dimensions <- function(train_x, train_y, predict_x) {
  if (nrow(train_x) != length(train_y)) {
    stop("Training genotype rows do not match training targets")
  }
  if (ncol(train_x) != ncol(predict_x)) {
    stop("Training and prediction marker counts differ")
  }
  if (any(!is.finite(train_x)) || any(!is.finite(predict_x))) {
    stop("Non-finite genotype value reached the R worker")
  }
  if (any(!is.finite(train_y)) || any(train_y == -999)) {
    stop("Invalid phenotype value reached the R worker")
  }
}
