#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/cran/glmnet

script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1]]))
source(file.path(
  dirname(script_path), "..", "..", "r_models_common", "worker_utils.R"
))

require_packages(c("jsonlite", "glmnet"))
args <- parse_worker_args(commandArgs(trailingOnly = TRUE))
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(args$seed)

train_x <- read_numeric_matrix(args$train_x)
train_y <- read_numeric_vector(args$train_y)
predict_x <- read_numeric_matrix(args$predict_x)
parameters <- jsonlite::read_json(args$parameters, simplifyVector = TRUE)
validate_dimensions(train_x, train_y, predict_x)

lambda <- as.numeric(parameters$lambda)
if (length(lambda) != 1 || !is.finite(lambda) || lambda <= 0) {
  stop("Lasso lambda must be one positive finite value")
}
captured <- capture_warnings(glmnet::glmnet(
  x = train_x,
  y = train_y,
  family = "gaussian",
  alpha = 1,
  lambda = lambda,
  standardize = isTRUE(parameters$standardize),
  intercept = !identical(parameters$intercept, FALSE)
))
fit <- captured$value
predictions <- as.numeric(stats::predict(fit, newx = predict_x, s = lambda))
write_predictions(predictions, args$output_dir)
saveRDS(fit, file.path(args$output_dir, "model.rds"))
write_json(
  list(
    alpha = 1,
    lambda = lambda,
    warnings = captured$warnings,
    session = session_lines()
  ),
  file.path(args$output_dir, "worker_metadata.json")
)
