#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/cran/rrBLUP

script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1]]))
source(file.path(
  dirname(script_path), "..", "..", "r_models_common", "worker_utils.R"
))

require_packages(c("jsonlite", "rrBLUP"))
args <- parse_worker_args(commandArgs(trailingOnly = TRUE))
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(args$seed)

train_x <- read_numeric_matrix(args$train_x)
train_y <- read_numeric_vector(args$train_y)
predict_x <- read_numeric_matrix(args$predict_x)
parameters <- jsonlite::read_json(args$parameters, simplifyVector = TRUE)
validate_dimensions(train_x, train_y, predict_x)

method <- as.character(parameters$method)
if (!(method %in% c("REML", "ML"))) {
  stop("rrBLUP method must be REML or ML")
}
captured <- capture_warnings(rrBLUP::mixed.solve(
  y = train_y,
  Z = train_x,
  method = method
))
fit <- captured$value
beta <- as.numeric(fit$beta)
marker_effects <- as.numeric(fit$u)
predictions <- beta[[1]] + as.numeric(predict_x %*% marker_effects)
write_predictions(predictions, args$output_dir)
saveRDS(
  list(
    beta = beta,
    marker_effects = marker_effects,
    Vu = fit$Vu,
    Ve = fit$Ve,
    method = method
  ),
  file.path(args$output_dir, "model.rds")
)
write_json(
  list(
    method = method,
    beta = beta,
    Vu = as.numeric(fit$Vu),
    Ve = as.numeric(fit$Ve),
    warnings = captured$warnings,
    session = session_lines()
  ),
  file.path(args$output_dir, "worker_metadata.json")
)
