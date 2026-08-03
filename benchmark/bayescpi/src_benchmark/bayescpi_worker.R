#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/YinLiLin/hibayes

script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1]]))
source(file.path(
  dirname(script_path), "..", "..", "r_models_common", "worker_utils.R"
))

require_packages(c("jsonlite", "hibayes"))
args <- parse_worker_args(commandArgs(trailingOnly = TRUE))
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(args$seed)

train_x <- read_numeric_matrix(args$train_x)
train_y <- read_numeric_vector(args$train_y)
predict_x <- read_numeric_matrix(args$predict_x)
parameters <- jsonlite::read_json(args$parameters, simplifyVector = TRUE)
validate_dimensions(train_x, train_y, predict_x)

niter <- as.integer(parameters$niter)
nburn <- as.integer(parameters$nburn)
thin <- as.integer(parameters$thin)
threads <- as.integer(parameters$threads)
printfreq <- as.integer(parameters$printfreq)
verbose <- isTRUE(parameters$verbose)
if (niter <= 0 || nburn < 0 || nburn >= niter || thin <= 0) {
  stop("BayesCpi requires niter > nburn >= 0 and thin > 0")
}
if (threads <= 0 || printfreq <= 0) {
  stop("BayesCpi threads and printfreq must be positive")
}

sample_ids <- sprintf("sample_%d", seq_len(nrow(train_x)))
phenotype <- data.frame(
  id = sample_ids,
  y = train_y,
  stringsAsFactors = FALSE
)
captured <- capture_warnings(hibayes::ibrm(
  formula = y ~ 1,
  data = phenotype,
  M = train_x,
  M.id = sample_ids,
  method = "BayesCpi",
  niter = niter,
  nburn = nburn,
  thin = thin,
  printfreq = printfreq,
  seed = args$seed,
  threads = threads,
  verbose = verbose
))
fit <- captured$value
marker_effects <- as.numeric(fit$alpha)
intercept <- as.numeric(fit$mu)[[1]]
if (length(marker_effects) != ncol(train_x)) {
  stop("hibayes returned an unexpected marker-effect vector")
}
predictions <- intercept + as.numeric(predict_x %*% marker_effects)
write_predictions(predictions, args$output_dir)
saveRDS(fit, file.path(args$output_dir, "model.rds"))

posterior_names <- intersect(
  c("mu", "alpha", "Pi", "vg", "ve", "h2", "Vg", "Ve"),
  names(fit)
)
posterior <- lapply(posterior_names, function(name) {
  value <- fit[[name]]
  list(
    name = name,
    class = class(value),
    dimensions = dim(value),
    length = length(value),
    mean = if (is.numeric(value)) mean(value, na.rm = TRUE) else NULL
  )
})
names(posterior) <- posterior_names
write_json(
  list(
    method = "BayesCpi",
    parameters = list(
      niter = niter,
      nburn = nburn,
      thin = thin,
      printfreq = printfreq,
      seed = args$seed,
      threads = threads,
      verbose = verbose
    ),
    posterior = posterior,
    warnings = captured$warnings,
    session = session_lines()
  ),
  file.path(args$output_dir, "worker_metadata.json")
)
