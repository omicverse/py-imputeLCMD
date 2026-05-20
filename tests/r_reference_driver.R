#!/usr/bin/env Rscript
# Drive the R imputeLCMD reference end-to-end on a TSV matrix dumped by
# the Python side. Used by tests/test_r_parity.py to verify bit/dist
# parity for impute.MinDet / impute.MinProb / impute.QRILC.
#
# Usage:
#   Rscript r_reference_driver.R <input_tsv> <out_dir> <method> [seed]
#
#   <input_tsv> : tab-separated proteins × samples matrix; NA marks missing
#   <out_dir>   : where the imputed TSV is written
#   <method>    : one of "mindet", "minprob", "qrilc"
#   [seed]      : optional R RNG seed (default 0); only used by MinProb/QRILC

suppressPackageStartupMessages({
  library(imputeLCMD)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript r_reference_driver.R input_tsv out_dir method [seed]")
input_tsv <- args[[1]]
out_dir   <- args[[2]]
method    <- tolower(args[[3]])
seed      <- if (length(args) >= 4) as.integer(args[[4]]) else 0L

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Read with NA = literal "NA" — matches pandas.to_csv(na_rep='NA').
X <- as.matrix(read.table(input_tsv, sep = "\t", header = TRUE,
                          row.names = 1, check.names = FALSE,
                          na.strings = c("NA", "")))

set.seed(seed)

t0 <- proc.time()[[3]]
if (method == "mindet") {
  out <- impute.MinDet(X, q = 0.01)
} else if (method == "minprob") {
  out <- impute.MinProb(X, q = 0.01, tune.sigma = 1.0)
} else if (method == "qrilc") {
  res <- impute.QRILC(X, tune.sigma = 1.0)
  # impute.QRILC returns an unnamed list: [[1]] = imputed matrix,
  # [[2]] = QR.model list (one lm() fit per sample). Names are
  # dropped in some imputeLCMD versions, so positional access is the
  # only safe approach.
  out <- res[[1]]
  qr_models <- res[[2]]
  n_s <- ncol(X)
  intercepts <- numeric(n_s)
  slopes <- numeric(n_s)
  for (k in seq_len(n_s)) {
    cf <- coef(qr_models[[k]])
    intercepts[k] <- as.numeric(cf[[1]])
    slopes[k]     <- as.numeric(cf[[2]])
  }
  coef_df <- data.frame(intercept = intercepts, slope = slopes,
                        row.names = colnames(X))
  write.table(coef_df,
              file = file.path(out_dir, "qrilc_coefs.tsv"),
              sep = "\t", quote = FALSE, col.names = NA)
} else {
  stop(sprintf("Unknown method: %s", method))
}
t1 <- proc.time()[[3]]

write.table(out,
            file = file.path(out_dir, sprintf("%s_imputed.tsv", method)),
            sep = "\t", quote = FALSE, col.names = NA)
cat(sprintf("R_%s_TIME=%.4f\n", toupper(method), t1 - t0))
cat(sprintf("R_%s_OUT=%s\n", toupper(method),
            file.path(out_dir, sprintf("%s_imputed.tsv", method))))
