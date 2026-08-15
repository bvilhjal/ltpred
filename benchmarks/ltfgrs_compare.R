#!/usr/bin/env Rscript
# Locked LTFGRS Pearson-Aitken pass on a threshold table written by
# bench_ltfhplus_compare.py. LTFHPlus 2.2.0 has no PA path; LTFGRS is the
# public PA package from the same authors (method = "PA", useMixture = FALSE).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("usage: ltfgrs_compare.R IN.csv OUT.csv h2 tol workers [method]")
}
in_csv <- args[[1]]
out_csv <- args[[2]]
h2 <- as.numeric(args[[3]])
tol <- as.numeric(args[[4]])
workers <- as.integer(args[[5]])
method <- if (length(args) >= 6L) args[[6]] else "PA"

if (!requireNamespace("LTFGRS", quietly = TRUE)) {
  stop("LTFGRS is not installed. From R: install.packages('LTFGRS')")
}
suppressPackageStartupMessages({
  library(LTFGRS)
  library(future)
})

if (workers <= 1L) {
  plan(sequential)
} else {
  plan(multisession, workers = workers)
}
on.exit(plan(sequential), add = TRUE)

tbl <- read.csv(in_csv, stringsAsFactors = FALSE)
tbl$lower <- as.numeric(tbl$lower)
tbl$upper <- as.numeric(tbl$upper)

t0 <- proc.time()[["elapsed"]]
est <- estimate_liability(
  .tbl = tbl,
  h2 = h2,
  pid = "indiv_ID",
  fid = "fam_ID",
  role = "role",
  out = c("genetic"),
  tol = tol,
  method = method,
  useMixture = FALSE
)
elapsed <- proc.time()[["elapsed"]] - t0

# LTFGRS PA returns columns est / var; Gibbs may use genetic_est / genetic_se.
if ("est" %in% names(est)) {
  genetic_est <- est$est
  genetic_se <- if ("var" %in% names(est)) sqrt(pmax(est$var, 0)) else NA_real_
} else {
  genetic_est <- est$genetic_est
  genetic_se <- est$genetic_se
}

out <- data.frame(
  fam_ID = est[[1]],
  genetic_est = genetic_est,
  genetic_se = genetic_se,
  seconds = elapsed,
  workers = future::nbrOfWorkers(),
  ltfgrs_version = as.character(utils::packageVersion("LTFGRS")),
  method = method,
  stringsAsFactors = FALSE
)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("LTFGRS %s  method=%s  workers=%s  families=%d  seconds=%.4f\n",
            out$ltfgrs_version[1], method, out$workers[1], nrow(out), elapsed))
