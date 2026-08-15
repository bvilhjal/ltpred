#!/usr/bin/env Rscript
# Locked LTFHPlus Gibbs pass on a threshold table written by
# bench_ltfhplus_compare.py. Not a standalone simulator: Python owns the
# families so both packages see identical C_F.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("usage: ltfhplus_compare.R IN.csv OUT.csv h2 tol workers")
}
in_csv <- args[[1]]
out_csv <- args[[2]]
h2 <- as.numeric(args[[3]])
tol <- as.numeric(args[[4]])
workers <- as.integer(args[[5]])

if (!requireNamespace("LTFHPlus", quietly = TRUE)) {
  stop("LTFHPlus is not installed. From R: ",
       "install.packages('LTFHPlus') or remotes::install_github('EmilMiP/LTFHPlus')")
}
suppressPackageStartupMessages({
  library(LTFHPlus)
  library(future)
})

if (workers <= 1L) {
  plan(sequential)
} else {
  plan(multisession, workers = workers)
}
on.exit(plan(sequential), add = TRUE)

tbl <- read.csv(in_csv, stringsAsFactors = FALSE)
# R reads "Inf"/"-Inf"; keep them numeric.
tbl$lower <- as.numeric(tbl$lower)
tbl$upper <- as.numeric(tbl$upper)

t0 <- proc.time()[["elapsed"]]
est <- estimate_liability(
  .tbl = tbl,
  h2 = h2,
  pid = "indiv_ID",
  fam_id = "fam_ID",
  role = "role",
  out = c("genetic"),
  tol = tol
)
elapsed <- proc.time()[["elapsed"]] - t0

out <- data.frame(
  fam_ID = est$fam_ID,
  genetic_est = est$genetic_est,
  genetic_se = est$genetic_se,
  seconds = elapsed,
  workers = future::nbrOfWorkers(),
  ltfhplus_version = as.character(utils::packageVersion("LTFHPlus")),
  stringsAsFactors = FALSE
)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("LTFHPlus %s  workers=%s  families=%d  seconds=%.3f\n",
            out$ltfhplus_version[1], out$workers[1], nrow(out), elapsed))
