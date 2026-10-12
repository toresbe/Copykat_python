#!/usr/bin/env Rscript
# Run CopyKAT-R unmodified and save the intermediate result of each step.
#
# Usage: run_copykat.R INPUT_DIR OUTPUT_DIR N_CORES [R_LIBRARY]
#
# INPUT_DIR holds matrix.mtx (genes x cells counts), genes.txt and barcodes.txt.
# R_LIBRARY, if given, is searched first for the copykat package (to run another
# release side by side, e.g. 1.1.0 installed with R CMD INSTALL -l R_LIBRARY).
# The package itself is not patched: trace() inserts "save this variable" calls
# into the running copy of copykat(), and everything else is CopyKAT's own code
# and its own output files. Stages are written to OUTPUT_DIR/stages/ in the
# format described in verify/stages.md, which the Python side writes too.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 4) .libPaths(c(args[4], .libPaths()))
suppressMessages({ library(copykat); library(Matrix) })
input <- normalizePath(args[1])
out <- args[2]
n_cores <- as.integer(args[3])
dir.create(out, recursive = TRUE, showWarnings = FALSE)
setwd(out)

counts <- readMM(file.path(input, "matrix.mtx"))
rownames(counts) <- readLines(file.path(input, "genes.txt"))
colnames(counts) <- readLines(file.path(input, "barcodes.txt"))

# ---------------------------------------------------------------------------
# save_stage(path, value): numbers as raw little-endian float64 (exact), text as
# one entry per line, lists as subdirectories. Row/column names go alongside.
# ---------------------------------------------------------------------------
save_stage <- function(path, value) {
  path <- file.path("stages", path)
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  if (is.list(value) && !is.data.frame(value)) {
    for (name in names(value)) {
      if (!is.language(value[[name]]) && !is.null(value[[name]])) {
        save_stage(file.path(sub("^stages/", "", path), name), value[[name]])
      }
    }
    return(invisible())
  }
  if (is.data.frame(value)) value <- as.matrix(value[, vapply(value, is.numeric, TRUE), drop = FALSE])
  if (is.factor(value)) value <- as.character(value)
  if (is.character(value)) {
    writeLines(value, file.path(path, "values.txt"))
  } else {
    writeBin(as.double(value), file.path(path, "values.f64"), size = 8, endian = "little")
  }
  writeLines(paste(if (is.null(dim(value))) length(value) else dim(value), collapse = " "),
             file.path(path, "shape.txt"))
  if (!is.null(dim(value))) {
    if (!is.null(rownames(value))) writeLines(rownames(value), file.path(path, "rownames.txt"))
    if (!is.null(colnames(value))) writeLines(colnames(value), file.path(path, "colnames.txt"))
  } else if (!is.null(names(value))) {
    writeLines(names(value), file.path(path, "names.txt"))
  }
}

ns <- asNamespace("copykat")

# Segmentation can run up to three times (KS.cut retries); keep every call.
segmentation_call <- 0
trace("CNA.MCMC", where = ns, print = FALSE,
      tracer = quote(segmentation_call <<- segmentation_call + 1),
      exit = quote(save_stage(sprintf("segmentation_%d", segmentation_call),
                              list(breaks = returnValue()$breaks, logCNA = returnValue()$logCNA,
                                   cut.cor = cut.cor))))

trace("convert.all.bins.hg20", where = ns, print = FALSE,
      exit = quote(save_stage("bins", returnValue()$RNA.adj)))

# Local variables of copykat() itself: just before step 4 and step 5, and on exit.
# (One trace() call: a second call on the same function would replace the first.)
statements <- as.list(body(copykat))
before <- function(label) which(vapply(statements, function(s) any(grepl(label, deparse(s), fixed = TRUE)), TRUE))[1]
trace(copykat, where = ns, print = FALSE,
      at = c(before('print("step 4'), before('print("step 5')),
      tracer = quote(
        if (!exists("norm.mat.relat", inherits = FALSE)) {
          save_stage("smoothed", norm.mat.smooth)
        } else {
          save_stage("relative", list(norm.mat.relat = norm.mat.relat, CL = CL, preN = preN, WNS = WNS,
                                      basel = if (exists("basel", inherits = FALSE)) basel))
        }),
      exit = quote({
        consensus_correlation <- if (exists("conses.diploid", inherits = FALSE)) cor(conses.diploid, conses.aneuploid)
        # R 1.1.0 reuses hc.umap for step 7, so only 1.2.5 has step-7 clusters at the end.
        step7 <- if (exists("hc.short", inherits = FALSE)) hc.short
        save_stage("final", list(mat.adj = mat.adj, step7_clusters = step7, step8_clusters = hc.umap,
                                 prediction = com.preN, consensus_correlation = consensus_correlation,
                                 WNS = WNS))
      }))

result <- copykat::copykat(rawmat = as.matrix(counts), id.type = "S", sam.name = "sample",
                           genome = "hg20", n.cores = n_cores, plot.genes = "FALSE")
save_stage("final_tree", result$hclustering[c("merge", "height", "order")])
writeLines(c(paste("copykat", packageVersion("copykat")), capture.output(sessionInfo())), "sessionInfo.txt")
