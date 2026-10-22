suppressMessages(library(mclust))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4) {
  stop("Usage: mclust_batch.R embedding.csv tasks.csv index_dir output_dir")
}

embedding_path <- args[[1]]
tasks_path <- args[[2]]
index_dir <- args[[3]]
output_dir <- args[[4]]

x <- as.matrix(read.csv(embedding_path, header=FALSE, check.names=FALSE))
tasks <- read.csv(tasks_path, stringsAsFactors=FALSE)

for (row_id in seq_len(nrow(tasks))) {
  task_id <- tasks$task_id[[row_id]]
  k <- as.integer(tasks$k[[row_id]])
  seed <- as.integer(tasks$seed[[row_id]])
  cat(sprintf("MCLUST_PROGRESS %d/%d %s K=%d seed=%d\n", row_id, nrow(tasks), task_id, k, seed))
  flush.console()
  index_path <- file.path(index_dir, paste0(task_id, ".txt"))
  indices <- scan(index_path, what=integer(), quiet=TRUE) + 1L
  if (length(indices) < k) {
    stop(paste("Task has fewer rows than clusters:", task_id))
  }
  set.seed(seed)
  fit <- Mclust(x[indices, , drop=FALSE], G=k, modelNames="EEE", verbose=FALSE)
  if (is.null(fit$classification) || length(fit$classification) != length(indices)) {
    stop(paste("Invalid mclust classification:", task_id))
  }
  output_path <- file.path(output_dir, paste0(task_id, ".csv"))
  write.table(fit$classification, output_path, row.names=FALSE, col.names=FALSE, sep=",")
}
