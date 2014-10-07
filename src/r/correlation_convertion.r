# sourceDirectory(".", modifiedOnly=TRUE); # Source dir, changed files

library(reshape2)
library(parallel)
library(plyr)
library(caret)

loadAndPivot <- function(filename) {
  # Loads and pivots one correlation csv file.
  # Why reshape2 > reshape: https://stat.ethz.ch/pipermail/r-packages/2010/001169.html
  # We get a ~4x speedup using reshape2 here
  # Args:
  #   filename: The name of the file to read
  #
  # Returns:
  #   A pivoted dataframe.
  a <- read.csv(filename, stringsAsFactors = FALSE, sep = "\t")

  a$channel_pair <- paste(a$channel_i, a$channel_j, sep=":")
  subset <- a[,c("channel_pair", "start_sample", "correlation")]
  melted_subset <- melt(subset, id.vars = c("start_sample", "channel_pair"))
  pivot_subset <- dcast(data = melted_subset, start_sample ~ channel_pair)
  row.names(pivot_subset) <- paste(row.names(pivot_subset), basename(filename), sep=":")

  return(pivot_subset)
}


convert_csv_files <- function(file_prefix, type, start_number, end_number) {
  #  Combines and pivots a number of correlation files and saves them to a new r data file.
  #
  # Args:
  #   file_prefix : A prefix to apply both when reading and writing files.
  #   type: "interictal" or "preictal".
  #   start_number: Start segment
  #   end_number: End segment
  .Deprecated("create_experiment_data")
  ## Use create_experiment_data to create data partitions. Need to manually export to file though
  num <- start_number:end_number
  files <-  paste(file_prefix, type, "_segment_", formatC(num, width=4, flag=0),
    "_cross_correlation_5.0s.csv", sep = "")
  bigDf <- do.call("rbind", lapply(files, loadAndPivot))

  saveRDS(bigDf, paste(file_prefix, type, ".rds", sep=""))
}

create_experiment_data <- function(filepath, no.cores = 4, training.perc = .8) {
  # Creates a split of the complete training data into a training and test set
  #
  # Args:
  #   filepath: The path under which the interictal and preictal files lie
  #   no.cores: the number of cores to use for parallel execution
  #   training.perc: The percentage of the data to be used as training data
  # Returns:
  #   A list containing the train and test splits of the data


  cl <- makeCluster(getOption("cl.cores", no.cores))
  clusterEvalQ(cl, library(reshape2))

  pre.files <- list.files(path=filepath, full.names = TRUE, pattern="(preictal).*5\\.0s\\.csv$")
  int.files <- list.files(path=filepath, full.names = TRUE, pattern="(interictal).*5\\.0s\\.csv$")

  # Create the list of dfs in parallel
  pre.list <- parLapply(cl, pre.files, loadAndPivot)
  int.list <- parLapply(cl, int.files, loadAndPivot)

  stopCluster(cl)

  # We can't really parallelize rbind, but using plyr makes it much faster
  pre.df <- rbind.fill(pre.list)
  int.df <- rbind.fill(int.list)

  # Assign class labels
  pre.df$preictal = 1
  int.df$preictal = 0

  # Combine dataframes
  comp.df <- rbind.fill(pre.df, int.df)

  # createDataPartition performs stratified sampling, attempting to keep the percentage of class
  # examples in the original data consistent in the test and train data.
  # Consider using createTimeSlices here as it specifically built for time series data
  # See: http://topepo.github.io/caret/splitting.html
  train.index <- createDataPartition(comp.df$preictal, p = training.perc, list = FALSE, times = 1)

  comp.train <- comp.df[ train.index,]
  comp.test  <- comp.df[-train.index,]

  return(list(comp.train, comp.test))
}