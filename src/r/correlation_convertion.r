
loadAndPivot <- function(filename) {
  # Loads and pivots one correlation csv file.
  #
  # Args:
  #   filename: The name of the file to read
  #
  # Returns:
  #   A pivoted dataframe.
  a <- read.csv(filename, stringsAsFactors = FALSE, sep = "\t")
  
  a$channel_pair <- paste(a$channel_i, a$channel_j)
  
  subset <- a[,c("channel_pair", "start_sample", "correlation")]
  pivot_subset <- cast(subset, start_sample ~ channel_pair, value = "correlation")
  return(pivot_subset)
}


convert_csv_files <- function(file_prefix, type, start_number, end_number) {
  # Combines and pivots a number of correlation files and saves them to a new r data file.
  #
  # Args:
  #   file_prefix : A prefix to apply both when reading and writing files.
  #   type: "interictal" or "preictal".
  #   start_number: Start segment
  #   end_number: End segment
  num <- start_number:end_number
  files <-  paste(file_prefix, type, "_segment_", formatC(num, width=4, flag=0), "_cross_correlation_5.0s.csv", sep = "")
  bigDf <- do.call("rbind", lapply(files, loadAndPivot))

  saveRDS(bigDf, paste(file_prefix, type, ".rds", sep=""))
}