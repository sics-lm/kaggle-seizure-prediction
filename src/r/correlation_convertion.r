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
  pivot_subset$segment <- basename(filename)

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


loadCorrelationFiles <- function(featureFolder, className,
                                 filePattern=".*5\\.0s\\.csv$",
                                 cluster=NULL, rebuildData=FALSE) {
    ## Loads the files matched by filePattern from from featureFolder using cluster for creating the dataframes in parallell
    ## Arguments:
    ##    featureFolder: the folder containing correlation csv files
    ##    className: the name to use for matching csv files, usually "interictal", "preictal" or "test"
    ##    filePattern: a regular expression used to match the filenames apart from the class name
    ##    cluster: optionally a cluster object created by the parallel package
    ##    rebuildData: flag to toggle wether to use cached data for the dataframes or re-read the features from the files
    ## Returns:
    ##    A dataframe containing the data from the files matched by the pattern
    fullPattern <- sprintf("(%s)%s", className, filePattern)
    cachedFile <- file.path(featureFolder,
                            sprintf("%s_cache.rds", className))
    
    if (rebuildData || !file.exists(cachedFile)) {
        ## We rebuild the dataframes by reading the csv files
        corrFiles <- list.files(path=featureFolder,
                                full.names = TRUE,
                                pattern=fullPattern)
        if (is.null(cluster)) {
            corrList <- lapply(corrFiles, loadAndPivot)
        }
        else {
            corrList <- parLapply(cluster, corrFiles, loadAndPivot)
        }
        
        ## We can't really parallelize rbind, but using plyr makes it much faster
        corrDF <- rbind.fill(corrList)
        saveRDS(corrDF, cachedFile)
        return(corrDF)
    }
    else {
        ## We use the cached data
        
        corrDF <- readRDS(cachedFile)
        return(corrDF)
    }
}

loadDataFrames <- function(featureFolder,  no.cores = 4, rebuildData=FALSE) {
    ## Loads the dataframes from featureFolder into three seperate dataframes, interictal, preictal and test
    ## Args:
    ##    featureFolder: a path to a folder containing correlation feature csv files
    ##    no.cores: the number of cores to use for parallel execution
    ##    rebuildData: Flag for rebuilding data even if there is a cached version in the feature folder
    ## Returns:
    ##    A three element list with (interictal, preictal, test) dataframes. 
    
    filePattern <- ".*5\\.0s\\.csv$"

    cl <- makeCluster(getOption("cl.cores", no.cores))
    clusterEvalQ(cl, library(reshape2))
    pre.df <- loadCorrelationFiles(featureFolder,
                                   className="preictal",
                                   filePattern=filePattern,
                                   cl,
                                   rebuildData)
    int.df <- loadCorrelationFiles(featureFolder,
                                   className="interictal",
                                   filePattern=filePattern,
                                   cl,
                                   rebuildData)
    test.df <- loadCorrelationFiles(featureFolder,
                                    className="test",
                                    filePattern=filePattern,
                                    cl,
                                    rebuildData)
    stopCluster(cl)
    
    return(list(int.df, pre.df, test.df))
}


splitExperimentData <- function(interictalDF,
                                preictalDF,
                                trainingPerc = .8,
                                downSample=FALSE) {
    ## Creates a stratified sample of the concatenation of the given interictal and preictal dataframes. Adds a class label column.
    ## Args:
    ##    interictalDF: dataframe containing interictal samples
    ##    preictalDF: dataframe containing preictal samples
    ##    trainingPerc: percentage of the data to use for training
    ##    downSample: Logic flag of whether to downsample the data to equal class distributions
    ## Returns:
    ## A list with two dataframes, the first is the training dataset and the second a test dataset. The dataframes contains the same columns as the input dataframes, with an additional class column called 'preictal' which is 1 if the row is a preictal sample and 0 if it's not

    ## Assign class labels
    interictalDF$preictal = 0
    preictalDF$preictal = 1

    ## Combine dataframes
    comp.df <- rbind.fill(preictalDF, interictalDF)

    ## createDataPartition performs stratified sampling, attempting to keep the percentage of class
    ## examples in the original data consistent in the test and train data.
    ## Consider using createTimeSlices here as it specifically built for time series data
    ## See: http://topepo.github.io/caret/splitting.html
    if (downSample) {
        comp.df <- downSample(comp.df,
                              factor(comp.df$preictal),
                              list=FALSE)
    }

    train.index <- createDataPartition(comp.df$preictal,
                                       p = trainingPerc,
                                       list = FALSE,
                                       times = 1)
    
    comp.train <- comp.df[ train.index,]
    comp.test  <- comp.df[-train.index,]
    return(list(comp.train, comp.test))
}


create_experiment_data <- function(filepath, no.cores = 4, training.perc = .8, rebuildData=FALSE) {
  # Creates a split of the complete training data into a training and test set
  #
  # Args:
  #   filepath: The path to a folder containing correlation csv files
  #   no.cores: the number of cores to use for parallel execution
  #   training.perc: The percentage of the data to be used as training data
  #   rebuild_data: Flag whether the dataframes should be rebuildt, even if there is a cached version of the data in the filepath folder
  # Returns:
  #   A list containing the train and test splits of the data

  dataSet <- loadDataFrames(filepath, no.cores, rebuildData)
  int.df <- dataSet[[1]]
  pre.df <- dataSet[[2]]

  trainingSplit <- splitExperimentData(interictalDF = int.df,
                                       preictalDF = pre.df,
                                       trainingPerc = training.perc)
  return(trainingSplit)
}


removeCol <- function(df, colname) {
    return(removeCols(df, c(colname)))
}
removeCols <- function(df, colnames) {
    ## Convenience function for removing columns from a dataframe. Returns a new dataframe where the column with names in colnames are removed
    return(df[, !(names(df) %in% colnames)])
}
