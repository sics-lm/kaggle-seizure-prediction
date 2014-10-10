source("correlation_convertion.r")
source("seizure_modeling.r")

runClassification <- function(featureFolder, rebuildData=FALSE, trainingRatio=.8, rebuildModel=FALSE, modelFile=NULL) {
    ## Runs the whole classification on the featureFolder
    ## Args:
    ##    featureFolder: a folder containing the feature csv files to train on
    ##    reuildData: Logic flag whether to rebuild the feature dataframes. If false, cached versions will be used if they exist
    ##    trainingRatio: the ratio of data to use for training the model. 1.0 uses almost all data for the training set (1 example be left in the test set)
    ##    modelFile: Name of a model RDS file to use. If NULL and rebuildModel is FALSE, the newest model file in featureFolder will be used. If NULL and rebuildModel is TRUE, a name will be generated based on the model label in caret and the time the model was built. 
    ##    rebuildModel: Logic flag whether to rebuild the model. If FALSE, the latest file with the suffix model_ in the feature folder will be used as the model
    ## Returns:
    ##    A list with (model, confMatrix, segmentClassification) for convenience. segmentClassification is a table of the original feature file names and a score based on how preictal they are. This data is automatically saved to a CSV file during the run of the function.
    ##
    
    dataSet <- loadDataFrames(featureFolder, rebuildData=rebuildData)
    interictal <- dataSet[[1]]
    preictal <- dataSet[[2]]
    unlabeledTests <- dataSet[[3]]

    ## For now this is just a single run, should probably be iterated
    experimentSplit <- splitExperimentData(interictal,
                                           preictal,
                                           trainingPerc=trainingRatio)
    trainingData <- experimentSplit[[1]]
    testData <- experimentSplit[[2]]

    if (is.null(modelFile)) {
        ## Read the first available model
        modelFiles <- file.path(featureFolder,
                                list.files(featureFolder,
                                           pattern="model_.*\\.rds"))
        if (length(modelFiles) > 0) {
            fileInfos <- file.info(modelFiles)
            fileDates <- data.frame(ctime=fileInfos$ctime,
                                    name=row.names(fileInfos),
                                    stringsAsFactors=FALSE)
            latestDate <- max(fileDates$ctime)
            modelFile <- fileDates[fileDates$ctime==latestDate,]$name[[1]]
        }
        else {
            rebuildModel <- TRUE
        }
    }

    timestamp <- format(Sys.time(), "%Y-%m-%d-%H:%M:%S")
    if (rebuildModel) {
        model <- trainModel(trainingData)

        if (is.null(modelFile)) {
            modelLabel <- model$modelInfo$label
            modelFileName <- sprintf("model_%s_%s.rds",
                                     modelLabel,
                                     timestamp)
            modelFile <- file.path(featureFolder, modelFileName)
        }
        saveRDS(model, modelFile)
    }
    else {
        model <- readRDS(modelFile)
    }

    segmentClassification <- assignSegmentProbability(model, unlabeledTests)
    ## ## Saving segment class probabilities ## ##
    probsFileName <- sprintf("classification_%s.csv", timestamp)
    segmentClassificationCSV <- file.path(featureFolder, probsFileName)
    write.csv(segmentClassification,
              file=segmentClassificationCSV,
              sep="\t", row.names=FALSE)
    ## ## Done saving segment class probabilities ## ##

    classProbs <- calculateClassProbs(model, testData)
    confMatrix <- tryCatch(
        confusionMatrix(data=classProbs, reference = testData$preictal),
        error = function(e) {
            NA
        }
    )

    list(model, confMatrix, segmentClassification)
}

