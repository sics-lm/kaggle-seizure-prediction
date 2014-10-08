source("correlation_convertion.r")
source("seizure_modeling.r")

runClassification <- function(featureFolder, rebuildData=FALSE, trainingRatio=.8) {
    ## Runs the whole classification on the featureFolder
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
    model <- trainModel(trainingData)

    ## ### Save the model to an rds ## ##
    modelLabel <- model$modelInfo$label
    modelTime <- format(Sys.time(), "%Y-%m-%d-%H:%M:%S")
    modelFileName <- sprintf("model_%s_%s.rds", modelLabel, modelTime)
    modelPath <- file.path(featureFolder, modelFileName)

    saveRDS(model, modelPath)
    ## ## Done saving model ## ##

    segmentClassification <- assignSegmentProbability(model, unlabeledTests)
    ## ## Saving segment class probabilities ## ##
    probsFileName <- sprintf("classification_%s.csv", modelTime)
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

