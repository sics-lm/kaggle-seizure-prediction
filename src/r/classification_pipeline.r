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
    fit <- trainModel(trainingData)
    classProbs <- calculateClassProbs(fit, testData)
    confMatrix <- confusionMatrix(data=classProbs, reference = testData$preictal)
    unlabeledProbs <- runTests(fit, unlabeledTests)
    list(fit, confMatrix, unlabeledProbs)
}



