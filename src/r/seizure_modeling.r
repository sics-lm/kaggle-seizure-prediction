library(caret)
library("stringr")  #String regular expressions

trainModel <- function(trainingData) {
    ## Creates a model using caret based on the given dataframe

    ## Create a resampling control object, "repeatedcv" uses repeated k-fold cross-validation with k=number
    ctrl <- trainControl(method="repeatedcv", repeats=4, number=5,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)


    classLabels <- trainingData$preictal  # Only the column preictal
    observations <- trainingData[,!(names(trainingData) %in% c("preictal", "segment"))]  # All columns except preictal and segment
    
    ## Train the model
    ## Penalized Logistic Regression
    ## model <- train(x=observations,
    ##                y=classLabels,
    ##                method="plr",
    ##                trControl=ctrl,
    ##                metric="ROC")

    ##Decision Tree classifier
    classLabels <- factor(classLabels)  # The class labels needs to be factors for the decision tree
    model <- train(x=observations,
                   y=classLabels,
                   method="rpart",
                   trControl=ctrl,
                   metric="ROC")
    return(model)
}


calculateClassProbs <- function(model, testing) {
    ## Calculates the class probabilities given the *testing* dataset using the fitted *model*
    plsPreds <- predict(model, newdata = testing)#, type = "prob")
    return(plsPreds)
}


preictalRatio <- function(classifications) {
    ## Returns the number of "1" elements in the *classifications* vector
    n <- max(1, length(classifications))
    length(classifications[classifications == "1"]) / n
}
    
assignSegmentProbability <- function(model, testSegments) {
    ## Returns an dataframe with the filenames for the segment features and counds of the assigned classes
    guesses <- predict(model, newdata = testSegments)
    
    segmentNames <- testSegments$segment
    namedGuesses <- data.frame(preictal=guesses, file=segmentNames)
    moltenPredictions <- melt(namedGuesses, id.vars = c("file"))
    segmentClassification <- dcast(moltenPredictions, file ~ variable,
                                   fun.aggregate=preictalRatio)
    segmentClassification
}
