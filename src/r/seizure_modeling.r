library(caret)
library("stringr")  #String regular expressions



## calculateCombinedMeans <- function(interictal, preictal, test) {
##     colMeans(getChannelDF(rbind.fill(interictal, preictal, test)))
## }


calculateCombinedMeans <- function(interictal, preictal, test) {
    ## This is substantially faster than using colMeans on the rbound result of the frames. There seems to be some accuracy differences though, not sure which one is better
    (colSums(getChannelDF(interictal)) +
            colSums(getChannelDF(preictal)) +
                    colSums(getChannelDF(test))) /
                        (nrow(interictal)+nrow(preictal)+nrow(test))
}


calculateCombinedStd <- function(interictal, preictal, test) {
    apply(getChannelDF(rbind.fill(interictal, preictal, test)), 2, sd, na.rm = TRUE)
}


standardizeChannels <- function(df, center=TRUE, scale=TRUE) {
    ## Centers and scales the channel columns of the given dataframe
    df[,getChannelCols(df)] <- scale(df[,getChannelCols(df)],
                                     center=center,
                                     scale=scale)
    df
}

trainModel <- function(trainingData) {
    ## Creates a model using caret based on the given dataframe

    ## Create a resampling control object, "repeatedcv" uses repeated k-fold cross-validation with k=number
    ctrl <- trainControl(method="repeatedcv", repeats=1, 10,
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
                   method="glm",
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
