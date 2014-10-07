library(caret)

trainModel <- function(trainingData) {
    ## Creates a model using caret based on the given dataframe

    ## Create a resampling control object, "repeatedcv" uses repeated k-fold cross-validation with k=number
    ctrl <- trainControl(method="repeatedcv", repeats=4, number=5,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)


    classLabels <- trainingData$preictal  # Only the column preictal
    observations <- trainingData[,!(names(trainingData) %in% c("preictal"))]  # All columns except preictal
    
    ## Train the model
    ## Penalized Logistic Regression
    ## model <- train(x=observations,
    ##                y=classLabels,
    ##                method="plr",
    ##                trControl=ctrl,
    ##                metric="ROC")

    ##Decision Tree classifier
    classLabels <- ordered(classLabels)  # The class labels needs to be factors for the decision tree
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


runTests <- function(model, testSegments) {
    ## Returns an array containing the file names of testSegments and the preictal probabilities
    plsProbs <- predict(model, newdata = testSegments)
#    return(cbind(row.names(testSegments), plsProbs))
    plsProbs
}
