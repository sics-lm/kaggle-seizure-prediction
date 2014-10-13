library(caret)
library("stringr")  #String regular expressions
source("correlation_convertion.r")


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


trainModel <- function(trainingData, method="glm") {
    ## Creates a model using caret based on the given dataframe

    ## Create a resampling control object, "repeatedcv" uses repeated k-fold cross-validation with k=number
    ctrl <- trainControl(method="repeatedcv", repeats=1, 10,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)


    classLabels <- factor(trainingData$Class)
    observations <- getChannelDF(trainingData)
    
    model <- train(x=observations,
                   y=classLabels,
                   method=method,
                   trControl=ctrl,
                   metric="ROC")
    model
}


trainModelBySegment <- function(trainingData,
                                number=10,
                                trainingRatio=.8,
                                method="plr") {
    ## Creates a model using caret based on the given dataframe
        trainIndice <- splitBySegment(trainingData,
                                      trainingRatio,
                                      number)
        
        ctrl <- trainControl(number=number,
                             index= trainIndice,
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary)

        classLabels <- factor(trainingData$Class) 
        observations <- getChannelDF(trainingData)

        ## Convenient list of models:
        ##         Penalized Logistic Regression
        ## method = 'plr'
        ## Type: Classification
        ## Tuning Parameters: lambda (L2 Penalty), cp (Complexity Parameter)
        ##         Quadratic Discriminant Analysis
        ## method = 'qda'
        ## Type: Classification
        ## No Tuning Parameters

        ##         ROC-Based Classifier
        ## method = 'rocc'
        ## Type: Classification
        ## Tuning Parameters: xgenes (#Variables Retained)

        ##         Neural Networks with Feature Extraction
        ## method = 'pcaNNet'
        ## Type: Classification, Regression
        ## Tuning Parameters: size (#Hidden Units), decay (Weight Decay)

        ## Generalized Linear Model
        ## method = 'glm'
        ## Type: Regression, Classification
        ## No Tuning Parameters
        
        model <- train(x=observations,
                       y=classLabels,
                       method=method,
                       trControl=ctrl,
                       metric="ROC")

        model
}


splitBySegment <- function(dataframe, trainingRatio=.8, number=3) {
    #splits the data according to segment. Returns a list of lists of indices to use for training
    segmentNames <- unique(dataframe[,c("segment", "Class")])
    trainIndice <- createDataPartition(segmentNames$Class, p=trainingRatio,times=number)
    trainSegments <- lapply(trainIndice,
                            FUN=function(indice) {
                                segmentNames[indice, ]$segment
                            })
    
    lapply(trainSegments, FUN=function(segments) {
        which(dataframe$segment %in% segments)
    })
}


calculateClassProbs <- function(model, testing) {
    ## Calculates the class probabilities given the *testing* dataset using the fitted *model*
    plsPreds <- predict(model, newdata = testing)#, type = "prob")
    return(plsPreds)
}


preictalRatio <- function(classifications) {
    ## Returns the number of "1" elements in the *classifications* vector
    n <- max(1, length(classifications))
    length(classifications[classifications == "Preictal"]) / n
}


probAverage <- function(classifications) {
    ## Returns the mean of the preictal guesses
    mean(classifications)
}


assignSegmentProbability <- function(model, testSegments, type="prob") {
    ## Returns an dataframe with the filenames for the segment features and counds of the assigned classes
    segmentNames <- testSegments$segment
    channelDF <- getChannelDF(testSegments)
    if (type=="prob") {
        guesses <- predict(model, newdata = channelDF, type=type)
        namedGuesses <- data.frame(preictal=guesses$Preictal,
                                   file=segmentNames)
        moltenPredictions <- melt(namedGuesses, id.vars = c("file"))
        segmentClassification <- dcast(moltenPredictions,
                                       file ~ variable,
                                       fun.aggregate=probAverage)
    }
    else {
        guesses <- predict(model, newdata = channelDF)
    
        namedGuesses <- data.frame(preictal=guesses, file=segmentNames)
        moltenPredictions <- melt(namedGuesses, id.vars = c("file"))
        segmentClassification <- dcast(moltenPredictions, file ~ variable,
                                       fun.aggregate=preictalRatio)
    }
    segmentClassification
}
