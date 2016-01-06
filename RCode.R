# Alessandro Sisti
# Springleaf Marketing Challenge: A Comparison of Three Machine Learning Models
# (Capstone Project for Ryerson's Big Data Certificate)
# Submitted 5 January 2016

library("caret") # for preprocessing, confusion matrix
library("mlbench") # machine learning for filling in missing values
library("ada")  # for adaboost
library("plyr")  # also for adaboost
library("xgboost")  # for xgboost
library("pROC") # for plotting ROC curves

# Load 30 percent of full training set and store it as training
# Load a different 30 percent of the training set and use it for preliminary
# testing before the full test set
trainingfull <- read.csv("~/R/RyersonCapstone/train.csv", strip.white = TRUE)
sampleSize <- round(0.3 * nrow(trainingfull))

set.seed(22221)
inTrain <- sort(sample(nrow(trainingfull), size = sampleSize))
training <- trainingfull[inTrain, ]
# Eliminate the rows in the training set from consideration in the provisional
# test set
trainingfull <- trainingfull[-inTrain, ]
inTest <- sort(sample(nrow(trainingfull), size = sampleSize))
test <- trainingfull[inTest, ]

## For my use: write smaller training and provisional test sets to files
#write.csv(training, file = "~/R/RyersonCapstone/trainingCapstone.csv")
#write.csv(test, file = "~/R/RyersonCapstone/prelimTestCapstone.csv")

## read the files, when needed
#training <- read.csv("~/R/RyersonCapstone/trainingCapstone.csv")
#test <- read.csv("~/R/RyersonCapstone/prelimTestCapstone.csv")


### Early cleaning

# Extract numeric variables
trainingNumeric <- training[, lapply(training, is.numeric) == TRUE]

# Remove the first two columns, which would interfere with analysis
trainingNumeric <- trainingNumeric[, -1:-2]

# Remove variables with zero variance
trainingNumeric <- trainingNumeric[, lapply(trainingNumeric, var, na.rm = T)!=0]

# Remove variables whose correlation with the target is less than 
# an arbitrary threshold of 0.1
corrs <- numeric()
for (column in 1:(ncol(trainingNumeric) - 1)) {
    corrs[column] <- cor(trainingNumeric[, column], trainingNumeric[, "target"],
                         use = "pairwise.complete.obs")
}
stronglyCorrelated <- (corrs >= 0.1)
target <- trainingNumeric$target  # Set this aside to reattach it afterwards
trainingNumeric <- trainingNumeric[, -ncol(trainingNumeric)]
trainingNumeric <- trainingNumeric[, stronglyCorrelated]
trainingNumeric <- cbind(trainingNumeric, target = target) # reattach target


# Find the cleanest data: remove columns where there are any NAs in data
NAvalues <- numeric()
for (column in 1:ncol(trainingNumeric)) {
    NAvalues[column] <- sum(is.na(trainingNumeric[, column]))
}
clean <- (NAvalues == 0)
trainingNumericClean <- trainingNumeric[, clean]
# Turn target column into factor
# (factor observations need to be renamed for training algorithm to work)
replacement_target <- rep("Yes", nrow(trainingNumeric))
replacement_target[trainingNumeric[, ncol(trainingNumeric)] == "0"] <- "No"
trainingNumericClean[, "target"] <- as.factor(replacement_target)



### Start making models

### Prepare evaluation in the caret package: we'll use 10-fold cross-validation

ctrl <- trainControl(method = "cv",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary
)

# Prepare vector of areas under curves ROC curves from models tested
# (this will be used to compare models after all evaluations are done)
AUCurves <- rep(0, 3)
names(AUCurves) <- c("Random Forest", "Adaboost", "XGBoost")

# Prepare a preliminary testing set with the same variables as the training set
# and the target stored as a factor.
goodVariables <- names(trainingNumericClean)
testEvaluate <- test[, goodVariables]
replacement_target_test <- rep("Yes", nrow(testEvaluate))
replacement_target_test[testEvaluate[, ncol(testEvaluate)] == "0"] <- "No"
testEvaluate[, "target"] <- as.factor(replacement_target_test)

# Random forest
set.seed(6666)
rForestFit <- train(target ~ .,
                    data = trainingNumericClean,
                    method = "rf",
                    trControl = ctrl,
                    metric = "ROC")
# Save model to external file (for possible use with validation set)
save(rForestFit, file = "~/R/RyersonCapstone/rForestFit")
# Evaluation

# Get AUC result on test data
testResults <- predict(rForestFit, testEvaluate, type = "prob")
testResults$obs <- testEvaluate$target
testResults$pred <- predict(rForestFit, testEvaluate)
rfAUC <- (multiClassSummary(testResults, lev = levels(testResults$obs)))["ROC"]
# Show graph of ROC curve
forestROC <- roc(testEvaluate$target, testResults[, "No"], 
                 levels = rev(testEvaluate$target))
forestPlot <- plot(forestROC, type = "S", print.thres = .5, 
                   main = "ROC Curve for Random Forest")
forestPlot
# Load AUC result into results vector
AUCurves["Random Forest"] <- rfAUC
remove(testResults) # To save memory and avoid error


# AdaBoost
set.seed(6666)
adaBoostFit <- train(target ~ .,
                     data = trainingNumericClean,
                     method = "ada",
                     trControl = ctrl,
                     metric = "ROC")
## Save model to external file (for possible use with validation set)
#save(adaBoostFit, file = "~/R/RyersonCapstone/adaBoostFit")
# Evaluation
goodVariables <- names(trainingNumericClean)
# Get AUC result on test data
testResults <- predict(adaBoostFit, testEvaluate[, -length(testEvaluate)], 
                       type = "prob")
testResults$obs <- testEvaluate$target
testResults$pred <- predict(adaBoostFit, testEvaluate[, -length(testEvaluate)])
adaBoostAUC <- (multiClassSummary(testResults, 
                                  lev = levels(testResults$obs)))["ROC"]
# Show graph of ROC curve
adaROC <- roc(testEvaluate$target, testResults[, "No"], 
              levels = rev(testEvaluate$target))
adaPlot <- plot(adaROC, type = "S", print.thres = .5, 
                   main = "ROC Curve for AdaBoost")
adaPlot
# Load AUC result into results vector
AUCurves["Adaboost"] <- adaBoostAUC
remove(testResults) # To save memory and avoid error

# XGBoost, clean data only
xgBoostFit <- train(target ~ .,
                    data = trainingNumericClean,
                    method = "xgbTree",
                    trControl = ctrl,
                    metric = "ROC")
# Evaluation

# Get AUC result on test data
testResults <- predict(xgBoostFit, testEvaluate[, -length(testEvaluate)], 
                       type = "prob")
testResults$obs <- testEvaluate$target
testResults$pred <- predict(xgBoostFit, testEvaluate[, -length(testEvaluate)])
XGBoostAUC <- (multiClassSummary(testResults, 
                                 lev = levels(testResults$obs)))["ROC"]
# Show graph of ROC curve
xgROC <- roc(testEvaluate$target, testResults[, "No"], 
             levels = rev(testEvaluate$target))
xgPlot <- plot(adaROC, type = "S", print.thres = .5, 
                main = "ROC Curve for XGBoost")
xgPlot
# Load AUC result into results vector
AUCurves["XGBoost"] <- XGBoostAUC
remove(testResults)  # To save memory




### Final validation and exporting of file in a form uploadable to Kaggle


validation <- read.csv("~/R/RyersonCapstone/validation.csv", strip.white = TRUE)
# Extract variables used in model, and also ID so that I can make a submission
ids <- validation[, "ID"]
validation <- validation[, goodVariables[-142]]
validationIDs <- cbind(ID = ids, validation)

# Use the model with the highest AUC--the XGBoost model--to produce
# my solution
my_prediction <- predict(xgBoostFit,
                         newdata = validation,
                         type = "raw")
# Convert factor to 1s and 0s for Kaggle
myPredict <- (as.numeric(my_prediction) - 1)

# Create a data frame with two columns: PassengerId & target
my_solution <- data.frame(ID = ids, 
                          target = myPredict)
# Write my solution to a .csv file to submit
write.csv(my_solution, row.names = FALSE, file = "ASisti_solution.csv")
