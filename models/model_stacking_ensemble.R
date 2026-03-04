# NPS Analysis (Stacked model - Logistic Regression + Decision Tree + Naive Bayes)
# - by John Melwin Richard
# Please Note: Load the data set as df
################################################################
# Load necessary libraries
#(Some of the libraries needs to be installed in the console line like >library(caretEnsemble))
#Please enter "No" if R-studio asks restart for installing some libraries
library(caret)
library(dplyr)
library(caretEnsemble)
library(ROSE)
library(plyr)
library(pROC)

################################################################
# Select relevant columns for correlation matrix
df_cor <- select(df, Age, Female, Years, Salary, Certficates, Feedback, NPS)

# Create correlation matrix
cor_matrix <- cor(df_cor)

# Print correlation matrix
print(cor_matrix)

################################################################
# Subset data for employees in the software product group with college degree
data <- subset(df, Business == "Software" & College == "Yes")

# Create a binary target variable for NPS>=9
data$NPS_9 <- ifelse(data$NPS >= 9, 1, 0)

# Print number of true and false class
cat("Number of true class:", sum(data$NPS_9 == 1), "\n")
cat("Number of false class:", sum(data$NPS_9 == 0), "\n")

#Mention the up sample percentage
Up_sample_percent = 0 # Up-sample true class to 10% of the false class

# Up-sample the true class using the sample() function
n_false <- sum(data$NPS_9 == 0)
n_true <- sum(data$NPS_9 == 1)
n_samples <- round(n_false * Up_sample_percent/ (1 - Up_sample_percent))
data_upsampled <- rbind(data, data[data$NPS_9 == 1,][sample(n_true, n_samples, replace = TRUE),])

# Print number of true and false class after up-sampling
cat("Number of true class after up-sampling:", sum(data_upsampled$NPS_9 == 1), "\n")
cat("Number of false class after up-sampling:", sum(data_upsampled$NPS_9 == 0), "\n")

################################################################
# Preparation for the model
# Define predictor variables
predictors <- c("Age", "Years", "Personality", "Certficates", "Feedback", "Salary" , "Female")

#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(data_upsampled$NPS_9, p=0.75, list=FALSE)
trainSet <- data_upsampled[ index,]
testSet <- data_upsampled[-index,]

################################################################
#Defining the training controls for multiple models

fitControl <- trainControl(
  method = "cv",
  number = 5,# five folds
  savePredictions = 'final',
  classProbs = TRUE)

################################################################

#Defining the predictors and outcome
predictors <- c("Age", "Years", "Personality", "Certficates", "Feedback", "Salary" , "Female")
outcomeName<-'NPS_9'
trainSet$NPS_9 <- ifelse(trainSet$NPS >= 9, "yes", "no")
testSet$NPS_9 <- ifelse(testSet$NPS >= 9, "yes", "no")
testSet$NPS_9 <- as.factor(testSet$NPS_9)

################################################################
# (Ensemble learning - Logistic Regression + Decision Tree + Naive Bayes)
control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)

algorithms_to_use <- c('rpart', 'glm', 'nb')

#Please wait for sometime to get results
stacked_models <- caretList(NPS_9 ~., data=trainSet[,c(predictors, "NPS_9")], trControl=control_stacking, methodList=algorithms_to_use)

stacking_results <- resamples(stacked_models)

summary(stacking_results)

dotplot(stacking_results)

################################################################
# stack using glm(Logistic regression model)
stackControl <- trainControl(method="repeatedcv", number=5, repeats=3, savePredictions=TRUE, classProbs=TRUE)

set.seed(100)

glm_stack <- caretStack(stacked_models, method="glm", metric="Accuracy", trControl=stackControl)

print(glm_stack)

################################################################
# Test using real world data
test_data <- data.frame(
  Sales_Rep = c(001 , 002),
  Business = c("Software", "Software"),
  Age = c(35, 30),
  Female = c(1, 0),
  Years = c(10, 1),
  College = c("Yes", "Yes"),
  Personality = c("Diplomat", "Analyst"),
  Certficates = c(6, 1),
  Feedback = c(4, 1),#important
  Salary = c(150000, 70000 )
)
prediction <- predict(glm_stack, newdata = test_data)
print(prediction)

################################################################
# Print validation results

predictions <- predict(glm_stack, newdata = testSet)
confusion_matrix <- confusionMatrix(predictions, testSet$NPS_9, mode = "everything")
print(confusion_matrix)

################################################################
#Plot Area under curve graph
predictions <-  ifelse(predictions == "yes", 1, 0)
testSet$NPS_9 <-  ifelse(testSet$NPS_9 == "yes", 1, 0)
roc_curve <- roc(predictions, testSet$NPS_9)
plot(roc_curve, main="ROC Curve for Stacked Model", print.auc=TRUE, auc.polygon=TRUE)

################################################################
# Model Features
#Data sub-setting
#Up-sampling minority class(sampling)
#Ensemble learning (stacking)
#k-fold cross validation
#Accuracy + F1 score validation
#AUC
################################################################


