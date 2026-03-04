# NPS Analysis (Logistic Regression model)

# Note: Load the data set as df
################################################################
# Load necessary libraries

library(caret)
library(dplyr)

################################################################
#Corelation Analysis

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
################################################################
#Up-sampling minority class

# Print number of true and false class
cat("Number of true class:", sum(data$NPS_9 == 1), "\n")
cat("Number of false class:", sum(data$NPS_9 == 0), "\n")

#Mention the up sample percentage
Up_sample_percent = 0.2# Up-sample true class to 10% of the false class

# Up-sample the true class using the sample() function
n_false <- sum(data$NPS_9 == 0)
n_true <- sum(data$NPS_9 == 1)
n_samples <- round(n_false * Up_sample_percent/ (1 - Up_sample_percent))
data_upsampled <- rbind(data, data[data$NPS_9 == 1,][sample(n_true, n_samples, replace = TRUE),])

# Print number of true and false class after up-sampling
cat("Number of true class after up-sampling:", sum(data_upsampled$NPS_9 == 1), "\n")
cat("Number of false class after up-sampling:", sum(data_upsampled$NPS_9 == 0), "\n")
################################################################

# Define predictor variables
predictors <- c("Age", "Years", "Personality", "Certficates", "Feedback", "Salary" , "Female")

# Create cross-validation folds
folds <- createFolds(data_upsampled$NPS_9, k = 5)# five folds

# Create empty vectors to store cross-validation results
cv_accuracy <- rep(NA, length(folds))
cv_prec <- rep(NA, length(folds))
cv_f1 <- rep(NA, length(folds))

################################################################
# Perform k-fold cross-validation

for (i in 1:length(folds)) {
  # Split data into training and validation sets
  train <- data_upsampled[-folds[[i]],]
  validation <- data_upsampled[folds[[i]],]
  
  # Train logistic regression model on training set
  model <- glm(NPS_9 ~ ., data = train[,c(predictors, "NPS_9")], family = "binomial")
  
  # Make predictions on validation set
  predictions <- predict(model, newdata = validation[,predictors], type = "response")
  
  # Convert predicted probabilities to binary class
  predicted_class <- ifelse(predictions >= 0.5, 1, 0)
  
  # Calculate accuracy and F1 score of predictions
  accuracy <- sum(predicted_class == validation$NPS_9) / length(validation$NPS_9)
  tp <- sum(predicted_class == 1 & validation$NPS_9 == 1)
  fp <- sum(predicted_class == 1 & validation$NPS_9 == 0)
  fn <- sum(predicted_class == 0 & validation$NPS_9 == 1)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1_score <- 2 * precision * recall / (precision + recall)
  
  # Store accuracy and F1 score in vectors
  cv_accuracy[i] <- accuracy
  cv_f1[i] <- f1_score
  cv_prec[i] <- precision
}
################################################################

# Print summary of final model
summary(model)
###############################################################
# Predicting using real-world scenario(please see report last page)

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
prediction <- predict(model, newdata = test_data, type = "response")
print(prediction)
################################################################

# Print mean and standard deviation of cross-validation results
cat("Mean accuracy:", mean(cv_accuracy), "\n")
cat("Standard deviation of accuracy:", sd(cv_accuracy), "\n")
cat("Mean Precision:", mean(cv_prec), "\n")
cat("Mean F1 score:", mean(cv_f1), "\n")
cat("Standard deviation of F1 score:", sd(cv_f1), "\n")

################################################################
#Plot model
#plot(model)
################################################################
# Model Features
#Data cleaning & sub-setting
#Up-sampling minority class
#k-fold cross validation
#Accuracy + F1 score validation
#AUC
################################################################