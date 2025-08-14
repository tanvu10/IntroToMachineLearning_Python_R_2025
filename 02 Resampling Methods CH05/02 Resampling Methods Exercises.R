################################################################################
# File Name: 02 Resampling Methods Exercises.R                                 
# Description: Exercises for the resampling methods module of the ML shortcourse. 
# Author: SAIG, Dylan Steberg
# Date: 07/07/25
# Course: Machine Learning Short Course
# R version: 4.3.0
# Package(s): ISLR, boot, MASS, and class
# Dataset(s): Weekly (from ISLR package)
#
# Revision History:
#     07/07/25: Updated to match Python code
################################################################################

# The following exercises, code, and explanations are adapted from An 
# Introduction to Statistical Learning with Applications in R (ISLR) 2nd Edition
# (James, Witten, Hastie, Tibshirani, and Taylor 2023).

################################################################################
########################## Exercise: The Stock Market ##########################
################################################################################

# This exercise is adapted from Chapter 5, Exercise 7 in ISLR. First we load the
# required packages.
library(ISLR)
library(boot)
library(MASS)
library(class)

# This exercise uses the "Weekly" data set which has information on weekly
# percentage returns for the S&P 500 stock index between 1990 and 2010. The data
# set includes 1089 observations and 9 variables which include:
#   - Year: The year that the observation was recorded.
#   - Lag1: Percentage return for previous week.
#   - Lag2: Percentage return for 2 weeks previous.
#   - Lag3: Percentage return for 3 weeks previous.
#   - Lag4: Percentage return for 4 weeks previous.
#   - Lag5: Percentage return for 5 weeks previous.
#   - Volume: Volume of shares traded (average number of daily shares traded in 
#       billions).
#   - Today: Percentage return for this week.
#   - Direction: A factor with levels Down and Up indicating whether the market
#       had a positive or negative return on a given week.

# Our goal with this data is to model Today using the other variables
 
# The following line loads the data from the ISLR package using the `data()` 
# function.
data(Weekly)

# The first few lines of the data set are shown below.
head(Weekly, n = 10)

# Here we look at scatterplots for each pair of variables.
pairs(Weekly)

################################################################################
####################### Part 1:  K-fold Cross Validation ####################### 
################################################################################

# We will begin by defining a function that splits our data into raining and 
# validation sets to be used in k-fold cross validation.
KFCVSplit <- function(dataset, k, i) {
  set.seed(921101)
  n0 <- nrow(dataset) # Number of rows in Weekly data
  n0num <- sample(1:n0, n0, replace = FALSE) # Randomize data
  knGroup <- findInterval(n0num, seq(0, n0, l = (k + 1))) # Separate the group
  
  if(k < n0){knGroup[knGroup == (k + 1)] <- k}
  if(k == n0){knGroup <- knGroup - 1}
  
  dataset2 <- data.frame(dataset, knGroup)
  test <- dataset2[which(dataset2$knGroup == i),] # Set the test data for k fold CV
  train <- dataset2[which(dataset2$knGroup != i),] # Set the train data for k fold CV
  traintest <- list(train, test)
  return(traintest)
}

# Make an empty MSE vector for k-fold cross validation.
MSEs1 <- matrix(0, 3, 10) 
rownames(MSEs1) <- c("Lags Model", "Full Model", "Stepwise Model")

# Now we use our KFCVSplit function to do a k-fold cross validation with k = 10.
# We will consider three different linear regressions:
#   1. A model using only the 5 lag variables
#   2. A model using all the variables
#   3. The "best" model (using AIC) found through step-wise selection

for(i in 1:10) {
  train <- KFCVSplit(Weekly, 10, i)[[1]]
  test <- KFCVSplit(Weekly, 10, i)[[2]]
  
  # Regression using just the lag variables
  regression1 = lm(Today ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5,
                   data = train)
  predictions1 = predict(regression1, test)
  MSEs1[1, i] = mean((predictions1 - test$Today)^2)
  
  # Regression using all predictors
  regression1 = lm(Today ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume + Direction,
                   data = train)
  predictions1 = predict(regression1, test)
  MSEs1[2, i] = mean((predictions1 - test$Today)^2)
  
  # Regression with the model given from step wise model selection
  regression1 = lm(Today ~ Direction + Lag3 + Lag1,
                   data = train)
  predictions1 = predict(regression1, test)
  MSEs1[3, i] = mean((predictions1 - test$Today)^2)
}

# Compare the average test MSE for the three models.
round(apply(MSEs1, 1, mean), 3)


################################################################################
################ Part 2: Leave-one-out Cross-validation (LOOCV) ################
################################################################################

# We will then apply the above user-defined function to compute LOOCV. Note that
# we simply set k to be the length of the data, as by definition of LOOCV, LOOCV 
# is the N-fold CV.

# Get number of rows in Weekly dataset.
n0 <- nrow(Weekly)

# Make an empty MSE vector for LOOCV.
MSEs2 <- matrix(0, 3, n0) 
rownames(MSEs2) <- c("Lags Model", "Full Model", "Stepwise Model")

# Loop through observations to perform LOOCV.
for(i in 1:n0){
  train <- KFCVSplit(Weekly, n0, i)[[1]]
  test <- KFCVSplit(Weekly, n0, i)[[2]]
  
  # Regression using just the lag variables
  regression1 = lm(Today ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5,
                   data = train)
  predictions1 = predict(regression1, test)
  MSEs2[1, i] = mean((predictions1 - test$Today)^2)
  
  # Regression using all predictors
  regression1 = lm(Today ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume + Direction,
                   data = train)
  predictions1 = predict(regression1, test)
  MSEs2[2, i] = mean((predictions1 - test$Today)^2)
  
  # Regression on the model given from step wise model selection
  regression1 = lm(Today ~ Direction + Lag3 + Lag1,
                   data = train)
  predictions1 = predict(regression1, test)
  MSEs2[3, i] = mean((predictions1 - test$Today)^2)
}

# Compare the average test MSE for the three models.
round(apply(MSEs2, 1, mean), 3)  


################################################################################
############################# Part 3: The Bootstrap ############################
################################################################################

# Finally, we move on to bootstrapping. Note that the below example is 
# estimating standard errors. Before moving to the coding part, let's take 
# a look at the output table from logistic regression on the "Weekly" data set
# to see the estimated coefficients, means, and standard deviations.
WeeklyLM <- lm(Today ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume + Direction,
               data = Weekly) 
summary(WeeklyLM)

# Make a function that will get the coefficients of one bootstrap data. Then 
# use the boot() function to run 1000 bootstrap samples.
boot.fn <- function(data, index){
  bootlm <- lm(Today ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume + Direction, 
                 data = data, subset = index)
  return(bootlm$coefficients)
}  
bootstrap_estimates = boot(Weekly, boot.fn, 1000)

# Here we can compare the estimated standard errors from the logistic regression
# model and the bootstrap and see that they are very similar although the
# estimates for the Lag variables tend to be overestimated.
summary(WeeklyLM)$coefficients[, "Std. Error"]
apply(bootstrap_estimates$t, 2, sd)

################################################################################
################################### Appendix ###################################
################################################################################
# Here is how we determined our third model using stepwise model selection.
intercept_model = lm(Today ~ 1, data = Weekly)
full_model = lm(Today ~ ., data = Weekly)
stepwise = step(intercept_model, direction = "both", 
                scope = formula(full_model), trace = 0)
stepwise$anova # The "best" model includes Direction, Lag3 and Lag1
