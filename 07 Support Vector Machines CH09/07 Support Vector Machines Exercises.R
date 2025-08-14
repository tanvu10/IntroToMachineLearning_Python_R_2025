################################################################################
# SAIG Machine Learning Short Course                                           #
# Chapter 9 - Support Vector Machines                                          #
# Instructor Code                                                              #
#                                                                              #                                                             #
# Last updated: 01/08/2025                                                      #
################################################################################

# The following exercises, code, and explanations are adapted from An 
# Introduction to Statistical Learning with Applications in R (ISLR) 2nd Edition
# (James, Witten, Hastie, Tibshirani, and Taylor 2023).

########################
# Exercise: OJ dataset #
########################

# This exercise is adapted from Chapter 9, Exercise 8 in ISLP. 
# This problem involves the OJ data set which is part of the ISLR package.
library(ISLR)
library(e1071) # for svm

# The OJ data pertains to Orange Juice purchases where customers bought either 
# Citrus Hill (CH) or Minute Maid (MM) Brand Orange Juice. We'll start by 
# reading in the data.

# Read in data
data(OJ)

# Data Preview
head(OJ)

# The Purchase column reflects the brand of OJ purchased - this will be our 
# response variable that we want to classify our observations by. The rest of 
# the columns reflect different characteristics about the customer, product, and 
# sale (including discounts, price, store location, etc). For a description of 
# all variables, see https://intro-stat-learning.github.io/ISLP/datasets/OJ.html.

# Number of rows and columns
dim(OJ)

n <- nrow(OJ)

# The code above shows that there are a total of 1070 observations and 18 
# variables. We have a mix of both categorical and continous variables. 

# Since we can't visualize all variables at the same time, we'll instead plot 
# each pair of variables. 

# In the pairwise plots CH = red and MM = blue.
pairs(OJ[,-1], pch=19, col=c(rgb(1, 0, 0, alpha=0.5), 
                             rgb(0, 0, 1, alpha=0.5))[as.factor(OJ$Purchase)])

# There's alot to look at here!  Notice that it's not as straightfoward to draw 
# a straight dividing line in many of these plots.  This is even more 
# complicated in 17 dimensions using all predictor variables.  

# Some have more clear patterns than others.  For example, the `LoyalCH` 
# variable represents how loyal each customer is to the Citrus Hill Brand.  
# Naturally, those who are more loyal to the brand are more likely to buy 
# Citrus Hill, so we see more red points on the right had side of each plot in 
# the `LoyalCH` column.  It may make sense to divide these pairs roughly around 
# `LoyalCH=0.5`.

##################################################################################
# a. Create a training set containing a random sample of 800 observations, and a #
#  test set containing the remaining observations.                               #
##################################################################################

# Ensure reproducibility of the random sampling
set.seed(42)

# Randomly sample 800 observations for the training set
train <- sample(seq_len(nrow(OJ)), 800) 
n_train <- 800

# Create a test set from the remaining observations
test <- setdiff(seq_len(nrow(OJ)), train)  
n_test <- n - n_train

#################################################################################
# b. Fit a support vector classifier to the training data using `cost = 0.01`,  #
#    with `Purchase` as the response and the other variables as predictors. Use #
#    the `summary()` function to produce summary statistics, and describe the   #
#    results obtained.                                                          #
#################################################################################

# Fit an SVM with a linear kernel and C=0.01

# Recall that C is our tuning parameter that can be considered as the "budget" 
# or "cost" for the amount that the margin can be violated by the n observations.

# In the function below we set C using the cost argument

fit <- svm(Purchase ~ ., data = OJ[train, ], kernel = "linear", cost = 0.01)
summary(fit)  # Summarize the fitted model

# c. What are the training and test error rates?

# Calculate training and test error rates for the model
errs <- function(model) {
  train_err <- sum(ifelse(model$fitted == OJ[train, "Purchase"], 0, 1)) / n_train
  test_err <- sum(ifelse(predict(model, OJ[test,]) == OJ[test, "Purchase"], 0, 1)) / n_test
  return(c(train = train_err, test = test_err))
}
errs(fit)

##################################################################################
# d. Use the `tune()` function to select an optimal C (cost). Consider values in # 
# the range 0.01 to 10.                                                          #
##################################################################################

# Use the tune() function to find an optimal cost parameter
tuned <- tune(svm, Purchase ~ ., data = OJ[train, ], kernel = "linear", 
              ranges = list(cost = seq(0.01, 10, by = 2)))
tuned$best.parameters  # Display the best cost parameter found
summary(tuned)  # Summarize the tuning results


##################################################################################
# e. Compute the training and test error rates using this new value for `cost`.  #
##################################################################################

errs(tuned$best.model)

################################################################################
# f. Repeat parts (b) through (e) using a support vector machine with a radial #
#    kernel. Use the default value for `gamma`.                                #
################################################################################

# cost = 0.01
fit2 <- svm(Purchase ~ ., data = OJ[train, ], kernel = "radial", cost = 0.01)
errs(fit2)

# Hyperparameter tuning for SVM with a radial kernel
tuned2 <- tune(svm, Purchase ~ ., data = OJ[train, ], kernel = "radial", 
               ranges = list(cost = seq(0.01, 10, by = 2)))
tuned2$best.parameters
errs(tuned2$best.model)  # Calculate error rates for the model with radial kernel

#########################################################################
# g. Repeat parts (b) through (e) using a support vector machine with a # 
#    polynomial kernel. Set `degree = 2`.                               #
#########################################################################

# cost = 0.01
fit3 <- svm(Purchase ~ ., data = OJ[train, ], kernel = "polynomial", degree = 2, 
            cost = 0.01)
errs(fit3)

# Hyperparameter tuning for SVM with a polynomial kernel
tuned3 <- tune(svm, Purchase ~ ., data = OJ[train, ], kernel = "polynomial", 
               ranges = list(cost = seq(0.01, 10, by = 2)), degree = 2)
tuned3$best.parameters
errs(tuned3$best.model)  # Calculate error rates for the model with polynomial kernel

###########################################################################
# h. Overall, which approach seems to give the best results on this data. #
###########################################################################