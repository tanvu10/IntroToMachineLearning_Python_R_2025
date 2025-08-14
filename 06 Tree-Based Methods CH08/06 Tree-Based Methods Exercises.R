
################################################################################
# File Name: 06 Tree-Based Methods Exercises.R                                 
# Description: Exercises for the tree-based methods module of the ML shortcourse. 
# Author: SAIG, Dylan Steberg
# Date: 07/25/25
# Course: Machine Learning Short Course
# R version: 4.3.0
# Package(s): ISLR, tree, and randomForest
# Dataset(s): Carseats (from ISLR package)
#
# Revision History:
#     11/14/23: Previous version last updated
#     07/25/25: Updated to match Python code
################################################################################

# The following exercises, code, and explanations are adapted from An 
# Introduction to Statistical Learning with Applications in R (ISLR) 2nd Edition
# (James, Witten, Hastie, Tibshirani, and Taylor 2023).

################################################################################
########################## Exercise: Carseats Dataset ##########################
################################################################################

# This exercise is adapted from Chapter 8, Exercise 8 in ISLR. First we load 
# the required R packages.
library(ISLR)
library(tree)
library(randomForest)

# This problem invovles the "Carseats" from the ISLR pacakge. This is a 
# simulated data set containing the sales of child car seats at 400 different 
# stores. The variables include:
# - Sales: Unit sales (in thousands) at each location
# - CompPrice: Price charged by competitor at each location
# - Income: Community income level (in thousands of dollars)
# - Advertising: Local advertising budget for company at each location 
#     (in thousands of dollars)
# - Population: Population size in region (in thousands)
# - Price: Price company charges for car seats at each site
# - ShelveLoc: A factor with levels Bad, Good and Medium indicating the 
#     quality of the shelving location for the car seats at each site
# - Age: Average age of the local population
# - Education: Education level at each location
# - Urban: A factor with levels No and Yes to indicate whether the store is 
#     in an urban or rural location
# - US: A factor with levels No and Yes to indicate whether the store is in 
#     the US or not
 
# The goal of this exercise is to predict `Sales` using regression trees and 
# related approaches, treating the response as a quantitative variable.

# First we will load in the data and inspect the first few rows.
data("Carseats")
head(Carseats, n = 5)

# Here we split the data into a training and a test set. A random seed is set 
# to ensure reproducibility of results.
set.seed(42)
train = sample(nrow(Carseats), 0.5*nrow(Carseats), replace = FALSE)
X_train = Carseats[train, 2:11]; y_train = Carseats[train, 1]
X_test = Carseats[-train, 2:11]; y_test = Carseats[-train, 1]

################################################################################
######################### Part 1: Full Regression Tree #########################
################################################################################

# We start by fitting a regression tree to the training set. We can do this
# using the `tree` function. Then we plot the tree and compute the MSE on the 
# testing set.
full_tree <- tree(y_train ~ ., data = X_train)
plot(full_tree)
text(full_tree, pretty = 0, digits = 2, cex = 0.8)

y_pred = predict(full_tree, newdata = X_test)
full_mse = mean((y_pred - y_test)^2)
full_mse

################################################################################
############################# Part 2: Tree Pruning #############################
################################################################################

# Next we use cross-validation to determine the optimal level of tree complexity.
# That is through cross-validation we prune our tree to the optimal depth.
cv.scores <- cv.tree(full_tree)
plot(cv.scores$size, cv.scores$dev, type = "b", xlab = "Tree size", ylab = "Deviance")
min <- which.min(cv.scores$dev)
abline(v = cv.scores$size[min], lty = 2, col = "red")

# A tree depth of 15 gives the smallest deviance although that is still pretty
# big. A depth of 6 gives essentially the same deviance with less chance of
# over-fitting. Thus we fit a regression tree to the training data with a maximum
# depth of 6. We can do this using the `prune.tree` function where we specify
# `best = 6` to be the depth to prune the full tree to. Once again we plot the 
# tree, and compute the MSE on the testing set.
pruned_tree <- prune.tree(full_tree, best = 8)
plot(pruned_tree)
text(pruned_tree, pretty = 0, digits = 2, cex = 0.8)

y_pred = predict(pruned_tree, newdata = X_test)
pruned_mse = mean((y_pred - y_test)^2)
pruned_mse

# We see that pruning the tree resulted in a tree with a smaller test MSE.

################################################################################
############################### Part 3: Bagging ################################
################################################################################

# So far we have only built a single regression tree. Now we will create multiple 
# trees (or a forest) using the `randomForest` function from the randomForest
# package. We define the number of trees in our forest using the `ntree` 
# argument. Our final predictions are averaged from the predictions of each 
# individual regression tree created. 

# To implement bagging we must set `mtry` to be the number of predictor 
# variables in our dataset. Setting it less than that still performs bagging, 
# but will also select only consider a certain number of variables at each split
# while setting it to 10 means all variables will always be considered. Here we 
# run the random forest which is creating 50 individual trees. The test MSE is 
# then computed. 

# Note: Since multiple trees are being built it is not possible to plot a random 
# forest, although you could plot each of the 50 trees.
bagging_for <- randomForest(y_train ~ ., data = X_train, mtry = 10, ntree = 50)

y_pred = predict(bagging_for, newdata = X_test)
bagging_mse = mean((y_pred - y_test)^2)
bagging_mse

# The test MSE from this random forest is much lower than the test MSE of an 
# individual tree. 

################################################################################
########################## Part 4: Tuning Parameters ###########################
################################################################################

# Now we will look at some other arguments to the `RandomForest` function that 
# can be tuned by the user. A brief list of tunable parameters are:
#  - ntree: The number of trees to grow
#  - mtry: Number of variables randomly sampled as candidates at each split.
#  - nodesize: Minimum size of terminal nodes
#  - maxnodes: Maximum number of terminal nodes trees in the forest can have.

# See the help page for `RandomForest` (type: ?randomForest() in the console) to
# get more information on these arguments and other arguments not listed above. 
 
# Now we will fit a tree with some of these other arguments defined. Note that 
# specifying `importance = TRUE` is used in the next part for investigating
# feature importance.
tuned_for <- randomForest(y_train ~ ., data = X_train, mtry = 8, ntree = 300,
                          maxnodes = 50, nodesize = 5, importance = TRUE)

y_pred = predict(tuned_for, newdata = X_test)
tuned_mse = mean((y_pred - y_test)^2)
tuned_mse

# The test MSE for this tuned forest is slightly smaller than the test MSE from 
# the bagging example above. 

# Note: This was not extensively tuned and the decrease is almost surely from the 
# number of trees increasing. One way to tune models fairly comprehensively is 
# by tuning over a grid of parameter settings to find the optimal parameter 
# setting. While being a solid approach, it tends to be very computationally 
# intensive as the number of models fit grows quite large given the number of 
# parameters being tuned and the number of levels for each parameter.

################################################################################
########################## Part 5: Feature Importance ##########################
################################################################################

# Finally, we look at feature importance from the random forest. Feature 
# importance is provided by the component `importance` from a randomForest object.
# The forest must include `importance = TRUE` in order for the importance to be 
# calculated while fitting the forest. Using our tuned forest from the previous 
# section we can investigate feature importance.
importance = tuned_for$importance
importance

# This is not a great output as it is hard to compare so we will plot the feature 
# importance using a bar chart.
barplot(importance[, 1], xlab = "", ylab = "Mean inc. in MSE", col = "skyblue",
        main = "Feature importance plot for the tuned random forest")

# We see that `Price` and `ShelveLoc` appear to be the most important features in 
# the model while `Urban`, `US`, and `Education` seem to be the least important. 
# While the numbers on the y-axis have meaning, it is better to compare the bars 
# relative to one another when determining importance.

# Note: There are multiple ways to look at variable importance in random forests. 
# We plotted the mean increase in MSE but could have looked at the second
# column which is the increase in node purity. Other methods include permutation 
# importance which looks at the mean accuracy decrease when the values of a 
# feature are permutated and SHAP values, which look at the marginal contribution 
# of a feature to predictions.
