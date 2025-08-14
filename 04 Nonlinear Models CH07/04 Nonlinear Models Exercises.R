
################################################################################
# File Name: 04 Nonlinear Models Exercises.R                                 
# Description: Exercises for the nonlinear models module of the ML shortcourse. 
# Author: SAIG, Dylan Steberg
# Date: 07/20/25
# Course: Machine Learning Short Course
# R version: 4.3.0
# Package(s): ISLR, MASS, splines, gam, and leaps
# Dataset(s): Boston, College, and Auto (all from ISLR package)
#
# Revision History:
#     11/10/23: Previous version last updated
#     07/20/25: Updated to match Python code
################################################################################

# The following exercises, code, and explanations are adapted from An 
# Introduction to Statistical Learning with Applications in R (ISLR) 2nd Edition
# (James, Witten, Hastie, Tibshirani, and Taylor 2023).

################################################################################
#################### Exercise: Predicting NO2 Concentration ####################
################################################################################

# This exercise is adapted from Chapter 7, Exercise 9 in ISLR. First we load 
# the required R packages.
library(ISLR)
library(MASS)
library(splines)
library(gam)
library(leaps)                

# In this exercise, we use the "Boston" data set, which is part of the ISLR 
# package. We will use two columns from this data: 
#   - dis: The weighted mean of distances to five Boston employment centers
#   - nox: Nitrogen oxides concentration in parts per 10 million

# We treat `dis` as the predictors and `nox` as the response.  Let's start by
# reading in the data and taking a look at a summary of the columns two columns
# we're interested in.
data(Boston)
summary(Boston[, c("dis","nox")])

# We can also look at the first few rows of our variables.
head(Boston[, c("dis","nox")], n = 5)

# Let's also split into training (80%) and test (20%) sets. In some examples we
# will use the train/test sets and in other cases the entire data set will be 
# used
train_id <- sample(nrow(Boston), nrow(Boston)*0.8, replace = FALSE)
trainBoston <- Boston[train_id, ] # Train set
testBoston <- Boston[-train_id, ] # Test set

################################################################################
###################### Part 1: Cubic Polynomial Regression ##################### 
################################################################################

# Initial inspection of our variables shows the nonlinear relationship plotted 
# below. With this in mind, we will begin by fitting a cubic polynomial 
# regression model to the training data.
plot(Boston$dis, Boston$nox, xlab = 'dis', ylab = 'nox')

# To fit the cubic polynomial regression model, we use the `poly()` function in 
# our model formula that will fit a 3rd degree polynomial in `dis`.
BosLM <- lm(nox ~ poly(dis, 3), data = trainBoston)
summary(BosLM)

# We can then predict on test set.
BosLMPred <- predict(BosLM, testBoston) 

# We can plot the fit for the test data. The plot shows the training points in 
# blue and the fitted line from the testing data in red.
plot(trainBoston$dis, trainBoston$nox, main = "Polynomial Regression, order 3",
     xlab = "dis", ylab = "nox", pch = 19, col = "blue")  
lines(sort(testBoston$dis), sort(BosLMPred, decreasing = TRUE), col = "red", lwd = 3)

################################################################################
################ Part 2: Different Degree Polynomial Regressions ############### 
################################################################################

# Now we will try out different degree polynomials. The following loop fits
# polynomial regressions of orders 1-10 and plots the fitted model. We create an
# empty vector of length 10 to store the RSS values to compare farther down.
BosPRMRSS <- numeric(10)

# sets up plotting layout
layout(rbind(1:5, 6:10))

for(i in 1:10){
  # fit model
  BosLMIter <- lm(nox ~ poly(dis,i), data = Boston) 
  
  # get RSS
  BosPRMRSS[i] <- sum(summary(BosLMIter)$residuals^2) 
  
  # sorts values of dis
  dissort <- data.frame(dis = sort(Boston$dis))
  
  # predicts at dis values
  BosLMPredIter <- predict(BosLMIter, dissort)
  
  # scatterplot and regression curve
  plot(Boston$dis, Boston$nox, col = "blue", main = paste("order", i),
       xlab = "dis", ylab = "nox", pch = 19, cex = 0.7)
  lines(sort(Boston$dis), BosLMPredIter, col = "red", lwd = 2)
}

# print RSS values for each degree
BosPRMRSS

# From the result, we can see that the RSS gets smaller as we add the ith
# order of predictor. The change of RSS stays quite still after adding 5th
# order of dis.

# Let's now explore how we might use K-fold cross-validation to select the 
# degree of polynomial we want in our model. We'll first create a function to 
# partition the data into k folds.
KFCVSplit <- function(dataset, k, i){
  # set seed for reproducibility
  set.seed(921101)
  
  # Number of rows in the data
  n0 <- nrow(dataset) 
  
  # Randomize data
  n0num <- sample(1:n0, n0, replace = FALSE) 
  
  # Separate the groups
  knGroup <- findInterval(n0num, seq(0, n0, l = (k + 1))) 
  if(k < n0){knGroup[knGroup == (k + 1)] <- k}
  if(k == n0){knGroup <- knGroup - 1}
  dataset2 <- data.frame(dataset, knGroup)
  
  # Set the train and test data for k fold CV
  test <- dataset2[which(dataset2$knGroup == i), ] 
  train <- dataset2[which(dataset2$knGroup != i), ] 
  traintest <- list(train, test)
  
  return(traintest)
}

# We'll also initialize a vector that will store the MSE for each degree.
LMMSE <- matrix(0, 10, 10)

# performs 10-fold cross validation for models of degree 1-10.
for(i in 1:10){
  # splits train and test sets into 10 folds
  BosTrain <- KFCVSplit(Boston, 10, i)[[1]]
  BosTest <- KFCVSplit(Boston, 10, i)[[2]]
  
  # performs 10-fold CV
  for(j in 1:10){
    BosLMIter <- lm(nox ~ poly(dis, j), data = BosTrain) # Linear model
    dissort <- data.frame(dis = sort(BosTest$dis))
    BosLMPredIter <- predict(BosLMIter, dissort) # Predicted values
    LMMSE[i, j] <- mean((BosTest$nox - BosLMPredIter)^2) # MSE
  }
}

# Now we can see the average MSE for each of the models
apply(LMMSE, 2, mean)

# Let's look at this in a plot to make it easier to see what's going on with 
# the MSE.
par(mfrow = c(1,1))
plot(1:10, apply(LMMSE, 2, mean), pch = 19, col = 'blue',
     xlab = "Degree of Polynomial", ylab = "MSE")
lines(1:10, apply(LMMSE, 2, mean), col = "blue")


# Using 10 fold cross validation, we can see that the MSE was the smallest
# when we are using only the first degree of dis. However, the
# differences are small.

################################################################################
########################## Part 3: Regression Splines ##########################
################################################################################

# Next we try the regression spline approach. One way to fit regression splines
# is by specifying knot locations. We do this using the `bs` function in our
# model formula. To get 4 degrees of freedom, we should select one knot for the 
# model (as the default degree is 3, that is a cubic spline is being fit). In 
# this case, we select the knot to be the median of the predictor but it does
# not necessarily need to be there. It is simply what we choose here.

# Fit a spline model where the knot is the median of "dis" variable and look 
# at the summary output.
BosSp1 <- lm(nox ~ bs(dis, knots = c(3.21)), data = Boston) 
summary(BosSp1)

# Notice that there are 4 spline coefficients rather than 5. This is because, by 
# default, `bs()` function assumes `intercept = FALSE`. Since we typically have 
# an overall intercept in the model, it generates the spline basis with the 
# given knots, and then discards one of the basis functions to account for the 
# intercept. 

# Now let's do this for 2 knots and 3 knots. We'll place our two knots at the 
# median and 1st quartile and our three knots at the median and first and third 
# quartiles.

# Spline model of Boston data with 2 knots.
BosSp2 <- lm(nox ~ bs(dis, knots = c(2.1, 3.21)), data = Boston)
summary(BosSp2)

# Spline model of Boston data with 3 knots.
BosSp3 <- lm(nox ~ bs(dis, knots = c(2.1, 3.21, 5.19)), data = Boston)
summary(BosSp3)

# We can compare the models using the `anova` function.
anova(BosSp1, BosSp2, BosSp3)

# From the ANOVA table, we can see the significance of the models sequentially
# We can see that including the second knot gives a small p-value, so it is 
# better to add that knot. However, adding the third knot is not significant
# meaning the model with three knots does not explain the data significantly
# better than the model with two knots (when that is the case we prefer the 
# model with less parameters). Thus, we select the model with two knots.

# Lets predict with this model and see what the fitted curve looks like.
# Note: This was fit using the whole data set. The two grey lines correspond to
# the locations of the two knots.
BosSp2Pred <- predict(BosSp2,Boston,se=TRUE)
par(mfrow = c(1, 1))
plot(Boston$dis, Boston$nox, pch = 19, main = "Cubic spline, 2 knots",
     xlab = "dis", ylab = "nox", col = "blue")
lines(sort(Boston$dis), sort(BosSp2Pred$fit, decreasing = TRUE), col = "red", lwd = 3)
abline(v = c(2.1, 3.21), lty = 2, lwd = 2, col = "darkgray")


# Another way to fit regression splines is by specifying degrees of freedom. We
# saw that above with three knots, the model summary states 6 degrees of freedom
# If we instead specify `df = 6` rather than the actual knots, `bs()` will 
# produce a spline with 3 knots chosen at uniform quantiles of the training 
# data. Now let's fit a spline for 4-7 degrees of freedom (i.e. 1-4 knots) and 
# print the RSS for each. We will also store the predictions and plot them.

# Spline model with 1 knot (Median)
BosDF4 <- lm(nox ~ bs(dis, df = 4), data = Boston)

# Spline model with 2 knots (33%, 67% quantile)
BosDF5 <- lm(nox  ~bs(dis, df = 5), data = Boston)

# Spline model with 3 knots (1st quantile, median, 3rd quantile)
BosDF6 <- lm(nox ~ bs(dis, df = 6), data = Boston)

# Spline model with 4 knots (20%, 40%, 60%, 80% quantile)
BosDF7 <- lm(nox ~ bs(dis, df = 7), data = Boston)

# Get the RSS of each model
sum(BosDF4$residuals^2)  # 4 df
sum(BosDF5$residuals^2)  # 5 df
sum(BosDF6$residuals^2)  # 6 df
sum(BosDF7$residuals^2)  # 7 df

# Notice how RSS decreases as we add more degrees of freedom. Let's plot what 
# each of these fits look like:
plot(Boston$dis, Boston$nox, pch = 19, main = "Cubic splines, DF = 4, 5, 6, 7",
     xlab = "dis", ylab = "nox", col = "lightblue")
BosDFList <- list(BosDF4, BosDF5, BosDF6, BosDF7)
colors = c("orange", "red", "purple", "blue")
for(i in 1:4){
  BosDFPredIter <- predict(BosDFList[[i]], Boston, se = TRUE)
  lines(sort(Boston$dis), sort(BosDFPredIter$fit, decreasing = TRUE), lwd = 3,
        col = colors[i])
}
legend("topright", fill = colors, legend = paste("DF =", 4:7))
abline(v = quantile(Boston$dis, c(0.25, 0.5, 0.75)), lty = 2, 
       col = "darkgray", lwd = 2)

# Now let's try more degrees of freedom, We'll loop through df = 3-22 and plot 
# the RSS to get a look at how it's changing based on degrees of freedom. We'll 
# first initialize a vector for RSS and then do the loop.
DFRSS <- numeric(20)

for(i in 1:20) {
  # Spline model with different number of knots
  BosDfIter2 <- lm(nox ~ bs(dis, df = i + 2), data = Boston)
  
  # get RSS
  DFRSS[i] <- sum((BosDfIter2$residuals)^2) 
}

# Let's print the RSS values and plot them
print(DFRSS)
plot(3:22, DFRSS, xlab = "Degrees of Freedom", ylab = "RSS", col="blue", pch= 19)
lines(3:22, DFRSS, col = "blue")

# Lastly, let's use cross-validation to get a better feel for which degrees of 
# freedom we should use. We'll start by using 10-fold cross-validation for 
# degrees of freedom = 4-13 and plot the average test MSE.
DFMSE <- matrix(0, 10, 10)
for(i in 1:10){
  BosTrain <- KFCVSplit(Boston, 10, i)[[1]]
  BosTest <- KFCVSplit(Boston, 10, i)[[2]]
  for(j in 1:10){
    # Spline model with different number of knots
    BosDfIter2 <- lm(nox ~ bs(dis, df = (j + 3)),data = BosTrain)
    
    # Predicted values
    BosDFPredIter2 <- predict(BosDfIter2, BosTest)
    
    # get MSE
    DFMSE[i, j] <- mean((BosTest$nox - BosDFPredIter2)^2)
  }
}

# get mean MSE for each number of degrees of freedom
apply(DFMSE, 2, mean)

# PLot MSE over degrees of freedom
plot(4:13, apply(DFMSE, 2, mean), pch = 19, xlab = "Degrees of Freedom", 
     ylab = "MSE", col = "blue")
lines(4:13, apply(DFMSE, 2, mean), col = "blue")

# Using 10-fold cross validation, we can see that the MSE was the smallest
# when we are having 7 knots. However, the differences among models (greater 
# than 4 df) are fairly negligible.

# Now let's try 20-fold cross-validation for degrees of freedom = 4-40 and plot
# the average test MSE.

BosSPMSE<-matrix(0, 20, 37)
for(i in 1:20) {
  BosTrainCV <- KFCVSplit(Boston, 20, i)[[1]]
  BosTestCV <- KFCVSplit(Boston, 20, i)[[2]]
  for(j in 1:37) {
    # Spline model with different number of knots
    BosSPIter <- lm(nox ~ bs(dis, df = j + 3),data = BosTrainCV)
    
    # Predicted values
    BosSPPredIter <- predict(BosSPIter, BosTrainCV)
    
    # get MSE
    BosSPMSE[i, j] <- mean((BosTestCV$nox - BosSPPredIter)^2)
  }
}

apply(BosSPMSE, 2, mean)
plot(4:40, apply(BosSPMSE, 2, mean), pch = 19,xlab = "Degrees of Freedom",
     ylab = "MSE", col = "blue")
lines(4:40, apply(BosSPMSE, 2, mean), col = "blue")

# We can see that the average test MSE actually increase as we add the number 
# of knots in the model. Therefore, it is better to keep the number of knots
# smaller.

################################################################################
################### Exercise: Predicting Out of State Tuition ##################
################################################################################

# This exercise is adapted from Chapter 7, Exercise 10 in ISLR.   

# In this exercise, we use the "College" data set, which is part of the `ISLP` 
# package. This data set includes statistics for a large number of US Colleges 
# from the 1995 issue of US News and World Report. It includes 777 observations 
# on the following 18 variables:
#   - Private`: A factor with levels No and Yes indicating private or public university
#   - Apps: Number of applications received
#   - Accept: Number of applications received
#   - Enroll: Number of new students enrolled
#   - Top10perc: Pct. new students from top 10% of H.S. class
#   - Top25perc: Pct. new students from top 25% of H.S. class
#   - F.Undergrad: Number of fulltime undergraduates
#   - P.Undergrad: Number of parttime undergraduates
#   - Outstate: Out-of-state tuition
#   - Room.Board: Room and board costs
#   - Books: Estimated book costs
#   - Personal: Estimated personal spending
#   - PhD: Pct. of faculty with Ph.D.'s
#   - Terminal: Pct. of faculty with terminal degree
#   - S.F.Ratio: Student/faculty ratio
#   - perc.alumni: Pct. alumni who donate
#   - Expend: Instructional expenditure per student
#   - Grad.Rate: Graduation rate

# The following code loads in the data and gives a summary of the variables.
data(College)
summary(College)

# We can also look at the first few rows of the data.
head(College, n = 5)

# And just for fun let's look at Virgina Tech's row
College[714, ]

# Set seed for reproducibility
set.seed(92110)

################################################################################
########################### Part 1: Forward Selection ##########################
################################################################################

# We'll start by splitting into a training (80%) and testing (20%) set.
train_id2 <- sample(nrow(College), nrow(College)*.8, replace = FALSE)
train2 <- College[train_id2, ] # Train set
test2 <- College[-train_id2, ] # Test set

# Notice that there's quite a few columns to use as predictors.  Let's use 
# forward selection to help pair down on the number of columns we use. We'll 
# look at Mallow's C_p and BIC for each subset.
ColFor <- regsubsets(Outstate ~ ., data = train2, method = "forward", nvmax = 15)
ColForSum <- summary(ColFor) 
ColForSum

# For Mallow's Cp, it is the smallest when selecting 13 predictors.
ColForSum$cp 
plot(1:15, ColForSum$cp, xlab = "Number of variables", ylab = "Mallow's Cp", 
     pch = 19, col = "blue")
lines(1:15, ColForSum$cp, col = "blue")
which(ColForSum$cp == min(ColForSum$cp))

# For BIC, it is the smallest when selecting 10 predictors.
ColForSum$bic
plot(1:15, ColForSum$bic, xlab = "Number of variables", ylab = "BIC", 
     pch = 19, col = "blue")
lines(1:15, ColForSum$bic, col = "blue")
which(ColForSum$bic == min(ColForSum$bic))

# According to the results, the Cp and the BIC are the smallest when we select 
# 13 and 10 predictors respectively. Just to make it easy, we will select the 12
# best predictors to build the model (Cp and BIC are both still relatively good 
# when we have 12 predictors). Thus the model will include everything except 
# "Top25perc", "F.Undergrad", "P.Undergrad", "Books", and "S.F.Ratio". 

################################################################################
####################### Part 2: Generalized Additive Model #####################
################################################################################

# Now that we've used forward selection to identify that we should use 12 
# predictors based on Mallow's Cp and BIC and which predictors those should be, 
# let's fit a generalize additive model with smoothing splines using those 9 
# predictors. A smoothing spline is a special case of a GAM with squared-error 
# loss and a single feature. To fit GAMs in R we will use the
# `gam` function from the 'gam' package. 'gam' is specified by associating each
# column of a model matrix with a particular smoothing operation: `s` for 
# smoothing spline and `lo` for loess (loess is not used in this example).

# First, we'll manually specify a cubic natural spline for each numeric predictor.
# Then we'll combine them with the categorical variable `Private`.
ColGAM <- gam(Outstate ~ Private + s(Apps, 3) + s(Accept, 3) + s(Enroll, 3) + 
                s(Top10perc, 3) + s(Room.Board, 3) + s(Personal, 3) + s(PhD, 3) +
                s(Terminal) + s(perc.alumni, 3) + s(Expend, 3) + s(Grad.Rate, 3), 
                data = train2)

# Now we can visualize the partial dependence of each variable on out of state 
# tuition.
layout(rbind(1:4, 5:8, 9:12))
plot.Gam(ColGAM, se = TRUE) 

# Although most of the predictors seem to have a linear line, there are some
# variables that seemed to have quadratic or cubic lines (Expend or Personal).

# Below is a summary of the GAM
summary(ColGAM)

# Now we can predict the testing set with this GAM and get the testing MSE.
ColGAMPred <- predict(ColGAM, test2) 
mean(((test2$Outstate-ColGAMPred)^2))

# Now let's compare testing MSE from the GAM to a basic linear model with the 
# same variables.
ColLM <- lm(Outstate ~ Private + Apps + Accept + Enroll + Top10perc + Room.Board + 
              Personal + PhD + Terminal + perc.alumni + Expend + Grad.Rate, 
            data = train2)
ColLMPred <- predict(ColLM, test2)
mean((test2$Outstate-ColLMPred)^2)

# We can see that the GAM model actually has smaller MSE compared to the 
# ordinary linear model.

################################################################################
##################### Exercise: Predicting Miles per Gallon ####################
################################################################################

# This exercise is adapted from Chapter 7, Exercise 8 in ISLR.   

# In this exercise, we'll use the "Auto" data from the `ISLR` package. This 
# dataset includes information on 392 vehicles for the following 9 variables:
#   - mpg: Miles per gallon  
#   - cylinders: Number of cylinders between 4 and 8
#   - displacement: Engine displacement (cu. inches)
#   - horsepower: Engine horsepower
#   - weight: Vehicle weight (lbs.)
#   - acceleration: Time to accelerate from 0 to 60 mph (sec.)
#   - year: Model year (modulo 100)
#   - origin: Origin of car (1. American, 2. European, 3. Japanese)
#   - name: Vehicle name

# Let's load the data and convert our categorical factors to categories as well
# as remove the nominal variable `name`.
data(Auto)
Auto$cylinders <- as.factor(Auto$cylinders)
Auto$origin <- as.factor(Auto$origin)
Auto <- Auto[,-9]

# We can also look at the first few rows of the data
head(Auto, n = 5)

# Set seed for reproducibility.
set.seed(921101)

# First we'll split the data into a training (80%) and test (20%) set.
train_id3 <- sample(nrow(Auto), 0.8*nrow(Auto), replace = FALSE)
trainAuto <- Auto[train_id3, ] # Train set
testAuto <- Auto[-train_id3, ] # Test set

# Our goal is to predict `mpg` using the other variables. Let's start with a 
# linear regression model.As usual, we fit the model on the training set, 
# predict the test set, and get test MSE.
AutoLM <- lm(mpg ~ ., data = trainAuto)
summary(AutoLM)
AutoLMPred <- predict(AutoLM, testAuto)
round(mean((testAuto$mpg-AutoLMPred)^2), 3)

################################################################################
######################### Part 1: Polynomial Regression ########################
################################################################################

# Now let's try a polynomial regression with up to the 5th order of each numeric 
# predictor (except year). Since having up to the 5th order of each predictor 
# creates alot of variables, we'll first use forward selection to narrow down 
# what variables to include. We'll follow the same steps as we saw in the 
# previous exercise. First let's set up our model matrix with our the powers of 
# our numeric variables.

# Remove the factor variables (`cylinders`, `origin`) and year from the dataset.
AutoM <- Auto[, c(-2, -7, -8)]
Auto2 <- AutoM

# Making a model matrix that has up to 5th order of predictors.
for(j in 2:5) {
  for(i in 1:4) {
    Auto2 <- data.frame(Auto2, (AutoM[,(i+1)])^j)
  }
}

# Fix the column names of the new model matrix.
aname <- colnames(AutoM)[-1]
colnames(Auto2) <- c(colnames(AutoM), paste(aname, rep(2:5, each = 4), sep=""))

# Add cylinders, origin, and year back in to the new model matrix.
Auto2 <- data.frame(Auto2, cylinders = as.factor(Auto$cylinders),
                    origin = as.factor(Auto$origin),
                    year = Auto$year)

# Now we once again split our data into a training and test set.
trainAuto2 <- Auto2[train_id3, ]
testAuto2 <- Auto2[-train_id3, ] 

# Use forward selection to choose variables.
Auto2For <- regsubsets(mpg ~ ., data = trainAuto2, method = "forward", nvmax = 20)
Auto2ForSum <- summary(Auto2For)

# Now let's figure out how many variables are suggested using Mallow's Cp.
Auto2ForSum$cp
par(mfrow = c(1, 1))
plot(1:20, Auto2ForSum$cp, xlab = "Number of variables", ylab = "Mallow's Cp", 
     pch = 19, col = "blue")
lines(1:20, Auto2ForSum$cp, col = "blue")
which(Auto2ForSum$cp == min(Auto2ForSum$cp))
 
# Suggests 19 is the best which includes: "displacement", "displacement2", 
# "displacement3", "horsepower", "horsepower2", "horsepower5", "weight",
# "weight2", "weight3", "weight4", "acceleration", "acceleration5", "cylinders",
# "origin", and "year".

# Now let's figure out how many variables are suggested using BIC.
Auto2ForSum$bic 
plot(1:20, Auto2ForSum$bic, xlab = "Number of variables", ylab = "BIC",
     pch = 19, col = "blue")
lines(1:20, Auto2ForSum$bic, col = "blue")
which(Auto2ForSum$bic == min(Auto2ForSum$bic))

# Suggests 8 is the best which includes: "horsepower", "horsepower2", "weight",
# "weight2", "acceleration", "acceleration5", "cylinders", and "year".

# Lets use the list of variables suggested by Mallow's Cp to fit a polynomial
# regression model on the training data. Then we can get the preditions with the
# testing data, and calculate the test MSE.
AutoPRM <- lm(mpg ~ displacement + displacement2 + displacement3 + horsepower +
                horsepower2 + horsepower5 + weight + weight2 + weight3 + weight4 +
                acceleration + acceleration5 + cylinders + origin + year,
                data=trainAuto2)
summary(AutoPRM)
AutoPRMPred <- predict(AutoPRM, testAuto2) 
mean((testAuto2$mpg - AutoPRMPred)^2)

# We can see that fitting polynomial regression model outperforms the simple
# linear regression.

################################################################################
############### Part 2: Generalized Additive Model with Splines ################ 
################################################################################

# Lastly, we'll fit a GAM model on Auto data set, where the splines of the 
# continuous variable are 4. We predict using the test data, and return the test
# MSE.
AutoGAM <- gam(mpg ~ s(weight, 4) + s(year, 4) + s(displacement, 4) + origin +
                 s(acceleration, 4) + s(horsepower, 4) + cylinders, 
               data = trainAuto)
AutoGAMPred <- predict(AutoGAM, testAuto) 
mean((testAuto$mpg - AutoGAMPred)^2)

# We can see that the MSE for the GAM model is smaller than the MSE for 
# the linear regression model and a little bit smaller than the MSE for the 
# polynomial regression model.
