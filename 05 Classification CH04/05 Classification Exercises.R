################################################################################
# SAIG Machine Learning Short Course                                           #
# Chapter 4 - Classification Exercises                                         #
# Instructor Code                                                              #
#                                                                              #                                                             #
# Last updated: 10/31/2023                                                     #
################################################################################

# The following exercises, code, and explanations are adapted from An 
# Introduction to Statistical Learning with Applications in R (ISLR) 2nd Edition
# (James, Witten, Hastie, Tibshirani, and Taylor 2023).

################################################################################
# Exercise: The Stock Market                                                   #
################################################################################

# This exercise is adapted from Chapter 4, Exercise 13 in ISLR.                

#####################################################
# Part 1: Import Packages and Load Data             #
#####################################################

# Load needed R packages
library(ISLR)
library(MASS)
library(class)

# Load Weekly data from ISLR package
data(Weekly)


#####################################################
# Part 2: Visualize Data                            #
#####################################################

# Summarize Data 
summary(Weekly)

# Get correlation matrix for all variables except Direction
cor(Weekly[,-9]) 

# Make pairs plot to visualize relationships between variables
pairs(Weekly[,-9], pch=".") 

# Make pairs plot by group (Direction = Up or Down)
pairs(Weekly[,-9], pch=".", col=Weekly$Direction) 

# Make histograms of numeric variables
layout(rbind(1:4,5:8)) 
for(i in 1:8){
  hist(Weekly[,i], xlab=colnames(Weekly)[i], main=colnames(Weekly)[i])
}

# Make boxplots by Direction group
layout(rbind(1:4,c(5:7,0))) 
for(i in 2:8){
  boxplot(Weekly[,i]~Weekly$Direction, ylab=colnames(Weekly)[i], xlab="Direction")
}


#####################################################
# Part 3: Build Training and Test Datasets          #
#####################################################

# Set seed so we can get the same train/test split each time to reproduce results 
set.seed(921101)

# Randomly select 85% of the data for the training set, remaining 15% is test set 

# Number of train data
n0 <- round(nrow(Weekly)*.85) 

# Number of test data
n1 <- nrow(Weekly) - n0 

# Get indices for training set
trnum <- sample(1:nrow(Weekly), size=n0) 
train <- Weekly[trnum,] # Make train data
test <- Weekly[-trnum,] # Make test data


#####################################################
# Part 4: Model Fitting, Prediction, and Evaluation #
#####################################################

########################################
## Fitting the Full Model              #
########################################

# Start by fitting the full model (i.e. using all Lag variables and Volume) to 
# training data. We do this for:
#   1) Logistic Regression
#   2) Linear Discriminant Analysis (LDA)
#   3) Quadratice Discriminant Analysis (QDA)
#   4) K-Nearest Neighbors (kNN)


# 1) Logistic Regression - Full Model 

# Fit logistic regression to training data using `glm()` function
trainglm1 <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, 
                 data=train, family=binomial())

# Summarize logistic regression results
summary(trainglm1)

# - According to the model, Lag2 was the only significant variable. But overall,
#   we cannot see a variable that is specially good at classifying the response.


# 2) LDA - Full Model 

# Fit LDA to training data using `lda()` function
trainlda1 <- lda(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data=train)

# - No summary to output for lda
# - You can extract the means (trainlda1$means), priors (trainlda1$prior), 
#   scaling (trainlda1$scaling), and other attributes from trainlda1


# 3) QDA - Full Model 

# Fit QDA to training data using `lda()` function
trainqda1 <- qda(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data=train)

# - No summary to output for qda
# - You can extract the means (trainqda1$means), priors (trainqda1$prior), 
#   scaling (trainqda1$scaling), and other attributes from trainqda1


# 4) kNN - Full Model

# Make modified training and testing sets with just predictors of interest
trainK <- train[, c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume")]
testK <- test[,c("Lag1","Lag2","Lag3","Lag4","Lag5","Volume")]

# set another seed as there is randomness to kNN fit
set.seed(1101)

# Get the KNN model and predicted values for k=1
knnPred1 <- knn(trainK, testK, train$Direction, k=1) 


########################################
## Predicting & Evaluating Full Model  #
########################################

metrics.stock <- matrix(NA, nrow=2, ncol=5)
colnames(metrics.stock) <- c("logistic", "lda", "qda", "knn1", "knn20")
rownames(metrics.stock) <- c("error_rate_full_mod", "error_rate_reduced_mod")

#######################################
# 1) Logistic Regression - Full Model #
#######################################

# Get the predicted values for logistic regression
# Probabilities greater than 0.5 => TRUE (i.e. predict 'Up')
glmPred1 <- predict(trainglm1, test, type="response") > .5 

# Convert to a factor with 1 = "Up" and 2 = "Down"
glmPred1 <- as.factor(glmPred1) # Change the type of the data
levels(glmPred1) <- c("Down", "Up") 

# Get confusion matrix
glmT1 <- table(glmPred1, test$Direction)
glmT1

# - The rows show you the original data, and the columns show you the predictions
#   from the model you just made. As we can imagine, the numbers on the diagonals
#   shows how many predictors were correctly categorized. The numbers on off
#   diagonal are the predictions that the model failed.

# - There are two types of errors we can find from this confusion matrix. One
#   is "Type 1 error", which is the model classifying as "positive" when the 
#   value is actually negative. This is also called "False positive". When we
#   say that the "Direction" level "Up" is the categories we want to find, the
#   number of false positive will therefore be 56.

# get type 1 error rate
glmT1[2,1] / sum(glmT1[,1])

# - We can also find "Type 2 error", also called as "False negative" in this
#   confusion matrix. Type 2 error occurs when the model predicts as negative
#   when the true value is actually positive. In the same manner from type 1
#   error, the number of false negatives in this case will be 16.

# get type 2 error rate
glmT1[1,2] / sum(glmT1[,2])

# get overall error rate
metrics.stock[1,1] <- 1 - sum(diag(glmT1))/n1
metrics.stock


#######################
# 2) LDA - Full Model #
#######################

# Obtain the predicted values
ldaPred1 <- predict(trainlda1, test) 

# Get Confusion matrix
ldaT1 <- table(ldaPred1$class, test$Direction) 
ldaT1

# Get the overall error rate
metrics.stock[1,2] <- 1 - sum(diag(ldaT1))/n1
metrics.stock


#######################
# 3) QDA - Full Model #
#######################

# Obtain the predicted values
qdaPred1<-predict(trainqda1,test) 

# Get Confusion matrix
qdaT1 <- table(qdaPred1$class, test$Direction) 
qdaT1

# Get the error rate
metrics.stock[1,3] <- 1 - sum(diag(qdaT1))/n1
metrics.stock


#######################
# 4) kNN - Full Model #
#######################

# get confusion matrix
knnT1 <- table(knnPred1, test$Direction) 
knnT1

# Get the error rate
metrics.stock[1,4] <- 1 - sum(diag(knnT1))/n1
metrics.stock


# For comparison, let's also fit kNN with k=20

# set another seed as there is randomness to kNN fit
set.seed(1101)

# Get the KNN model and predicted values for k=1
knnPred1_k20 <- knn(trainK, testK, train$Direction, k=20) 

# get confusion matrix
knnT1_k20  <- table(knnPred1_k20, test$Direction) 
knnT1_k20 

# Get the error rate
metrics.stock[1,5] <- 1 - sum(diag(knnT1_k20 ))/n1
metrics.stock


#####################################################
## Fitting, Predicting, & Evaluating Reduced Model ##
#####################################################

# Now we'll fit a reduced model with just Lag1, Lag2, and Volume

##########################################
# 1) Logistic Regression - Reduced Model #
##########################################

# Fit logistic regression to training data using `glm()` function
trainglm2 <- glm(Direction~Lag1+Lag2+Volume, 
                 data=train, family=binomial())

# Summarize logistic regression results
summary(trainglm2)

# - According to the model, Lag2 was the only significant variable again. But overall,
#   we cannot see a variable that is specially good at classifying the response.

# Get the predicted values for logistic regression
# Probabilities greater than 0.5 => TRUE (i.e. predict 'Up')
glmPred2 <- predict(trainglm2, test, type="response") > .5 

# Convert to a factor with 1 = "Up" and 2 = "Down"
glmPred2 <- as.factor(glmPred2) # Change the type of the data
levels(glmPred2) <- c("Down","Up") 

# Get confusion matrix
glmT2 <- table(glmPred2, test$Direction)
glmT2

# get overall error rate
metrics.stock[2,1] <- 1 - sum(diag(glmT2))/n1
metrics.stock


##########################
# 2) LDA - Reduced Model #
##########################

# Fit LDA to training data using `lda()` function
trainlda2 <- lda(Direction~Lag1+Lag2+Volume, data=train)

# Obtain the predicted values
ldaPred2 <- predict(trainlda2, test) 

# Get Confusion matrix
ldaT2 <- table(ldaPred2$class, test$Direction) 
ldaT2

# Get the overall error rate
metrics.stock[2,2] <- 1 - sum(diag(ldaT2))/n1
metrics.stock


##########################
# 3) QDA - Reduced Model #
##########################

# Fit QDA to training data using `lda()` function
trainqda2 <- qda(Direction~Lag1+Lag2+Volume, data=train)

# Obtain the predicted values
qdaPred2 <- predict(trainqda2, test) 

# Get Confusion matrix
qdaT2 <- table(qdaPred2$class, test$Direction) 
qdaT2

# Get the error rate
metrics.stock[2,3] <- 1 - sum(diag(qdaT2))/n1
metrics.stock


##########################
# 4) kNN - Reduced Model #
##########################

# Make modified training and testing sets with just predictors of interest
trainK <- train[, c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume")]
testK <- test[,c("Lag1","Lag2","Lag3","Lag4","Lag5","Volume")]

# set another seed as there is randomness to kNN fit
set.seed(1101)

# Get the KNN model and predicted values for k=20
knnPred2 <- knn(trainK, testK, train$Direction, k=20) 

# get confusion matrix
knnT2 <- table(knnPred2,test$Direction) 
knnT2

# Get the error rate
metrics.stock[2,5] <- 1 - sum(diag(knnT2))/n1
metrics.stock



################################################################################
# Exercise: Predicting Gas Mileage                                             #
################################################################################

# This exercise is adapted from Chapter 4, Exercise 14 in ISLR.                
# In this problem, you will develop models (logistic regression, LDA, QDA, kNN) 
# to predict whether a given car gets high or low gas mileage based on the `Auto` 
# data set from the `ISLR` package.

#####################################################
# Part 1: Import & Clean Data                       #
#####################################################

# Load Auto data from ISLR package
data(Auto)

# Make response variable: mpg01, 0 = Low, 1 = High
Auto$mpg01 <- as.numeric(Auto$mpg > median(Auto$mpg)) #Make the response variable

# convert cylinders to a factor
Auto$cylinders <- as.factor(Auto$cylinders)


#####################################################
# Part 2: Visualize Data                            #
#####################################################

# Look at first few rows of the data 
head(Auto)

# Scatterplot matrix
pairs(Auto[, c(-8,-9, -10)], pch=19, col=Auto$mpg01+2, cex=.7) 

# Boxplots
layout(rbind(c(1,2,3), c(4,5,0)))
for(i in 3:7){
  boxplot(as.numeric(Auto[,i])~Auto$mpg01, ylab=colnames(Auto)[i], xlab="mpg01", 
          main=colnames(Auto)[i])
}


#####################################################
# Part 3: Build Training and Test Datasets          #
#####################################################

# Set seed so we can get the same train/test split each time to reproduce results 
set.seed(921101)

# Randomly select 85% of the data for the training set, remaining 15% is test set 

# Number of train data
n0 <- round(nrow(Auto)*.85) 

# Number of test data
n1 <- nrow(Auto) - n0 

# Get indices for training set
trnum <- sample(1:nrow(Auto), size=n0) 
train <- Auto[trnum,] # Make train data
test <- Auto[-trnum,] # Make test data


#####################################################
# Part 4: Model Fitting, Prediction, and Evaluation #
#####################################################

metrics.mpg <- matrix(NA, nrow=2, ncol=4)
colnames(metrics.mpg) <- c("logistic", "lda", "qda", "knn1")
rownames(metrics.mpg) <- c("error_rate_full_mod", "error_rate_reduced_mod")

########################################
## Fitting the Full Model              #
########################################

# Start by fitting the full model (i.e. using all Lag variables and Volume) to 
# training data. We do this for:
#   1) Logistic Regression
#   2) Linear Discriminant Analysis (LDA)
#   3) Quadratice Discriminant Analysis (QDA)
#   4) K-Nearest Neighbors (kNN)


#######################################
# 1) Logistic Regression - Full Model #
#######################################

# fit logistic regression with all predictors
trainglm1 <- glm(mpg01~cylinders+displacement+horsepower+weight+acceleration+
                 year, data=train, family=binomial()) 

summary(trainglm1)

# Get predictions
glmPred1 <- predict(trainglm1, test, type="response") > .5

# Change type of data into numeric vector
glmPred1 <- as.numeric(glmPred1) 

# Confusion matrix
glmT1 <- table(glmPred1,test$mpg01) 
glmT1

# Get the error rate
metrics.mpg[1,1] <- 1 - sum(diag(glmT1))/n1
metrics.mpg

#######################
# 2) LDA - Full Model #
#######################

# Fit LDA
trainlda1 <- lda(mpg01~cylinders+displacement+horsepower+weight+acceleration+
                 year, data=train) 

# Get the predicted values
ldaPred1 <- predict(trainlda1, test) 

# Confusion matrix
ldaT1 <- table(ldaPred1$class, test$mpg01) 
ldaT1

# Get the error rate
metrics.mpg[1,2] <- 1 - sum(diag(ldaT1))/n1
metrics.mpg


#######################
# 3) QDA - Full Model #
#######################

# Fit QDA
trainqda1 <- qda(mpg01~cylinders+displacement+horsepower+weight+acceleration+
                   year, data=train) 

# Get the predicted values
qdaPred1 <- predict(trainqda1, test) 

# Confusion matrix
qdaT1 <- table(qdaPred1$class, test$mpg01) 
qdaT1

# Get the error rate
metrics.mpg[1,3] <- 1 - sum(diag(qdaT1))/n1
metrics.mpg

#######################
# 4) kNN - Full model #
#######################

# Make modified training and test set without response
trainK <- train[, c("cylinders","displacement","horsepower","weight",
                    "acceleration", "year")] 
testK <- test[, c("cylinders","displacement","horsepower","weight",
                  "acceleration","year")] 

# set seed for reproducible results
set.seed(1101)

# fit knn with k = 1
trainknn1 <- knn(trainK, testK, train$mpg01, k=1)  

# get confusion matrix
knnT1 <- table(trainknn1, test$mpg01)
knnT1

# Get the error rate 
metrics.mpg[1,4] <- 1 - sum(diag(knnT1))/n1
metrics.mpg


# set seed for reproducible results
set.seed(1101)



########################################
## Fitting the Reduced Model           #
########################################

# Fit reduced models with 3 predictors: displacement, weight and horsepower

##########################################
# 1) Logistic Regression - Reduced Model #
##########################################

# fit logistic regression with all predictors
trainglm2 <- glm(mpg01~displacement+horsepower+weight, data=train, family=binomial()) 
summary(trainglm2)

# Get predictions
glmPred2 <- predict(trainglm2, test, type="response") > .5

# Change type of data into numeric vector
glmPred2 <- as.numeric(glmPred2) 

# Confusion matrix
glmT2 <- table(glmPred2,test$mpg01) 
glmT2

# Get the error rate
metrics.mpg[2,1] <- 1 - sum(diag(glmT2))/n1
metrics.mpg


##########################
# 2) LDA - Reduced Model #
##########################

# Fit LDA
trainlda2 <- lda(mpg01~displacement+horsepower+weight, data=train) 

# Get the predicted values
ldaPred2 <- predict(trainlda2, test) 

# Confusion matrix
ldaT2 <- table(ldaPred2$class, test$mpg01) 
ldaT2

# Get the error rate
metrics.mpg[2,2] <- 1 - sum(diag(ldaT2))/n1
metrics.mpg


##########################
# 3) QDA - Reduced Model #
##########################

# Fit QDA
trainqda2 <- qda(mpg01~displacement+horsepower+weight, data=train) 

# Get the predicted values
qdaPred2 <- predict(trainqda1, test) 

# Confusion matrix
qdaT2 <- table(qdaPred1$class, test$mpg01) 
qdaT2

# Get the error rate
metrics.mpg[2,3] <- 1 - sum(diag(qdaT2))/n1
metrics.mpg

##########################
# 4) kNN - Reduced model #
##########################

# Make modified training and test set without response and extra predictors
trainK <- train[, c("displacement","horsepower","weight")] 
testK <- test[, c("displacement","horsepower","weight")] 

# set seed for reproducible results
set.seed(1101)

# fit knn with k = 1
trainknn2 <- knn(trainK, testK, train$mpg01, k=1)  

# get confusion matrix
knnT2 <- table(trainknn2, test$mpg01)
knnT2

# Get the error rate 
metrics.mpg[2,4] <- 1 - sum(diag(knnT2))/n1
metrics.mpg



