################################################################################
# SAIG Machine Learning Short Course                                           #
# Chapter 6 - Linear Model Adjustments                                         #
# Instructor Code                                                              #
#                                                                              #                                                             #
# Last updated: 1/15/2024                                                      #
################################################################################

# The following exercises, code, and explanations are adapted from An 
# Introduction to Statistical Learning with Applications in R (ISLR) 2nd Edition
# (James, Witten, Hastie, Tibshirani, and Taylor 2023).

################################################################################
# Exercise: Model Selection                                                    #
################################################################################

# This exercise is adapted from Chapter 6, Exercise 8 in ISLR.                

#####################################################
# Import Packages                                   #
#####################################################

library(leaps)
library(glmnet)
library(ISLR)
library(pls)
library(MASS)

##############################################################
# Part 1: Create simulated dataset                           #
##############################################################

set.seed(921101)

x <- rnorm(100) # predictor variable
err <- 5*rnorm(100) # Error term
y <- 5 + 5*x + 5*x^2 + 5*x^3 + err # response

# All the coefficients are 5.

plot(x, y, pch=19, cex=1.5, col="blue", main="Plot of y")

##############################################################
# Part 2: Best Subset, Forward, Backward, Stepwise Selection #
##############################################################

#########################
# Best Subset Selection #
#########################

xydat <- data.frame(y)

# Make a data that contains y, x, ..., x^10
for(i in 1:10){
  xydat <- data.frame(xydat, x^i) 
}
# Size of the matrix
dim(xydat) 
# Change the names of the columns
colnames(xydat) <- c("y", paste("x", 1:10, sep="")) 

# Getting subset of predictors using exhaustive method for regression
xydatReg <- regsubsets(y~., data=xydat, nvmax=10) 

# Summary of the model when using exhaustive method
xydatRegSum <- summary(xydatReg)
xydatRegSum
# - When using exhaustive method, we can see that it first selects the x^3,
#   then x^2, and x^1 as the predictor for the model. So we can see that the
#   the model succeeded to find the right predictors.

# Linear model of the data
lm(y ~ x1 + x2 + x3, data=xydat)

### Now, we make the plots.   

numpred <- 1:10
layout(cbind(c(1,2,3)))

xydatRegSum$bic
plot(numpred, xydatRegSum$bic, type="b", col="skyblue2", pch=19, lwd=2,
     main="Plots of BIC", xlab="Number of predictors", ylab="BIC")
# - In case of BIC, we can see that the value is the smallest when we are 
#   selecting 3 variables. And in case of exhaustive model, it selects the 
#   first 3 variables. 

xydatRegSum$cp
plot(numpred, xydatRegSum$cp, type="b", col="orange2", pch=19, lwd=2,
     main="Plots of cp", xlab="Number of predictors", ylab="cp")
# - In case of Mallow's cp, we can see that the cp is the smallest when 
#   selecting 3 variables. The model with lower cp is considered to be the
#   desirable one.

xydatRegSum$adjr2
plot(numpred, xydatRegSum$adjr2, type="b", col="green3", pch=19, lwd=2,
     main="Plots of Adjusted R", xlab="Number of predictors", ylab="R^2")
# - In case of adjusted R square, we can see that the value is the largest when
#   we are selecting 3 variables in the model. Like other two criteria, we can
#   get the best model when selecting 3 variables. 



#########################
# Forward Selection     #
#########################

# Getting subset of predictors using foward method for regression
xydatRegF <- regsubsets(y~., data=xydat, method="forward", nvmax=10)

# Summary of the model when using forward method
xydatRegFSum <- summary(xydatRegF)
xydatRegFSum

# - When using forward selection, we can see that it first selects the x^3,
#   then x^2, and x^1 as the predictor for the model. This step complies to
#   the way exhaustive model did. However, we can see there is a difference
#   when it comes to selecting subsets with 6 variables. In the exhaustive
#   model, it will discard the 8th variable, which was selected in the 5th
#   subset, and will select x^5 and x^7 instead. Forward selection keeps 
#   x^8 and select x^10. This is because exhaustive way would search the best
#   model under every possible case, but forward selection doesn't drop the
#   predictor once it's included.

### Now, we make the plots. ###
numpred<-1:10
layout(cbind(c(1,2,3)))

xydatRegFSum$bic
plot(numpred, xydatRegFSum$bic, type="b", col="skyblue2", pch=19, lwd=2,
     main="Plots of BIC", xlab="Number of predictors", ylab="BIC")
# - In case of BIC, we can see that the value is the smallest when we are 
#   selecting 3 variables. 

xydatRegFSum$cp
plot(numpred, xydatRegFSum$cp, type="b", col="orange2", pch=19, lwd=2,
     main="Plots of cp", xlab="Number of predictors", ylab="cp")
# - In case of Mallow's cp, we can see that the cp is the smallest when 
#   selecting 3 variables. 

xydatRegFSum$adjr2
plot(numpred, xydatRegFSum$adjr2, type="b",col="green3", pch=19,lwd=2,
     main="Plots of Adjusted R", xlab="Number of predictors", ylab="R^2")
# - In case of adjusted R square, the value is the largest when we are using
#   3 variables. The model is considered to be better when it has larger
#   adjusted R square.

#########################
# Backward Elimination  #
#########################

# Getting subset of predictors using backward method for regression
xydatRegB <- regsubsets(y~., data=xydat, method="backward", nvmax=10)

# Summary of the model when using backward method
xydatRegBSum <- summary(xydatRegB)
xydatRegBSum

# - When using backward elimination, the first three models with one, two, and
#   three predictors are the same as other exhaustive and forward selection. 
#   However, it will include x^7 when having for 4-predictor subset, instead 
#   of x^8 as the other two methods. 


### Now, we make the plots. ###
numpred <- 1:10
layout(cbind(c(1,2,3)))

xydatRegBSum$bic
plot(numpred, xydatRegBSum$bic, type="b", col="skyblue2", pch=19, lwd=2,
     main="Plots of BIC", xlab="Number of predictors", ylab="BIC")
# - In case of BIC, we can see that the value is the smallest when we are 
#   selecting 3 variables. 

xydatRegBSum$cp
plot(numpred, xydatRegBSum$cp, type="b", col="orange2", pch=19, lwd=2,
     main="Plots of cp", xlab="Number of predictors", ylab="cp")
# - In case of Mallow's cp, we can see that the cp is the smallest when 
#   selecting 3 variables. 

xydatRegBSum$adjr2
plot(numpred, xydatRegBSum$adjr2, type="b", col="green3", pch=19, lwd=2,
     main="Plots of Adjusted R", xlab="Number of predictors", ylab="R^2")
# - In case of adjusted R square, the value is the largest when we are using
#   3 variables. 

##############################################################
# Part 3: LASSO                                              #
##############################################################

set.seed(921101)

trainidx <- sample(100, 70, replace=FALSE)
lambdas <- 10^seq(3, -2, l=100) # Possible values of lambda
Xdat <- xydat[,-1] # The model matrix
Xdat <- as.matrix(Xdat)

# Get the Lasso model
xydatLasso <- glmnet(Xdat[trainidx,], y[trainidx], alpha=1, lambda=lambdas)

# Draw the plot of the coefficients
plot(log(lambdas), xydatLasso$beta[3,], type="l", col="skyblue2",
     main="Plot of Coefficients")
lines(log(lambdas), xydatLasso$beta[2,], type="l", col="orange2")
lines(log(lambdas), xydatLasso$beta[1,], type="l", col="green3")
legend("topright", legend=c("X^3", "X^2", "X"),
       fill=c("skyblue2", "orange2", "green3")) # Draw a legend box

# Cross validation for Lasso
xydatLassoCV <- cv.glmnet(Xdat, y, alpha=1, lambda=lambdas) 

# Get MSE for the best lambda case #
xydatLassoCV$lambda.min # The lambda that returns the best result
xydatLassoPred <- predict(xydatLasso, s=xydatLassoCV$lambda.min,
                        newx=Xdat[-trainidx,])
mean((xydatLassoPred - y[-trainidx])^2)

# Now, get the plot of MSE
xyLassoMSEs <- numeric(100)
for(i in 1:100){
  xydatLassoPred1 <- predict(xydatLasso, s=lambdas[i],
                           newx=Xdat[-trainidx,])
  xyLassoMSEs[i] <- mean((xydatLassoPred1 - y[-trainidx])^2)
}
summary(xyLassoMSEs)
plot(log(lambdas), xyLassoMSEs, type="l", col="skyblue2", main="MSE plot")
abline(v=log(xydatLassoCV$lambda.min), col="red", lty=2)

# We can also get the coefficients with the cross validation
xydatLassoC <- glmnet(Xdat, y, alpha=1, lambda=lambdas)
xydatLassoCoefB <- predict(xydatLassoC, type="coefficients", s=xydatLassoCV$lambda.min)
xydatLassoCoefB
# Get the predicted parameters when the model using best lambda, applying to 
# the test set

xydatLassoCoef <- predict(xydatLassoC, type="coefficients")
plot(log(lambdas), xydatLassoCoef[4,], type="l", col="skyblue2", lwd=2)
lines(log(lambdas), xydatLassoCoef[3,] , type="l", col="orange2", lwd=2)
lines(log(lambdas), xydatLassoCoef[2,], type="l", col="green3", lwd=2)
# Draw the plots of estimated coefficients by lambdas
abline(v=log(xydatLassoCV$lambda.min), col="red", lty=2)
legend("topright", legend=c("X^3","X^2","X"),
       fill=c("skyblue2", "orange2", "green3")) # Draw a legend box



################################################################################
# Exercise: College Applications                                               #
################################################################################

# This exercise is adapted from Chapter 6, Exercise 9 in ISLR.       

data(College)
n0 <- nrow(College)
set.seed(921101)

trainidx2<-sample(n0,n0*.8,replace=FALSE) # Select 80% of the data as train set
train2<-College[trainidx2,] # Train set
test2<-College[-trainidx2,] # Test set
train2$Private<-as.factor(train2$Private)
levels(train2$Private)<-c(0,1)
test2$Private<-as.factor(test2$Private)
levels(test2$Private)<-c(0,1)


CollegeLM<-lm(Accept~.,data=train2) 
# Make a linear model where the response is "Accept"
summary(CollegeLM) # Summary of the linear model
# - 9 predictors turned out to be significant under alpha=.05 out of 17.

CollegeLMPred<-predict(CollegeLM,test2,type="response")
mean((CollegeLMPred-test2$Accept)^2) # MSE


train2m<-train2[,-3] # Make a matrix model without the response
train2m<-as.matrix(train2m) # Make it as a matrix form
train2M<-matrix(0,nrow(train2m),17)
for(i in 1:17){train2M[,i]<-as.numeric(train2m[,i])}
colnames(train2M)<-colnames(train2m)

test2m<-test2[,-3] # Make a matrix model without the response
test2m<-as.matrix(test2m) # Make it as a matrix form
test2M<-matrix(0,nrow(test2m),17)
for(i in 1:17){test2M[,i]<-as.numeric(test2m[,i])}
colnames(test2M)<-colnames(test2m)


lambdas3<-10^seq(-3,5,l=100)

ColRidge<-glmnet(train2M,train2$Accept,alpha=0,lambda=lambdas3)
# Get the Ridge Regression model 

ColRidgeCV<-cv.glmnet(train2M,train2$Accept,alpha=0,lambda=lambdas3) 
# Cross validation for Ridge Regression

# Get MSE for the best lambda case #
ColRidgeCV$lambda.min # The lambda that returns the best result
ColRidgePred<-predict(ColRidge,s=ColRidgeCV$lambda.min,newx=test2M)
mean((ColRidgePred-test2$Accept)^2)

# Coefficients #

ColRidgeCoefB<-predict(ColRidge,type="coefficients",s=ColRidgeCV$lambda.min)
ColRidgeCoefB

# - All of the variables turned out to be significant, as ridge regression
#   model should return. The variable with largest absolute coefficients were
#   "Private", "Top10Perc" and "Top20Perc".


ColLasso<-glmnet(train2M,train2$Accept,alpha=1,lambda=lambdas3)
# Get the Lasso model 

ColLassoCV<-cv.glmnet(train2M,train2$Accept,alpha=1,lambda=lambdas3) 
# Cross validation for Lasso

# Get MSE for the best lambda case #
ColLassoCV$lambda.min # The lambda that returns the best result
ColLassoPred<-predict(ColLasso,s=ColLassoCV$lambda.min,newx=test2M)
mean((ColLassoPred-test2$Accept)^2) # MSE

# Coefficients #

ColLassoCoefB<-predict(ColLasso,type="coefficients",s=ColLassoCV$lambda.min)
ColLassoCoefB

# - We can see that the Lasso model's result and Ridge regression model have
#   little difference. In this case, it seems that every variable wasn't zero.


ColPCR<-pcr(Accept~., data=train2,scale=TRUE,validation="CV")
# Build the PCR model with the train set
summary(ColPCR)
# Summary of PCR

validationplot(ColPCR,val.type="MSEP") #
# -From the model and summary, we can see that the model has lesser MSE as
#  it is added with more predictors.

ColPCRPred=predict(ColPCR,test2,ncomp=16)
mean((ColPCRPred-test2$Accept)^2) # MSE


ColPLS<-plsr(Accept~., data=train2,scale=TRUE,validation="CV")
# Build the PLS model with the train set
summary(ColPLS)
# Summary of PLS

validationplot(ColPLS,val.type="MSEP") #
# -From the model and summary, we can see that the model stays pretty constant 
#  from having 7 components.

ColPLSPred=predict(ColPLS,test2,ncomp=7)
mean((ColPLSPred-test2$Accept)^2) # MSE



# - From the results above, we can see that the MSE of the three methods are
#   all around 240,000 and 270,000. 
#   We could see from the summary of this model that it was being unstable; the
#   Under this specific seed, we could see that the partial least squares
#   method is slightly better than other models.


################################################################################
# Exercise: Predicting Crime Rates                                             #
################################################################################

# This exercise is adapted from Chapter 6, Exercise 11 in ISLR.       

### Handling data ###

data(Boston)
set.seed(921101)
n0<-nrow(Boston)

trainidx3<-sample(n0,n0*.8,replace=FALSE) # Select 80% of the data as train set
train3<-Boston[trainidx3,] # Train set
test3<-Boston[-trainidx3,] # Test set
train3M<-train3[,-1] # Model matrix of train set
train3M<-as.matrix(train3M) # Make it a matrix form
test3M<-test3[,-1] # Model matrix of test set
test3M<-as.matrix(test3M) # Make it a matrix form


### Best subset of variables ###

# Exhaustive method

BosReg1<-regsubsets(crim~.,data=train3,nvmax=10) 
# Getting subset of predictors using exhaustive method for regression
BosReg1Sum<-summary(BosReg1)
BosReg1Sum
BosReg1Sum$bic # Smallest when having 2 variables
BosReg1Sum$adjr2 # Largest when having 9 variables
BosReg1Sum$cp # Smallest when having 7 variables

# Forward Selection

BosReg2<-regsubsets(crim~.,data=train3,nvmax=10,method="forward") 
# Getting subset of predictors using forward selection for regression
BosReg2Sum<-summary(BosReg2)
BosReg2Sum
BosReg2Sum$bic # Smallest when having 2 variables
BosReg2Sum$adjr2 # Largest when having 9 variables
BosReg2Sum$cp # Smallest when having 8 variables

# Backward Elimination

BosReg3<-regsubsets(crim~.,data=train3,nvmax=10,method="backward") 
# Getting subset of predictors using backward elimination for regression
BosReg3Sum<-summary(BosReg3)
BosReg3Sum
BosReg3Sum$bic # Smallest when having 4 variables
BosReg3Sum$adjr2 # Largest when having 9 variables
BosReg3Sum$cp # Smallest when having 7 variables

# Stepwise Regression

BosReg4<-regsubsets(crim~.,data=train3,nvmax=10,method="seqrep") 
# Getting subset of predictors using exhaustive method for regression
BosReg4Sum<-summary(BosReg4)
BosReg4Sum
BosReg4Sum$bic # Smallest when having 2 variables
BosReg4Sum$adjr2 # Largest when having 7 variables
BosReg4Sum$cp # Smallest when having 7 variables

# - Different subset selection methods have difference choice of variables to
#   select. Also, the number of variables are different too. BIC usually requires
#   us to select less variables, while adjusted R squared suggests to select
#   more variables.


### Linear Regression using all variables

BostonLM<-lm(crim~.,data=train3) 
# Make a linear model where the response is "crim"
summary(BostonLM) # Summary of the linear model
# - The only variables that seemed to be significant were zn, dis, rad and
#   medv among 13 predictors.

BostonLMPred<-predict(BostonLM,test3,type="response")
mean((BostonLMPred-test3$crim)^2) # MSE


### Ridge Regression model

lambdas4<-10^seq(-3,5,l=200)

BosRidge<-glmnet(train3M,train3$crim,alpha=0,lambda=lambdas4)
# Get the Ridge Regression model 
BosRidgeCV<-cv.glmnet(train3M,train3$crim,alpha=0,lambda=lambdas4) 
# Cross validation for Ridge Regression

# Get MSE for the best lambda case #
BosRidgeCV$lambda.min # The lambda that returns the best result
BosRidgePred<-predict(BosRidge,s=BosRidgeCV$lambda.min,newx=test3M)
mean((BosRidgePred-test3$crim)^2) # MSE

# Coefficients #
BosRidgeCoefB<-predict(BosRidge,type="coefficients",s=ColRidgeCV$lambda.min)
BosRidgeCoefB


### Lasso Model


BosLasso<-glmnet(train3M,train3$crim,alpha=1,lambda=lambdas4)
# Get the Lasso model 

BosLassoCV<-cv.glmnet(train3M,train3$crim,alpha=1,lambda=lambdas4) 
# Cross validation for Lasso

# Get MSE for the best lambda case #
BosLassoCV$lambda.min # The lambda that returns the best result
BosLassoPred<-predict(BosLasso,s=BosLassoCV$lambda.min,newx=test3M)
mean((BosLassoPred-test3$crim)^2) # MSE

# Coefficients #
BosLassoCoefB<-predict(BosLasso,type="coefficients",s=BosLassoCV$lambda.min)
BosLassoCoefB


### Principal Components Regression

BosPCR<-pcr(crim~.,data=train3,scale=TRUE,validation="CV")
# Build the PCR model with the train set
summary(BosPCR)
# Summary of PCR

validationplot(BosPCR,val.type="MSEP") # Get the cross validation MSE plot

BosPCRPred<-predict(BosPCR,test3,ncomp=8)
mean((BosPCRPred-test3$crim)^2) # MSE


### Partial Least Squares


BosPLS<-plsr(crim~.,data=train3,scale=TRUE,validation="CV")
# Build the PLS model with the train set
summary(BosPLS)
# Summary of PLS

validationplot(BosPLS,val.type="MSEP") # Get the cross validation MSE plot

BosPLSPred=predict(BosPLS,test3,ncomp=8)
mean((BosPLSPred-test3$crim)^2) # MSE

# - Among linear models, ridge regression, Lasso, PCR, and PLS method, the 
#   PLS method had the smallest MSE.
