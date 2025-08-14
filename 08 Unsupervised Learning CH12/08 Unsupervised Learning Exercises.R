
################################################################################
# File Name: 08 Unsupervised Learning Exercises.R                                 
# Description: Exercises for the unsupervised learning module of the ML shortcourse. 
# Author: SAIG, Dylan Steberg
# Date: 07/25/25
# Course: Machine Learning Short Course
# R version: 4.3.0
# Package(s): corrplot, dendextend, and clue
# Dataset(s): CountryData.csv, OliveOils.csv
#
# Revision History:
#     07/25/25: R code created to match Python code
################################################################################

# The following exercises, code, and explanations are adapted from An 
# Introduction to Statistical Learning with Applications in R (ISLR) 2nd Edition
# (James, Witten, Hastie, Tibshirani, and Taylor 2023).

################################################################################
######################### Exercise: Countries Dataset ##########################
################################################################################

# First we load the required R packages.
library(corrplot)
library(dendextend)
library(clue)

# Thise exercise uses the "Countries: data set, which contains information on 
# 167 countries. It contains ten variables:
# - country: Name of the country
# - child_mort: Death of children under 5 years of age per 1000 live births
# - exports: Exports of goods and services per capita. Given as %age of the GDP
#     per capita
# - health: Total health spending per capita. Given as %age of GDP per capita
# - imports: Imports of goods and services per capita. Given as %age of the GDP
#     per capita
# - income: Net income per person
# - inflation: The measurement of the annual growth rate of the Total GDP
# - life_expec: The average number of years a new born child would live if the 
#     current mortality patterns are to remain the same
# - total_fer: The number of children that would be born to each woman if the 
#     current age-fertility rates remain the same.
# - gdpp: The GDP per capita. Calculated as the Total GDP divided by the total 
#     population.
 
# The code below loads the `Countries` data set from a csv file.
CountryData = read.csv("CountryData.csv")

# Before we jump into the analysis, let's inspect the data using the `head` 
# function. We can also look at the correlation between the variables. Using the 
# `corrplot` function from the corrplot package gives a nice heat map of the 
# correlations.
head(CountryData, n = 5)
corrplot(cor(CountryData[, -1]), method = "number", type = "lower")


# Notice there is multicolinearity present in our data. Examining some of the 
# correlations, we see:
# - total_fer has a high positive correlation with child_mort (0.85)
# - gdpp has a high positive correlation with income (0.9)
# - life_expec has a high negative correlation with child_mort and 
#     total_fer (-0.89)

# We can visualize these correlations via the following scatterplot matrix.
pairs(CountryData[, -1])

# We standardize the data here which is used farther down.
country = CountryData$country
CountryScaled = data.frame(scale(CountryData[, -1], center = TRUE))
CountryScaled = cbind(country, CountryScaled)

# It is always good practice to inspect our data sets.
head(CountryScaled, n = 5)

################################################################################
############ Part 1: Principal Component Analysis for Visualization ############
################################################################################

# This part is adapted from Chapter 12, Lab 5 in ISLR (which uses a different
# data set). We'll perform PCA on the "Countries" dataset and look at some 
# visualizations. Remember, we are not concerned with prediction here (in fact 
# we have no predictor variable!)

# The function `prcomp` is used to perform the principal component analysis. It 
# is recommended to standardize the variables before performing PCA so we will 
# use our scaled data. 
 
# Note: `prcomp` does have a `center` and `scale.` argument that can be used 
# rather than supplying scaled data.
pca_countries = prcomp(as.matrix(CountryScaled[, -1]))

# We can get the extract the scores after the PCA has been fit. They are held in
# the x component of our prcomp object.
head(pca_countries$x, n = 5)

# The loadings can be extracted as well. They are held in the rotation component
# of our prcomp object.
pca_countries$rotation

# Using `summary` on our prcomp object gives the proportion of variation 
# explained by each principal component and the cumulative variation explained
# as additional components are added. They can also be accessed separately.
summary(pca_countries)
summary(pca_countries)$importance[2, ] # proportion of variation explained
summary(pca_countries)$importance[3, ] # cumulative variation explained

# It is easier to visualize these values using plots.
plot(summary(pca_countries)$importance[2, ], ty = "l", main = "Scree Plot",
     xlab = "Principal Component", ylab = "Explained Variance")

plot(summary(pca_countries)$importance[3, ], ty = "s", ylab = "Explained Variance",
     main = "Cumulative Variance Plot", xlab = "Principal Component")
axis(1, at = 1:9); abline(h = 0.9, col = "orange") # tick marks and line at 0.9

# If we wanted to find the "optimal" number of PCs, we would use these two plots. 
# It looks like there are both elbows at 2 and 6 components, although that might 
# be two few and too many components respectively. It takes 5 components to 
# retain >90% of the variation in our original data so it would probably be best
# to consider 5 components.

# We can create a biplot to visualize the first two principal components.
# Changing the values for i and j will graph other pairs of principal components.
i = 1; j = 2 
biplot(x = data.frame(pca_countries$x[, i], pca_countries$x[, j]), cex = 0.8,
       y = data.frame(pca_countries$rotation[,1], pca_countries$rotation[, 2]),
      col = c("steelblue", "darkorange"), xlab = paste("Principal Component", i), 
      ylab = paste("Principal Component", j), arrow.len = 0,)

# Notice there are some outliers in our data (92, 99, and 134)

################################################################################
########## Part 2: Classifying Countries Using Hierarchical Clustering #########
################################################################################

# This part is adapted from Chapter 12, Lab 5 in ISLR. We implement hierarchical
# clustering on the "Countries" data set. We will plot the dendrograms using 
# different linkages. Euclidean distance is used as the dissimilarity measure. We 
# will also investigate how standardizing affects the clustering.

# The hierarchical clustering will be done using the `hclust` function. Important
# parameters include:
# - d: A dissimilarity structure (such as a distance matrix from `dist`)
# - method: Which agglomeration method (linkage) to use, includes options such as:
#     - ward.D: Minimizes the variance of the clusters being merged,
#     - average: Uses the average distances of each observation,
#     - complete: Uses the maximum distances between all observations, and
#     - single: Uses the minimum distance between all observations

# We start by doing full hierarchical clustering on the orignal data using
# complete linkage.
HClustFull_Complete = hclust(dist(CountryData[, -1]), method = "complete")

# The following code prints the countries in each cluster. We see that there are 
# 167 clusters that include only one country each.
for (i in 1:167) {
  cat("Cluster", i, "contains:", country[HClustFull_Complete$order == i], "\n\n")
}

# We can plot the dendrogram using the base R `plot` function.
plot(HClustFull_Complete, labels = country, xlab = "", ylab = "", sub = "",
     cex = 0.6, xaxt = "n", main = "Full dendrogram")

# Now let's make it so we have only three clusters. We can do this using the
#  `cutree` function and specifying our number of clusters.
HClustThree_Complete = cutree(HClustFull_Complete, k = 3)

# The following code prints the countries in the three clusters.
for (i in 1:3) {
  cat("Cluster", i, "contains:", country[HClustThree_Complete == i], "\n\n")
}

# There is not a great way to graph the three clusters using base R plotting. 
# A line can be added showing where the cut is made and that is all. The y-axis
# number really doesn't mean much and it is essentially guess and check to get
# the line in the correct spot.
plot(HClustFull_Complete, labels = country, xlab = "", ylab = "", sub = "",
     cex = 0.6, xaxt = "n", main = "Full dendrogram with cut at 3 clusters")
abline(h = 80000, lty = 2, col = "red")

# Using the "dendextend" package allows us to color the leaf nodes by the 
# cluster they correspond to. While this is still not great, it is an 
# improvement. It can be somewhat tough to get the labels and colors ordered
# correctly if each group requires a specific color.
ThreeClustDend = as.dendrogram(HClustFull_Complete)
group_map = order.dendrogram(ThreeClustDend)
color_map = c("1" = "green", "2" = "red", "3" = "orange")
labels_colors(ThreeClustDend) = color_map[HClustThree_Complete][group_map]
plot(ThreeClustDend, xlab = "", ylab = "", sub = "",
     cex = 0.6, xaxt = "n", main = "Dendrogram for 3 clusters")

# We can also change the linkage which will affect our clustering. Let's try 
# single linkage, but feel free to explore the other linkages as well. We will
# cut the tree at 3 clusters once again.
HClustFull_Single = hclust(dist(CountryData[, -1]), method = "single")
HClustThree_Single = cutree(HClustFull_Single, k = 3)

# We can print the countries in each cluster. Since we did full clustering here, 
# there are 167 clusters that include only one country each.
for (i in 1:3) {
  cat("Cluster", i, "contains:", country[HClustThree_Single == i], "\n\n")
}

# This time it creates two clusters with a single country (Luxembourg and Qatar)
# and puts the other 165 countries into a single cluster.

# Once again we can plot the dendrogram
ThreeClustDend = as.dendrogram(HClustFull_Single)
group_map = order.dendrogram(ThreeClustDend)
color_map = c("1" = "green", "2" = "red", "3" = "orange")
labels_colors(ThreeClustDend) = color_map[HClustThree_Single][group_map]
plot(ThreeClustDend, xlab = "", ylab = "", sub = "",
     cex = 0.6, xaxt = "n", main = "Dendrogram for 3 clusters")

# Now we will cluster with the scaled data to see how that yields different 
# results. We will look at three clusters and plot the dendrogram. We are using
# complete linkage and will compare it to the hierarchical clustering with three
# clusters and complete linkage with the orginal data.
HClustFull_CompleteScaled = hclust(dist(CountryScaled[, -1]), method = "complete")
HClustThree_CompleteScaled = cutree(HClustFull_CompleteScaled, k = 3)

# We can print the countries in each cluster. Since we did full clustering here, 
# there are 167 clusters that include only one country each.
for (i in 1:3) {
  cat("Cluster", i, "contains:", country[HClustThree_CompleteScaled == i], "\n\n")
}

# Once again we can plot the dendrogram
ThreeClustDend = as.dendrogram(HClustFull_CompleteScaled)
group_map = order.dendrogram(ThreeClustDend)
color_map = c("1" = "green", "2" = "red", "3" = "orange")
labels_colors(ThreeClustDend) = color_map[HClustThree_CompleteScaled][group_map]
plot(ThreeClustDend, xlab = "", ylab = "", sub = "",
     cex = 0.6, xaxt = "n", main = "Dendrogram for 3 clusters")

# We see that this dendrogram and clustering much different from the previous 
# example.

################################################################################
############### Part 3: Investigating K-means Clustering with PCs ############## 
################################################################################

# This part implements clustering on the `Countries` data set using K-means and 
# principle components. The clustering will be done using the `kmeans` functions. 
# Remember that in K-means we must specify the number of clusters. We will 
# consider splitting our data into 3 clusters, but you can change this number to 
# see how things change. This part will somewhat show the impact that scaling 
# has on clustering as well.

# Note: `nstart` is how many times the algorithm is run with different centroid 
# seeds (that is, different random assignment in the first step).

# First, we will apply K-means clustering using principle components built using 
# the unscaled `Countries` data. We are considering 5 principle components since 
# that is the smallest number that retained >90% of the variation in our original 
# data. The PCA is fit below.
pca_countries = prcomp(as.matrix(CountryData[, -1]), rank. = 5)

# We run k-means and get the labels from the group.
kmeans_unscaled = kmeans(pca_countries$x, centers = 3, nstart = 20)
clust_labels_unscaled = kmeans_unscaled$cluster

# Here we can visualize the clusters using the first two principle components. 
# It is important to understand that this is a two-dimensional look at the 
# clusters (which are actually 5-dimensional). We see that there is good 
# seperation into the three clusters.
plot(pca_countries$x[, 1:2], col = clust_labels_unscaled, pch = 16)

# Next, we will apply K-means clustering using principle components built using 
# the scaled "Countries" data. Once again we are considering 5 principle 
# components. The PCA is fit below.
pca_countries_scaled = prcomp(as.matrix(CountryScaled[, -1]), rank. = 5)

# We run k-means and get the labels from the group.
kmeans_scaled = kmeans(pca_countries_scaled$x, centers = 3, nstart = 20)
clust_labels_scaled = kmeans_scaled$cluster

# Again we visualize the clusters using the first two principle components. We 
# see that the clusters are very different from the unscaled clusters. While 
# this may seem worse than the in the unscaled case above, we still see very 
# good separation in our clusters (it is also only one way to "look down" on the 
# five dimensions).
plot(pca_countries_scaled$x[, 1:2], col = clust_labels_scaled, pch = 16)

# Lets inspect our clusters a little more in depth. We will start with counts 
# and grouped averages of the original variables.
table(clust_labels_scaled)
aggregate(CountryData[, 2:10], list(clust_labels_scaled), mean)[, -1]

# It looks like we have pretty good separation here as the means are all fairly 
# different. If we inspect the clusters we infer what these clusters might be. 
# We see that there the clusters are:
# - Countries with very high GDP, exports, imports, income, health, and life 
#     expectancy and very low child mortality, inflation, and total fertility. 
# - Countries with low GDP, exports, imports, income, health, and life 
#     expectancy and high child mortality, inflation, and total fertility.
# - Countries that fall in between the other two groups.

# We can describe these clusters as developed, least developed, and developing 
# countries respectively.

# Note: Due to label switching, the three groups might change if run again. For
# example, the developing countries may be cluster 3 in one run of the algorithm
# and cluster 1 in another run. While the cluster stays the same, the label may
# change.

# Finally we can look at the countries in these group to see if they make sense.
for (i in 1:3) {
  cat("Cluster", i, "contains:", country[clust_labels_scaled == i], "\n\n")
}

################################################################################
##################### Part 4: Does Clustering Really Work? #####################
################################################################################

# This exercise will look at the performance of clustering methods on data that 
# is actually clustered. Both unscaled K-means and scaled K-means will be fit. 
# Remember that clustering is **unsupervised**. In practice we don't know the 
# true cluster labels, we but will use the true cluster labels here to show that 
# the methods actually work.

# We will consider the `OliveOil` data set which contains 572 observations 
# corresponding to olive oils from 9 different regions of Italy. The variables 
# include Area.name (the label) which is the region the olive oil is from and 
# eight measurements on fatty acid contained in the oils. The goal is to cluster 
# the oils into regions based on the fatty acid measurements.

# The following code loads the `OliveOil` data set from a csv file.
OliveOil = read.csv("OliveOils.csv")

# The nine different regions are given by the code below.
regions = unique(OliveOil$Area.name)
regions

# Here we inspect the first few rows of the data set.
head(OliveOil, n = 10)

# We standardize the data and inspect the first few rows once again.
region_vals = OliveOil$Area.name
OliveOilScaled = data.frame(scale(OliveOil[, -1], center = TRUE))
OliveOilScaled = cbind(region_vals, OliveOilScaled)
head(OliveOilScaled, n = 10)  
  
# First, we will apply K-means clustering to the original (unscaled) "OliveOil" 
# data. Remember, in kmeans we have to specify how many clusters we want. 
# Specifying the number of clusters is not trivial in most cases. We do not
# touch on methods for determining the number of clusters in this course. For
# example, there are certain metrics that can be used.
kmeans_unscaled = kmeans(OliveOil[, -1], centers = 9, nstart = 20)
clust_labels_unscaled = kmeans_unscaled$cluster
  
# Label switching is a big issue here as R has no idea that "North-Apulia" is 
# the first group in the data set which means "North-Apulia" might not be the 
# first cluster. The following code remedies this by re-labeling the predicted 
# labels.

# Note: This is not a great fix as we **should not** apply classification 
# evaluation measures to clustering.
unscaled_conf_mat = table(clust_labels_unscaled, region_vals)
aligned_preds = solve_LSAP(unscaled_conf_mat, maximum = TRUE)[clust_labels_unscaled]
unscaled_conf_mat = table(aligned_preds, region_vals)

# Here we can look at a confusion matrix and the accuracy of our clustering.
unscaled_conf_mat
cat("Accuracy: ", round(sum(diag(unscaled_conf_mat))/nrow(OliveOilScaled), 3))

# Next, we apply K-means clustering to the scaled "OliveOil" data.
kmeans_scaled = kmeans(OliveOilScaled[, -1], centers = 9, nstart = 20)
clust_labels_scaled = kmeans_scaled$cluster

# We once again fix the labels.
scaled_conf_mat = table(clust_labels_scaled, region_vals)
aligned_preds = solve_LSAP(scaled_conf_mat, maximum = TRUE)[clust_labels_scaled]
scaled_conf_mat = table(aligned_preds, region_vals)

# Here we can look at a confusion matrix and the accuracy of our clustering.
scaled_conf_mat
cat("Accuracy: ", round(sum(diag(scaled_conf_mat))/nrow(OliveOilScaled), 3))

# We see that scaling the data slightly improved our accuracy. This does an 
# alright job of categorizing the oils but does not do amazingly well as many of 
# the oils are fairly similar to each other. With that being said, if you were 
# asked to cluster these oils into nine groups and you did it randomly, you 
# would probably have close to a 0.111 (1/9) accuracy rate which is much worse 
# than what we got in both the scaled and unscaled cases.
