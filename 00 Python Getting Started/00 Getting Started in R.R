
################################################################################
# File Name: 00 Getting Started in R.R                                 
# Description: 
# Author: SAIG, Dylan Steberg
# Date: 08/04/25
# Course: Machine Learning Short Course
# R version: 4.3.0
# Package(s): none
# Dataset(s): none
#
# Revision History:
#     08/04/25: Created this file to match the "Getting Started in Python.ipynb" 
#           file used in the Python version of this course.
################################################################################

# Getting Started with R and R scripts

# In some applications, it's more common to work with R Markdown documents, 
# we're going to work with a R script instead. R Markdown documents have a 
# more professional appearance, separating text and code cells. We have elected 
# to use R script as each line can be run manually and the results of each line
# can be seen immediately. In R Marksdown, entire code cells must be run which
# can hide some details 

################################################################################
############################# Part 1: Running Code ############################# 
################################################################################
# The first line below this (without a #) is our first line of code. To run the 
# code below, you can highlight the line and click the Run button in the top
# right of this document. Or you can place your cursor before the line then hit
# Ctrl+Enter on PC or Cmd+Enter on Mac. After running with the output should 
# show up in your console. Try it with the code cell below!
2 + 2

# The code that is run and the output are shown in the console (typically the
# bottom left panel in RStudio. Variables that are assigned a value are saved 
# into your R environment, or workspace (typically the top right panel in 
# RStudio). The following line of code saves a variable which can be seen in
# our environemnt and called again later. 
text = "Welcome to the Machine Learning Course!"
text

# The bottom left panel (typically), shows your file explorer, plots,
# currently installed packages, and help files. Running the following line of 
# code should make a plot appear in that panel.
plot(rnorm(50, 1, 2), runif(50), xlab = "X", ylab = "Y", main = "Pretty Plot")

# The zoom button will pop the current plot into a separate window where small
# text may be easier to read. 

# Finally clicking and dragging at the intersection of any two panes can make 
# those sections smaller or larger.

################################################################################
################### Part 2: Installing and Loading Packages #################### 
################################################################################

# Most of the commonly used R packages must be installed before the functions
# contained in them can be used. One package used throughout this course is the
# "ISLR" package which provides datasets used in many of the exercises. To 
# access these datasets we need to install the package. This can be done using
# the `install.packages` function as shown below. 
install.packages("ISLR")

# Once a package is installed, we cannot use that package until we load it into
# our work space. This is done using the `library` function as shown below.
library(ISLR)

# Since we will need a few other packages , we will install them all at once 
# using the "requirements.R" file. Sourcing this file using the following line
# of code lets us run functions from that file in the current R session. 
source("requirements.R")

# Those with keen eyes see that a function called `load_ML_course_packages` was
# loaded into our R environment. Running that function as shown below will
# install and load the rest of the packages required for the class. Note this
# may take a few minutes on some machines.
load_ML_course_packages()

################################################################################
################### Part 3: Load Data from the "ISLR" Package ##################
################################################################################

# To load data from a package and store it in the variable named "df", we use 
# the following structure: `data("dataset_name")`. Let's try this with the "OJ"
# dataset from the "ISLR" package. We'll use this data later in the course. Note
# that sometimes you may have to specify the package the data is coming from,
# but that shouldn't be an issue in this class.
data("OJ")

# We can look at the first few rows of the dataset using the `head` function.
head(OJ)

# An entire dataset can be inspected using the `View` function as shown below.
# This open the dataset in a new tab where all the rows and columns can be 
# Viewed by scrolling down or to the right.
View(OJ)

# In practice, your data will likely not come from R packages. Instead, you'll 
# want to read in data from an separate file, such as a .csv or .xlsx file. We 
# will not cover reading in data files in this course. See *** for more
# information on reading in data files check out this data camp tutorial.
# https://www.datacamp.com/tutorial/r-data-import-tutorial
