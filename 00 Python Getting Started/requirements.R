
################################################################################
# File Name: requirements.R                                 
# Description: Requirments file for the ML shortcourse.This file streamlines 
#   installing and loading the packages required for the course.
# Author: SAIG, Dylan Steberg
# Date: 08/04/25
# Course: Machine Learning Short Course
# R version: 4.3.0
# Package(s): none
# Dataset(s): none
#
# Revision History:
#     08/04/25: Created this file to match the "requirements.py" file used in 
#           the Python version of this course.
################################################################################

# This function checks if each package in the "package_list" vector is installed
# in the current name space. If a package is not installed, it is installed and 
# loaded. If a package is already installed then it is loaded. 
load_ML_course_packages = function() {
  package_list = c("ISLR", "boot", "MASS", "class", "splines", "gam", "leaps",
                   "tree", "randomForest", "corrplot", "dendextend", "clue")
  for (pkg in package_list) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
    library(pkg, character.only = TRUE, quietly = TRUE)
  }
  print("Required packages successfully installed and loaded.")
}

# In the future for any additional packages that need to be installed and 
# loaded (because an exercise is changed, updated, etc.), the name of the 
# package can be added to the "package_list" vector.

# To utilize this file, it must be sourced in the current R session and the 
# `load_ML_course_packages` function must be run. This can be done by running
# the following two lines of code in the console or another R script.

# source("requirements.R") 
# load_ML_course_packages()

