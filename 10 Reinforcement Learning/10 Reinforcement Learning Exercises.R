
################################################################################
# File Name: 10 Reinforcement Learning Exercises.R                                 
# Description: Exercises for the reinforcment learning module of the ML 
#   shortcourse. 
# Author: SAIG, Dylan Steberg
# Date: 07/25/25
# Course: Machine Learning Short Course
# R version: 4.3.0
# Package(s): none
# Dataset(s): none
#
# Revision History:
#     07/25/25: Created R code to match Python code
################################################################################

# The following exercises, code, and explanations are adapted from An 
# Introduction to Statistical Learning with Applications in R (ISLR) 2nd Edition
# (James, Witten, Hastie, Tibshirani, and Taylor 2023).

################################################################################
##################### Exercise: Umbrella Carry Simulation ######################
################################################################################

# The following simulation code was created by ChatGPT to demonstrate the 
# "Umbrella Carry Scenario" discussed in the presentation.  

# It models a simple reinforcement learning agent deciding each day whether or 
# not to carry an umbrella, based on past experience with random weather 
# outcomes.

# First we will set up the simulation settings. We set the learning rate at 0.1. 
# We assume that there is a 50/50 change of rain each day. A random seed is set
# for reproducibility.
set.seed(314)
num_days = 30
learning_rate = 0.1
rain_prob = 0.5

# We set our initial Q-values at 0.5 for both taking and not taking an umbrella.
Q_take_umbrella = 0.5
Q_dont_take_umbrella = 0.5

# We also create vectors for holding our history which will be used for plotting.
Q1_history = c()
Q2_history = c()
rain_history = c()
action_history = c()
reward_history = c()

# Now we run the simulation cell and visualize how the agent learns over time.
for (day in 1:num_days) {
  # Simulate weather (rain or not)
  rain_today = rnorm(1) < rain_prob
  rain_history = c(rain_history, rain_today)
  
  # Choose action based on current Q-values (greedy policy)
  if (Q_take_umbrella > Q_dont_take_umbrella) {
    action = "Take Umbrella"
  }
  else {
      action = "Don't Take Umbrella"
  }
  action_history = c(action_history, action)
  
  # Assign reward based on action and rain
  if (action == "Take Umbrella") { # If you take the umbrella
    if (rain_today) { # And it rains today
      reward = 1.0 # Good call (large reward)
    }
    else { # And it doesn't rain today
      reward = 0.5 # Had to carry around an umbrella (small reward)
    }
    Q_take_umbrella = Q_take_umbrella + learning_rate * (reward - Q_take_umbrella)
    Q_dont_take_umbrella = 1 - Q_take_umbrella
  }
  else { # If you don't take the umbrella
    if (rain_today) {
      reward = 0.0 # You got rained on :( (no reward)
    }
    else {
      reward =  1.0 # Perfect call (large reward)
    }
    Q_dont_take_umbrella = Q_dont_take_umbrella + learning_rate * (reward - Q_dont_take_umbrella)
    Q_take_umbrella = 1 - Q_dont_take_umbrella
  }
  reward_history = c(reward_history, reward)
  Q1_history = c(Q1_history, Q_take_umbrella)
  Q2_history = c(Q2_history, Q_dont_take_umbrella)
}

leg_lab = c("Q(Take Umbrella)", "Q(Don't Take Umbrella)", "True Probability of Rain")
leg_col = c("#861F41", "#E5751F", "#75787b")
plot(1:num_days, Q1_history, ty = "l", col = leg_col[1], ylim = c(0, 1),
     main = "Learning Whether to Take an Umbrella (Q-values over 30 Days)",
     xlab = "Day", ylab = "Estimated Q-value", )
lines(Q2_history, ty = "l", col = leg_col[2])
abline(h = 0.5, lty = 2, col = leg_col[3])
legend("topleft", legend = leg_lab, lty = c(1, 1, 2), col = leg_col, cex = 0.65)
