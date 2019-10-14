---
layout: default
mathjax: true
title: Proposal
---
# Summary
In this project, we plan to develop a autopilot system under Minecraft setting. We treat the main character "Steve" as a car by letting him ride a horse (or drive a car provided by mods), and let him run on serval designed maps. Our goal is to make our agent be able to avoid all obstacles (or even other agents) on the road, and reach the destination successfully. In our plan, we will use pixel of the screen as the input, and the output is the action that our agent should take such as speeding up, turning, and breaking.
# Algorithms
Our main algorithm will use convolutional neural network to do the segmentation of the images, and use deep Q-learning to learn how to take action.
# Evaluation Plan
## Quantitative Evaluation:
In our project, we have two main algorithm: CNN and DQ-Learning. For the segmentation task, we want to minimize the following loss function given a $n\times m$ image:<br>

$$
L(\pmb{y},\pmb{\hat{y}})=\sum_{i=0}^n\sum_{j=0}^m\textit{H}(y_{ij},\hat{y}_{ij})
$$

where $\pmb{y}$ is labels in a 2D array, and $\pmb{\hat{y}}$ is our prediction. $\textit{H}$ is the cross entropy. Then, to evaluate the CNN model, we want to get a small loss value on the validation set, and also want to get high accuracy rate.<br>
For DQ Learning, we set several different rewards. For example, if the agent hits the wall, it will get a negative reward $r_h$. If the agent reaches the destination, it will get a positive reward $r_d$. We want the total reward $r_t$ to be as high as possible.
## Qualitative analysis



# Appointment