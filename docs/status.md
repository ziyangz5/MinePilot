---
layout: default
mathjax: true
title: Proposal
---
# Summary
In this project, we plan to develop a autopilot system under Minecraft setting. We treat the main character "Steve" as a car by letting him ride a horse (or drive a car provided by mods), and let him run on serval designed maps. Our goal is to make our agent be able to avoid all obstacles (or even other agents) on the road, and reach the destination successfully as fast as possible. In our plan, we will use pixel of the screen as the input, and the output is the action that our agent should take such as speeding up, turning, and braking.
# Approch
Our main algorithm first generate a map that have some random distribution of obstacles and roads. Then we will use convolutional neural network to do the segmentation of the images, and use deep Q-learning to learn how to take action.
## Map Generator
The map generator will randomly generate a map (a open space with roads and obstacles). The map has three different types of blocks: redstone_block, diamond_block, and iron_block. Iron block represents the obstacle that the car trying to avoid. Diamond block represents roads that the car can drive on, and redstone block represents the destination that the car should go to.
## Segmentation on Map
TBD
## Obstacle Avoidance
TBD
# Evaluation
## Quantitative Evaluation:
In our project, we have two main algorithm: CNN and DQ-Learning. For the segmentation task, we want to minimize the following loss function given a $n\times m$ image:<br>

$$
L(\pmb{y},\pmb{\hat{y}})=\sum_{i=0}^n\sum_{j=0}^m\textit{H}(y_{ij},\hat{y}_{ij})
$$

where $\pmb{y}$ is labels in a 2D array, and $\pmb{\hat{y}}$ is our prediction. $\textit{H}$ is the cross entropy. Then, to evaluate the CNN model, we want to get a small loss value on the validation set, and also want to get high accuracy rate.<br>
For DQ Learning, we set several different rewards. For example, if the agent hits the wall, it will get a negative reward $r_h$. If the agent reaches the destination, it will get a positive reward $r_d$. If the agent reaches the destination very fast, we may give it a big postive reward $r_{d+}$. We want the total reward $r_t$ to be as high as possible.
## Qualitative analysis
If our algorithms work well, we can expect our agent can avoid most of obstacles and reach the destination successfully. For our segmentation model, it should be able to tell which part is road, which part is sky, which part is walls on images. Then, it can give nice information (labels) to the DQ learning model. The DQ learning model should learn how to control the agent's speed and yaw rotation to maintain itself on the road, and avoid hitting the walls. Our best agent should avoid all the obstacles, and reach the destination in a short time (which means it almost never brakes).


# Remaining Goals and Challenges
The remaining challenges we have is that we need to implement the addtional feature of acceleration and braking of our agent to simulate the actual car feature in the reality. Our agent can currently avoid only one kind of obstacle (Iron block). So, our next challenge can be to train the agent to avoid differrnt kind of obstacles (even agents, if possible).

# Resources Used
We have used Python Malmo module to simulate the car driving environment and to operate the Minecraft agent. We use PyTorch library to train our agent to avoid obstacles.