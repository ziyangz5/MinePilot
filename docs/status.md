---
layout: default
mathjax: true
title: Proposal
---

<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
    table {
  width: 100%;
}

th {
  height: 50px;
}
td {
  height: 50px;
}
    
</style>

# Summary
In this project, we try to solve a sub-problem of the self-driving problem, which is automatic obstacle avoidance. We s treat the main character “Steve” as a car. For now, it goes forward at a constant speed, and it can either horizontally move to right, or horizontal move to left, or just go straight. We want it to avoid all obstacles set on a road, and reach the destination. The destination is the red stone walls shown in the figure below. The size of the road is 9 by 30, and there are 9 pillars as obstacles on the road. On the edge of the road, there are two iron walls to detect if the agent drives off the road. You can see the details of the map from the figure below:
<div style="text-align:center"><img src="figures/fig_2.png" /></div>

# Approach
We are using two main algorithm for this project:
1. Image Segmentation
2. Deep Q-learning 

We want the SNN (Segmentation Neural Network) to learn a very efficient representation of the input image ($256 \times 256$ pixels), and pass the output of SNN into the DQN (Deep Q Network). The reason why we are doing this is because we believe by using the high efficient representation from the SNN, DQN doesn't have to learn much image information representation in its CNN layers, which will improve the performance. (And this approach actually does improve the performance of the DQN, see this). 

## Segmentation Neural Network
The SNN is used as a sensor of our agent. The reason why the representation from SNN is more efficient is that there are only 5 possible values and 1 channel in its output. The 5 values are:

|                | &nbsp;0         | &nbsp;1                                 | &nbsp;2           | &nbsp;3                 | &nbsp;4          |
| -------------- | --------------- | --------------------------------------- | ----------------- | ----------------------- | ---------------- |
| **Represents** | &nbsp;sky&nbsp; | &nbsp;pillars&nbsp;and&nbsp;walls&nbsp; | &nbsp;grass&nbsp; | &nbsp;destination&nbsp; | &nbsp;road&nbsp; |
{: .tablelines}

<br>


This representation is much more efficient than the original images (3 channels with 256 values).
### Data Generation
Since the SNN is trained by using supervised learning. The most important problem is how to get the dataset with enough data. We developed an approach to generate data by ourselves.<br>
First, we generate $n$ random maps ($n=500$ in our case). Then, we replace the resource package of Minecraft to a pure color texture package made by ourselves(shown below) 
<div style="text-align:center"><img src="figures/fig_3.png" width="400" height="400"/></div>

<br>
Then, we just scan the image pixel by pixel, and we set different threshold of RGB values for types of blocks. Then, we final dataset looks like as shown below:
<div style="text-align:center"><img src="figures/fig_4.png"/></div>
<center>Input Data</center>
<div style="text-align:center"><img src="figures/fig_5.png"/></div>
<center>Ground Truth Labels</center> <br>

### Network Structure and Loss Function
We are using one of the most popular network structure, [ResNet50](https://arxiv.org/abs/1512.03385), to do the segmentation. This network can achieve a very high accuracy ([See Evaluation](#Evaluation)). We train the SNN by minimizing the pixel-wise cross entropy between the ground truth and our prediction. The pixel-wise cross entropy function is given below:<br>

<br>

$$
L(\pmb{y},\pmb{\hat{y}})=\sum_{i=0}^c p_ilog(q_i)
$$

We use Adam as the optimizer, and trained only 10 epochs with batch size 4 under RTX 2080 graphics cards. After 10 epochs, the validation accuracy reached around 85%.


## Deep Q Network

### Actions
Once got the segmentation from the SNN, our DQN needs to decide the action of agent. For now, our agent only does discrete actions. The actions can be:

|        | 0                        | 1                      | 2                       |
| ------ | ------------------------ | ---------------------- | ----------------------- |
| Action | Stop moving horizontally | Move left horizontally | Move right horizontally |
{: .tablelines}

### Reward Function
We originally defined that the agent receives 200 points after reaching the destination, and receives -75 points after hitting the obstacles or going off the road. However, we found this reward function is very sparse. The agent may not be able to reach the destination for a very long time. Therefore, to encourage our agent to live longer and do less meaningless action, our reward function is defined as:

$$
R(s)=\left\{
\begin{aligned}
    200 &\ (\text{Agent reaches destination})\\
    -75 &\ (\text{Collision happens})\\
    7.5 &\ (\text{No action})\\
    2.5 &\ (\text{Moving to the left or right})
\end{aligned}
\right.
$$

# Evaluation
## Quantitative Evaluation:
In our project, we have two main algorithm: CNN and DQ-Learning. For the segmentation task, we want to minimize the following loss function given a $n\times m$ image:<br>



where $\pmb{y}$ is labels in a 2D array, and $\pmb{\hat{y}}$ is our prediction. $\textit{H}$ is the cross entropy. Then, to evaluate the CNN model, we want to get a small loss value on the validation set, and also want to get high accuracy rate.<br>
For DQ Learning, we set several different rewards. For example, if the agent hits the wall, it will get a negative reward $r_h$. If the agent reaches the destination, it will get a positive reward $r_d$. If the agent reaches the destination very fast, we may give it a big postive reward $r_{d+}$. We want the total reward $r_t$ to be as high as possible.
## Qualitative analysis
If our algorithms work well, we can expect our agent can avoid most of obstacles and reach the destination successfully. For our segmentation model, it should be able to tell which part is road, which part is sky, which part is walls on images. Then, it can give nice information (labels) to the DQ learning model. The DQ learning model should learn how to control the agent's speed and yaw rotation to maintain itself on the road, and avoid hitting the walls. Our best agent should avoid all the obstacles, and reach the destination in a short time (which means it almost never brakes).


# Remaining Goals and Challenges
The remaining challenges we have is that we need to implement the addtional feature of acceleration and braking of our agent to simulate the actual car feature in the reality. Our agent can currently avoid only one kind of obstacle (Iron block). So, our next challenge can be to train the agent to avoid differrnt kind of obstacles (even agents, if possible).

# Resources Used
We have used Python Malmo module to simulate the car driving environment and to operate the Minecraft agent. We use PyTorch library to train our agent to avoid obstacles.