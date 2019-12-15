---
layout: default
title: Final Report
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


## Video

## Project Summary
In this project, we try to solve a sub-problem of the self-driving problem, which is automatic obstacle avoidance. We treat the main character “Steve” as a car. The agent can do the following action:
1. Accelerating forward speed
2. Reducing forward speed (braking)
3. Moving to the left horizontally
4. Moving to the right horizontally
5. Stop moving horizontally

We want our agent to drive as far as possible without hitting obstacles set on a road in a given period of time (30 seconds). The boundary of the road is surrounded by redstone wall, so the agent must make sure it drive on the road and avoid driving onto the shoulder. The size of the road is 9 by 150, and there are 22 pillars as obstacles on the road. You can see the details of the map from the figure below:
<div style="text-align:center"><img src="figures_f/f1.png" /></div>
<br>
We want to develop our agent in a way similar to modern self-driving solutions. Therefore, we are using Deep Q-learning Network (DQN) with computer vision (image segmentation) to solve this problem. We also set the forward speed of our agent as a continuos variable to simulate the reality. Using machine learning algorithm is essential to solve this problem since in real world, it is very hard to get simple grid representation of roads, and the action space is continuos in the real world. Auto-driving agents can only get complex vision information and limited depth information from cameras and radars.


## Approaches

In this project, we use the following models to be baselines or to solve this problem:
1. Random model (baseline 1)
2. DQN without Segmentation Neural Network (SNN) (baseline 2)
3. DQN with SNN. No continuos speed (model in status report. baseline 3)
4. DQN with SNN and continuos speed. (final model)

#### **Random model**
As discussed above, our agent has 5 different actions described below:

|        |      0     |             1            |             2            |         3         |         4         |
|:------:|:----------:|:------------------------:|:------------------------:|:-----------------:|:-----------------:|
| Action | Do nothing/End moving horizontally | Horizontal velocity set to -0.25 | Horizontally velocity set to 0.25 | Forward speed +0.1 | Forward speed -0.1 |
{: .tablelines}

<br>
The Random model is a very simple baseline, it will only take action 1-5  randomly following a uniform distribution.

#### **DQN without SNN**

Our second baseline is a deep Q-learning network. The reinforcement learning part of this model is identical to our final model. The only difference between this model and the final model is that this model does not use SNN as a vision-preprocessing network. Since this model does not use SNN, it will have more reaction time than models with SNN because SNN is a relative large network, and requires much time to run. However, without SNN, the DQN must learn the representation of the image and the policy at the same time, which can be very hard. Also, since the learning of the DQN is based on a black box reward, the representation in convolutional layers might be very imprecise. The action of the model is defined [here](#Random-model).<br><br>

**Network structure** <br><br>

**Reward function** <br><br>
We want to define a non-sparse reward function. Therefore, we decide to use the current speed of the agent as the main reward. We use the 10 times of the speed subtracted from 1.6 to be the main reward. The reward will be negative if the speed is too slow. Also, we want to encourage our agent to avoid hitting pillars and do less meaningless actions. Therefore, we design the reward function as given below:

$$
R(s)=\left\{
\begin{aligned}
    &(S\times 10)-1.6 \ (\text{No action})\\
    &(S\times 10)-4.2 \ (\text{Any action is taken})\\
    &-75\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (\text{Collision happens})
\end{aligned}
\right.
$$

where $S$ indicates the forward speed, $S \in [0.1,0.8]$.<br><br>

**Loss function**
<br><br>

The goal of Deep Q-learning is that instead of building a Q table, we want to find a Q function $Q$, and a policy $\pi$, so that $\pi(s)=\underset{s}{\mathrm{argmax}}(Q(s,a))$. $Q$ may be very complex, but according to universal approximation theorem, our network can fit the $Q$. Every epoch, we update the $Q$ by minimizing the loss function given below:

<br>

$$
    \delta = Q(s,a)-(r+\gamma \underset{a}{\mathrm{max}}(Q(s',a)))
$$

<br>

To train our model, we apply the Huber loss upon the $\delta$

$$
L(\delta)=\left\{
\begin{aligned}
    0.5\delta^2 &\ \ \ \text{if |$\delta$|<1}\\
    |\delta|-\frac{1}{2} &\ \ \ \text{Otherwise}\\ 
\end{aligned}
\right.
$$

 We are using Huber loss because it would make the loss not very sensitive to outliers and more stable. The performance of this model 

#### DQN with SNN (discrete speed)