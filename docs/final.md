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

# Project Summary
In this project, we try to solve a sub-problem of the self-driving problem, which is automatic obstacle avoidance. We treat the main character “Steve” as a car. The agent can do the following action:
1. Accelerating forward
2. Reducing forward speed (braking)
3. Moving to the left horizontally
4. Moving to the right horizontally
5. Stop moving horizontally

We want our agent to drive as far as possible without hitting obstacles set on a road in a given period of time (30 seconds). The boundary of the road is surrounded by redstone wall, so the agent must make sure it drive on the road and avoid driving onto the shoulder. The size of the road is 9 by 150, and there are 20 pillars as obstacles on the road. You can see the details of the map from the figure below:
<div style="text-align:center"><img src="figures_f/f1.png" /></div>

We want to develop our agent in a way similar to modern self-driving solutions. Therefore, we are using Deep Q-learning with computer vision (image segmentation) to solve this problem.


## Approach
We are using two main algorithm for this project:
1. Image Segmentation
2. Deep Q-learning 

We want the SNN (Segmentation Neural Network) to learn a very efficient representation of the input image (\$256 \times 256\$ pixels), and pass the output of SNN into the DQN (Deep Q Network). The reason why we are doing this is because we believe by using the high efficient representation from the SNN, DQN doesn't have to learn much image information representation in its CNN layers, which will improve the performance. (And this approach actually does improve the performance of the DQN, see this). 