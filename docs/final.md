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
We want to develop our agent in a way similar to modern self-driving solutions. Therefore, we are using Deep Q-learning Network (DQN) with computer vision (image segmentation) to solve this problem. We also set the forward speed of our agent as a continuos variable to simulate the reality.


## Approach

In this project, we use the following models to be baselines or to solve this problem:
1. Random model (baseline 1)
2. DQN without Segmentation Neural Network (SNN) (baseline 2)
3. DQN with SNN. No continuos speed (model in status report. baseline 3)
4. DQN with SNN and continuos speed. (final model)

#### Random model
As discussed above, our agent has 5 different actions described below:
|        |      0     |             1            |             2            |         3         |         4         |
|:------:|:----------:|:------------------------:|:------------------------:|:-----------------:|:-----------------:|
| Action | Do nothing/End moving horizontally | Horizontal velocity set to -0.25 | Horizontally velocity set to 0.25 | Forward speed +0.1 | Forward speed -0.1 |
{: .tablelines}

<br>
The Random model is a very simple baseline, it will only take action 1-5  randomly following a uniform distribution.

#### DQN without SNN

Our second baseline is a very simple deep Q-learning network.