import torch
from torch import nn
from PIL import Image
import  matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch.backends.cudnn as cudnn
from collections import namedtuple
import torch.nn.functional as F
import torch
import time
import random
import torchvision.transforms as T
import json
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class CFG:
    lr = 0.01
    # weight_decay = 0.001
    cuda = True
    gamma = 0.85
    batch_size = 8
    gae_lambda = 0.96
    value_loss_coef = 0.98
    entropy_coef = 0.025
    max_grad_norm = 20

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
def img_preprocessing(img,dep,speed,ts,device,size=(128,128)):
    img = img.resize(size)
    dep = dep.resize(size)
    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ]
    )
    tensor_img = transform1(img).reshape((1,size[0],size[1]))
    dep_img = transform1(dep).reshape((1,size[0],size[1]))
    result_tensor = torch.cat((tensor_img.to(device), dep_img.to(device)), 0)
    result_tensor = torch.cat((result_tensor, speed), 0)
    result_tensor = torch.cat((result_tensor, ts), 0)
    return result_tensor

def frame_process(frame_list:bytearray,size=(128,128)):
    int_list = list(frame_list)
    img_o = np.array(int_list).reshape((1024,1024,4))
    img = img_o[:,:,2]
    depth = img_o[:,:,-1].reshape((1024,-1))
    image = Image.fromarray(img.astype('uint8'), 'RGB').resize(size)
    depth = Image.fromarray(depth.astype('uint8'), 'L').resize(size)
    return image,depth


class ReplayMemory(object):

    def __init__(self, capacity: int) -> None:
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self,
             state: np.ndarray,
             action: int,
             reward: int,
             next_state: np.ndarray,
             done: bool) -> None:
        """Creates `Transition` and insert
        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state,
                                              action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) :
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the length """
        return len(self.memory)

class A3C_net(nn.Module):
    def __init__(self):
        super(A3C_net, self).__init__()
        self.lstm = nn.LSTMCell(2312, 256)
        self.conv_1 = nn.Sequential(
                                    nn.Conv2d(4, 8, 5, 1, 3),
                                    nn.SELU(),
                                    nn.MaxPool2d((2, 2)))

        self.conv_2 = nn.Sequential(nn.Conv2d(8, 8, 5, 1, 3),
                                    nn.SELU(),
                                    nn.MaxPool2d((2, 2)))
        self.conv_3 = nn.Sequential(nn.Conv2d(8, 8, 5, 1, 3),
                                    nn.SELU(),
                                    nn.MaxPool2d((2, 2)))
        self.out_layer = nn.Sequential(nn.Linear(2312, 2048),
                                       nn.SELU(),
                                       nn.Linear(2048, 256),
                                       nn.Dropout(0.15),
                                       nn.SELU(),
                                       nn.Linear(256, 64),
                                       nn.Dropout(0.15),
                                       nn.SELU()
                                       )  # naive version: no bomb.
        self.critic_layer = nn.Sequential(nn.Linear(64, 1))
        self.actor_layer = nn.Sequential(nn.Linear(64, 5))

    def forward(self, image,hx,cx):
        x = self.conv_1(image)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(x.size()[0], -1)
        hx, cx = self.lstm(x, (hx, cx))
        f = self.out_layer(x)

        return self.critic_layer(f),self.actor_layer(f),hx, cx


def optimize_model(memory,device,cfg,policy_net,target_net,optimizer):
    if len(memory) < cfg.BATCH_SIZE:
        return
    transitions = memory.sample(cfg.BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8).bool()
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch.to(device)).gather(1, action_batch)

    next_state_values = torch.zeros(cfg.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * cfg.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
import math
steps_done = 0
def select_action(policy_net,state,cfg,device,n_actions):
    global steps_done
    sample = random.random()
    eps_threshold = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * \
        math.exp(-1. * steps_done / cfg.EPS_DECAY)
    print("eps =",eps_threshold)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def get_img(world_state,agent_obj,device):
    img,dep = frame_process(world_state.video_frames[0].pixels)
    speed = torch.zeros(1, 128, 128).to(device) + agent_obj.current_speed
    ts = torch.zeros(1, 128, 128).to(device) + agent_obj.current_turning_speed
    input_img = img_preprocessing(img,dep, speed,ts, device).to(device)
    return input_img

def play_episode(model,target_net,agent_obj,memory,world_state,optimizer,cfg,device,i_episode):
    print(i_episode)
    skip_sec_frame = 2
    reward_report = 0
    values = []
    log_probs = []
    rewards = []
    entropies = []
    ttr = 0
    time.sleep(1.5)
    cx = torch.zeros(1, 256).to(device)
    hx = torch.zeros(1, 256).to(device)
    while world_state.is_mission_running:
        current_r = 0
        reward_report += 1


        try:
            input_img_temp = get_img(world_state, agent_obj, device)
        except:
            import traceback
            traceback.print_exc()
            print("Skipped")
            world_state = agent_obj.agent_host.getWorldState()
            continue
        state = input_img_temp
        value, logit, hx, cx = model(state.unsqueeze(0),hx, cx)
        err = False
        for error in world_state.errors:
            print("Error:", error.text)
            err =True
        if err:
            break
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        entropies.append(entropy)
        action = prob.multinomial(num_samples=1).detach()
        log_prob = log_prob.gather(1, action)
        values.append(value)
        agent_obj.set_action(agent_obj.get_act_list()[int(action.item())])
        time.sleep(0.1)

        world_state = agent_obj.agent_host.getWorldState()
        for reward in world_state.rewards:
            if reward.getValue() > 0:
                print("D!")
            current_r += reward.getValue() + 0.0
        if agent_obj.current_speed <0.1:
            current_r -= 4
        if agent_obj.current_turning_speed != 0:
            current_r -= 10
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floorAll', 0)
            end = 0
            for i, block in enumerate(grid):
                if block == 'redstone_block':
                    end = i
                    break
            x = end//21;y = end - (end//21)*21
            distance = math.sqrt((x-10)**2 + (y-10)**2)
            current_r += (5 - distance)*1.5
        current_r = max(-10,current_r)

        ttr += current_r

        log_probs.append(log_prob)
        rewards.append(current_r)


        if reward_report % 5 == 0:
            print(current_r)



        #agent_obj.set_action(agent_obj.get_act_list()[random.randint(0, 4)])


        time.sleep(0.1)
    R = torch.zeros(1, 1).to(device)
    values.append(R)
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1).to(device)
    for i in reversed(range(len(rewards))):
        R = cfg.gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        # Generalized Advantage Estimation
        delta_t = rewards[i] + cfg.gamma * \
                  values[i + 1] - values[i]
        gae = gae * cfg.gamma * cfg.gae_lambda + delta_t

        policy_loss = policy_loss - \
                      log_probs[i] * gae.detach() - cfg.entropy_coef * entropies[i]

    optimizer.zero_grad()

    (policy_loss + cfg.value_loss_coef * value_loss).backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

    optimizer.step()

    print("R=",ttr)