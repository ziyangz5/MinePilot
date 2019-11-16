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
    # gamma = 0.5
    # batch_size = 8
    # gae_lambda = 0.96
    # value_loss_coef = 0.98
    # entropy_coef = 0.025
    # max_grad_norm = 20
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

def img_preprocessing(img,dep,dirct,ts,device,size=(128,128)):
    img = img.resize(size)
    dep = dep.resize(size)
    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ]
    )
    tensor_img = transform1(img).reshape((1,3,size[0],size[1]))
    dep_img = transform1(dep).reshape((1,1,size[0],size[1]))
    result_tensor = torch.cat((tensor_img.to(device), dep_img.to(device)), 1)
    result_tensor = torch.cat((result_tensor, dirct), 1)
    result_tensor = torch.cat((result_tensor, ts), 1)
    return result_tensor

def frame_process(frame_list:bytearray,size=(128,128)):
    int_list = list(frame_list)
    img_o = np.array(int_list).reshape((1024,1024,4))
    img = img_o[:,:,:3]
    depth = img_o[:,:,-1].reshape((1024,-1))
    image = Image.fromarray(img.astype('uint8'), 'RGB').resize(size)
    depth = Image.fromarray(depth.astype('uint8'), 'L').resize(size)
    return image,depth


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5,stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(3,3)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Sequential(nn.Linear(2592, 1024),
                                  nn.SELU(),
                                  nn.Linear(1024, 512),
                                  nn.SELU(),
                                  nn.Linear(512, outputs),
                                  )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.maxpool(F.selu(self.conv1(x)))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



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
def select_action(policy_net,state,cfg,device,n_actions,i_episode):
    global steps_done
    sample = random.random()


    eps_threshold = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * \
        math.exp(-1. * steps_done / cfg.EPS_DECAY)
    if i_episode + 1 % 15 == 0:
        eps_threshold = 0
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

def get_img(world_state,agent_obj,dirct,device):
    img,dep = frame_process(world_state.video_frames[0].pixels)
    speed = torch.zeros(1, 1, 128, 128).to(device) + agent_obj.current_speed
    ts = torch.zeros(1, 1, 128, 128).to(device) + agent_obj.current_turning_speed
    input_img = img_preprocessing(img,dep, speed,ts, device).to(device)
    # msg = world_state.observations[-1].text
    # observations = json.loads(msg)
    # grid = observations.get(u'floorAll', 0)
    # result = []
    # dict = {"grass":0,"diamond_block":1,"iron_block":2,"redstone_block":3}
    # r = -1
    # for i, block in enumerate(grid):
    #     if i % 21 == 0:
    #         result.append([])
    #         r += 1
    #     result[r].append(dict[block])
    # result[21//2][21//2] *= 3
    # result = torch.tensor(result).reshape(1,1,21,21).float().to(device)
    # result =  torch.cat((result, speed), 1)
    # result = torch.cat((result, ts), 1)
    return input_img

def play_episode(policy_net,target_net,agent_obj,memory,world_state,optimizer,cfg,device,i_episode):
    print(i_episode)
    skip_sec_frame = 2
    reward_report = 0
    init = True
    action = 0
    dirct = torch.zeros(1,1,128,128).to(device)
    ttr = 0
    time.sleep(1.5)
    if i_episode + 1 % 15 == 0:
        print("Testing best policy")
    last_dis = 99999
    while world_state.is_mission_running:
        current_r = 0
        reward_report += 1


        try:
            input_img_temp = get_img(world_state, agent_obj,dirct, device)
        except:
            import traceback
            traceback.print_exc()
            print("Skipped")
            world_state = agent_obj.agent_host.getWorldState()
            continue
        state = input_img_temp
        action = select_action(policy_net,state,cfg,device,len(agent_obj.get_act_list()),i_episode)
        err = False
        for error in world_state.errors:
            print("Error:", error.text)
            err =True
        if err:
            break
        agent_obj.set_action(agent_obj.get_act_list()[action])
        time.sleep(0.05)

        world_state = agent_obj.agent_host.getWorldState()
        for reward in world_state.rewards:
            if reward.getValue() > 0:
                print("D!")
            current_r += reward.getValue() + 0.0
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floorAll', 0)
            end = 0
            for i, block in enumerate(grid):
                if block == 'redstone_block':
                    end = i
                    break
            x = end//21;y = end%21
            distance = math.sqrt((x-40)**2 + (y-4)**2)
            print("d=",distance)
            if last_dis > distance:
                current_r += min((50 - distance)*0.3,1)*10
                dirct = dirct*0
            elif last_dis < distance:
                current_r -= 5
                dirct = dirct * 0 -1
            last_dis = distance
        #current_r = max(-50,current_r)

        #current_r *= 0.3

        ttr += current_r




        try:
            current_screen = get_img(world_state, agent_obj,dirct, device)
        except:
            continue

        current_r = torch.tensor([current_r], device=device).float()
        if world_state.is_mission_running:
            next_state = current_screen
        else:
            next_state = None

        if reward_report % 1 == 0:
            print(current_r.item())

        # if i_episode + 1 % 15 == 0:
        #     continue
        memory.push(state.cpu(), action, next_state, current_r)

        optimize_model(memory,device,cfg,policy_net,target_net,optimizer)


        #agent_obj.set_action(agent_obj.get_act_list()[random.randint(0, 4)])


        time.sleep(0.05)

    if i_episode % cfg.TARGET_UPDATE == 0:
        print("Target Updated.")
        target_net.load_state_dict(policy_net.state_dict())
    print("R=",ttr)