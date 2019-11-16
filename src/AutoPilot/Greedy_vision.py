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
    result_tensor = torch.cat((tensor_img, dep_img), 1)
    # result_tensor = torch.cat((result_tensor, dirct), 1)
    # result_tensor = torch.cat((result_tensor, ts), 1)
    return result_tensor

def frame_process(frame_list:bytearray,size=(128,128)):
    int_list = list(frame_list)
    img_o = np.array(int_list).reshape((1024,1024,4))
    img = img_o[:,:,:3]
    depth = img_o[:,:,-1].reshape((1024,-1))
    image = Image.fromarray(img.astype('uint8'), 'RGB').resize(size)
    depth = Image.fromarray(depth.astype('uint8'), 'L').resize(size)
    return image,depth



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
    # speed = torch.zeros(1, 1, 128, 128).to(device) + agent_obj.current_speed
    # ts = torch.zeros(1, 1, 128, 128).to(device) + agent_obj.current_turning_speed
    input_img = img_preprocessing(img,dep, None,None, device)
    return input_img

def play_episode(agent_obj,world_state,i_episode):
    print(i_episode)
    skip_sec_frame = 2
    reward_report = 0
    init = True
    action = 0
    ttr = 0
    time.sleep(1.5)
    if i_episode + 1 % 15 == 0:
        print("Testing best policy")
    last_dis = 99999
    while world_state.is_mission_running:
        current_r = 0
        reward_report += 1


        try:
            input_img_temp = get_img(world_state, agent_obj,None, None)*255
        except:
            import traceback
            traceback.print_exc()
            print("Skipped")
            world_state = agent_obj.agent_host.getWorldState()
            continue
        if torch.any(input_img_temp[0,2,32:96,32:96]<20) and torch.any(input_img_temp[0,0,60:68,60:68]>90) :
            agent_obj.set_action("move")
        elif torch.any(input_img_temp[0,3,50:70,50:70] < 30):
            index = random.randint(0,1)
            al = ["right","left"]
            agent_obj.set_action(al[index])
        else:
            index = random.randint(0, len(agent_obj.get_act_list())-1)
            agent_obj.set_action(agent_obj.get_act_list()[index])
        world_state = agent_obj.agent_host.getWorldState()
        time.sleep(0.05)