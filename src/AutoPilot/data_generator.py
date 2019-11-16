from __future__ import print_function
from __future__ import division
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #7: The Maze Decorator

from builtins import range
from past.utils import old_div
import MalmoPython
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
from map_generator import Cfg

from map_generator import load_maze_xml
if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

def GetMissionXML(i):
    return load_maze_xml(i)

GROUND_TRUTH = False

# Create default Malmo objects:


def frame_process(frame_list:bytearray,index=None):
    int_list = list(frame_list)
    img = np.array(int_list).reshape((256,256,4))
    img = img[:,:,:3]
    if GROUND_TRUTH:
        image = Image.fromarray(img.astype('uint8'), 'RGB')
        image.save(f"dataset/ground_truth/gt_{index}.png")
        return
    result_data_set = []
    for i in range(256):
        result_data_set.append([])
        for j in range(256):
            r,g,b = img[i][j][0],img[i][j][1],img[i][j][2]
            label = 0
            if r>110 and g>150 and b>225:
                label = 0#sky
            elif r>150 and g < 10 and b<50:
                label = 1#wall
            elif r<30 and g<30 and b<30:
                label = 2#grass
            elif g>130 and r<20 and b<30:
                label = 3#destination
            elif g<10 and r>170 and b>180:
                label = 4#road

            result_data_set[i].append(label)
    return result_data_set
num_repeats = 450
result_dataset = []
for i in range(0,num_repeats):
    time.sleep(0.5)
    print(i)
    print(len(result_dataset))
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    # if agent_host.receivedArgument("test"):
    #     num_repeats = 1
    # else:
    #     num_repeats = 420
    my_mission = MalmoPython.MissionSpec(GetMissionXML(i), True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission running ", end=' ')

    # Loop until mission ends:
    init_count = 18
    once = False
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if init_count<=1:
            if not once:
                result_dataset.append(frame_process(world_state.video_frames[0].pixels,i))
                once = True

        else:
            init_count -= 1

        for error in world_state.errors:
            print("Error:",error.text)

    cfg = Cfg()
    for x in range(cfg.maze_bound[0],cfg.maze_bound[2]):
        for z in range(cfg.maze_bound[1],cfg.maze_bound[3]):
            for y in range(0,5):
                my_mission.drawBlock(x, y, z, "air")
    print()
    print()
    print("Mission ended")
    # Mission has ended.

if not GROUND_TRUTH:
    print(len(result_dataset))
    np.save("dataset/labels",np.array(result_dataset))
    # for i in range(len(result_dataset)):
    #     plt.imshow(result_dataset[i])
    #     plt.axis('off')
    #     plt.savefig(f'dataset/labels_img/label_v_{i}.png', bbox_inches='tight', pad_inches=0)