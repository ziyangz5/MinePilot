from builtins import range
from status_model.RL_learning_DQN_Avoid import play_episode,DQN,CFG,ReplayMemory
import MalmoPython
import os
import sys
import time
import torch
import torch.optim as optim
from map_generator import get_maze_xml

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools

    print = functools.partial(print, flush=True)

def GetMissionXML(i):
    return get_maze_xml()


# class MineAgent:
#     def __init__(self,agent_host):
#         self.agent_host = agent_host
#         self.current_speed = 0.3
#         self.current_turning_speed = 0
#     def set_action(self,act:str):
#         if act == "move":
#             self.current_speed = 0.5
#         elif act == "brake":
#             self.current_speed = 0
#             self.current_turning_speed = 0
#         elif act == "right":
#             self.current_turning_speed = 0.4
#         elif act == "left":
#             self.current_turning_speed = -0.4
#         elif act == "end_turn":
#             self.current_turning_speed = 0
#         elif act == "nothing":
#             return
#         self.agent_host.sendCommand(f"move {self.current_speed}")
#         self.agent_host.sendCommand(f"strafe {self.current_turning_speed}")
#     def get_act_list(self):
#         return ["move","brake","right","left","end_turn"]

class MineAgent:
    def __init__(self,agent_host):
        self.agent_host = agent_host
        self.current_speed = 0.245
        self.current_turning_speed = 0.25
    def set_action(self,act:str):
        if act == "move":
            self.current_speed = 0.245
        elif act == "brake":
            #self.current_speed = 0
            self.current_turning_speed = 0
            self.current_speed = 0.245
        elif act == "right":
            self.current_turning_speed = 0.255
            self.current_speed = 0.245
        elif act == "left":
            self.current_turning_speed = -0.255
            self.current_speed = 0.245
        elif act == "end_turn":
            self.current_turning_speed = 0
        elif act == "nothing":
            return
        self.agent_host.sendCommand(f"move {self.current_speed}")
        self.agent_host.sendCommand(f"strafe {self.current_turning_speed}")
    def get_act_list(self):
        return ["brake","right","left"]


def main():
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()


    cfg = CFG()
    device = torch.device("cuda:0" if cfg.cuda else "cpu")
    policy_net = DQN(256,256,3).to(device)
    policy_net.load_state_dict(torch.load("saved_params.wts"))
    target_net = DQN(256, 256, 3).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    memory = ReplayMemory(15000)
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    if agent_host.receivedArgument("test"):
        num_repeats = 1
    else:
        num_repeats = 420
    i = 0
    xml_st = GetMissionXML(i)
    while True:
        my_mission = MalmoPython.MissionSpec(xml_st, True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        i += 1
        # Attempt to start a mission:
        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        # Loop until mission starts:
        print("Waiting for the mission to start ", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.05)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

        agent_obj = MineAgent(agent_host)
        print()
        print("Mission running ", end=' ')

        play_episode(policy_net,target_net,agent_obj,memory,world_state,optimizer,cfg,device,i)


        print()
        print()
        print("Mission ended")

if __name__ == "__main__":
    main()