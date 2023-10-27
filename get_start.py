import time
import random
import numpy as np
from DunkCityDynasty.env.gym_env import GymEnv



import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class RandomAgent():
    def __init__(self):
        pass

    def get_actions(self, states):
        return {key: np.random.randint(0, 8) for key in states}


def main():
    # env config
    # --- win env
    config = {
    'id': 1,
    'env_setting': 'win',  # 或者 'linux' 如果您用的是Ubuntu系统
    'client_path': 'D:\\1_GitProject\\game_package_release',
    'rl_server_ip': '127.0.0.1',
    'rl_server_port': 6666,
    'game_server_ip': '47.111.110.225',
    'game_server_port': 18001,
    'machine_server_ip': '',  # 如果您用的是多机设置，请填写机器服务器的IP地址
    'machine_server_port': 0,  # 如果您用的是多机设置，请填写机器服务器的端口号
    'user_name': 'qmxzk7jjeias2',  # 或者您可以选择其他用户名
    'episode_horizon': 100000  # max game steps
    }

    # --- linux env
    # config = {
    #     'id': 1,
    #     'env_setting': 'linux',
    #     'client_path': 'path-to-game-client',
    #     'rl_server_ip': '127.0.0.1',
    #     'rl_server_port': 42636,
    #     'game_server_ip': '127.0.0.1',
    #     'game_server_port': 18000,
    #     'machine_server_ip': '',
    #     'machine_server_port': 0,
    #     'user_name': 'xxxx',
    # }

    # # --- multi_machine
    # config = {
    #     'id': 1,
    #     'env_setting': 'multi_machine',
    #     'client_path': '',
    #     'rl_server_ip': '10.219.204.81',
    #     'rl_server_port': 42636,
    #     'game_server_ip': '127.0.0.1',
    #     'game_server_port': 18000,
    #     'machine_server_ip': '10.219.204.76',
    #     'machine_server_port': 6667,
    #     'user_name': 'xxxx',
    # }

    env = GymEnv(config)
    agent = RandomAgent()
    user_name = "qmxzk7jjeias2"
    states, infos = env.reset(user_name = user_name, render = True)
    while True:
        actions = agent.get_actions(states)
        states, rewards, dones, truncated, infos = env.step(actions)
        print(actions)
        if dones['__all__']:
            break

if __name__ == '__main__':
    main()