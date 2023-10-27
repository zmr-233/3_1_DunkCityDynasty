import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import os
class BaseWrapper():
    def __init__(self, config):
        self.config = config
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(226,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(52)
    def state_wrapper(self, state_infos):
        states, infos = {}, {}
        for key in state_infos.keys():
            infos[key] = state_infos[key][0]
            states_dict = state_infos[key][1]
            global_state = np.array(list(states_dict['global_state'].values()))
            self_state = np.array(list(states_dict['self_state'].values()))
            ally0_state = np.array(list(states_dict['ally_0_state'].values()))
            ally1_state = np.array(list(states_dict['ally_1_state'].values()))
            enemy0_state = np.array(list(states_dict['enemy_0_state'].values()))
            enemy1_state = np.array(list(states_dict['enemy_1_state'].values()))
            enemy2_state = np.array(list(states_dict['enemy_2_state'].values()))
            action_mask = np.array(state_infos[key][-1])
            observations = [global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action_mask]
            states[key] = observations
        return states,infos
    def reward_wrapper(self, states):
        return 0
    def info_wrapper(self, states):
        return {}
    def truncated_wrapper(self, states):
        return False


def onehot(num, size):
    """
    """
    onehot_vec = np.zeros(size)
    onehot_vec[num] = 1
    return onehot_vec.tolist()


class SimpleWrapper(BaseWrapper):
    ''' Simple Wrapper for Baseline
    '''
    def __init__(self, config):
        super().__init__(config)

    def state_wrapper(self, state_infos):
        states, infos = {}, {}
        for key in state_infos.keys():
            infos[key] = state_infos[key][0]
            states_dict = state_infos[key][1]
            global_state = self.handle_global_states(states_dict['global_state'])
            self_state = self.handle_agent_states(states_dict['self_state'])
            ally0_state = self.handle_agent_states(states_dict['ally_0_state'])
            ally1_state = self.handle_agent_states(states_dict['ally_1_state'])
            enemy0_state = self.handle_agent_states(states_dict['enemy_0_state'])
            enemy1_state = self.handle_agent_states(states_dict['enemy_1_state'])
            enemy2_state = self.handle_agent_states(states_dict['enemy_2_state'])

            action_mask = np.array(state_infos[key][-1])
            observations = [global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action_mask]
            states[key] = observations
        return states,infos
    def handle_global_states(self, global_states_dict):
        global_states_list = []
        global_states_list.append(global_states_dict['attack_remain_time'] * 0.05)
        global_states_list.append(global_states_dict['match_remain_time'] * 0.02)
        global_states_list.append(global_states_dict['is_home_team'])
        global_states_list.append(global_states_dict['ball_position_x']*0.2)
        global_states_list.append(global_states_dict['ball_position_y']*0.5)
        global_states_list.append(global_states_dict['ball_position_z']*0.2)
        global_states_list.append(global_states_dict['vec_ball_basket_x']*0.2)
        global_states_list.append(global_states_dict['vec_ball_basket_y']*0.5)
        global_states_list.append(global_states_dict['vec_ball_basket_z']*0.2)
        global_states_list.append(global_states_dict['team_own_ball'])
        global_states_list.append(global_states_dict['enemy_team_own_ball'])
        global_states_list.append(global_states_dict['ball_clear'])
        ball_status = onehot(int(global_states_dict['ball_status']), 6)
        global_states_list += ball_status
        global_states_list.append(global_states_dict['can_rebound'])
        global_states_list.append(global_states_dict['dis_to_rebound_x']*0.2)
        global_states_list.append(global_states_dict['dis_to_rebound_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_rebound_y']*0.2)
        global_states_list.append(global_states_dict['can_block'])
        global_states_list.append(global_states_dict['shoot_block_pos_x']*0.2)
        global_states_list.append(global_states_dict['shoot_block_pos_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_x']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_z']*0.2)
        global_states_list.append(global_states_dict['dis_to_block_pos_y']*0.2)
        global_states_list.append(global_states_dict['block_diff_angle']*0.3)
        global_states_list.append(global_states_dict['block_diff_r']*0.2)
        return np.array(global_states_list)
    def handle_agent_states(self, agent_states_dict):
        agent_states_list = []
        agent_states_list.append(agent_states_dict['character_id'])
        agent_states_list.append(agent_states_dict['position_type'])
        agent_states_list.append(agent_states_dict['buff_key'])
        agent_states_list.append(agent_states_dict['buff_value']*0.1)
        agent_states_list.append((agent_states_dict['stature']-180)*0.1)
        agent_states_list.append(agent_states_dict['rational_shoot_distance']-7)
        agent_states_list.append(agent_states_dict['position_x']*0.2)
        agent_states_list.append(agent_states_dict['position_y']*0.5)
        agent_states_list.append(agent_states_dict['position_z']*0.2)
        agent_states_list.append(agent_states_dict['v_delta_x']*0.3)
        agent_states_list.append(agent_states_dict['v_delta_z']*0.3)
        agent_states_list.append(agent_states_dict['player_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['player_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['basket_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['basket_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['ball_to_me_dis_x']*0.2)
        agent_states_list.append(agent_states_dict['ball_to_me_dis_z']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_me_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_me_r']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_basket_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_basket_r']*0.2)
        agent_states_list.append(agent_states_dict['polar_to_ball_angle']*0.3)
        agent_states_list.append(agent_states_dict['polar_to_ball_r']*0.2)
        agent_states_list.append(agent_states_dict['facing_x'])
        agent_states_list.append(agent_states_dict['facing_y'])
        agent_states_list.append(agent_states_dict['facing_z'])
        agent_states_list.append(agent_states_dict['block_remain_best_time'])
        agent_states_list.append(agent_states_dict['block_remain_time'])
        agent_states_list.append(agent_states_dict['is_out_three_line'])
        agent_states_list.append(agent_states_dict['is_ball_owner'])
        agent_states_list.append(agent_states_dict['own_ball_duration']*0.2)
        agent_states_list.append(agent_states_dict['cast_duration'])
        agent_states_list.append(agent_states_dict['power']* 0.001)
        agent_states_list.append(agent_states_dict['is_cannot_dribble'])
        agent_states_list.append(agent_states_dict['is_pass_receiver'])
        agent_states_list.append(agent_states_dict['is_marking_opponent'])
        agent_states_list.append(agent_states_dict['is_team_own_ball'])
        agent_states_list.append(agent_states_dict['inside_defence'])
        is_my_team = onehot(int(agent_states_dict['is_my_team']), 2)
        agent_states_list += is_my_team
        player_state = onehot(int(agent_states_dict['player_state']), 6)
        agent_states_list += player_state
        skill_state = onehot(int(agent_states_dict['skill_state']), 27)
        agent_states_list += skill_state
        return np.array(agent_states_list)