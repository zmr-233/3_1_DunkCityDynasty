import sys,os;sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np
from collections import defaultdict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter 
#from baselines.common.model import Model
from baselines.common.wrappers import RLWrapper #关键导入
from DunkCityDynasty.env.gym_env import GymEnv

from pprint import pprint
from hpn_policy import *

#====================================================================================
class zmrPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #----------------------------------------------
        # 主要调参
        self.hpn_hidden_dim = 128
        self.rnn_hidden_dim = 128
        self.n_heads_input = 3
        self.n_heads_output = 3

        self.hidden_state_SamAct = None
        self.hidden_state_Eva = None
        #----------------------------------------------
        self.memory = Memory()
        self.model = HPNPolicy(self.hpn_hidden_dim, self.rnn_hidden_dim,
                               self.n_heads_input, self.n_heads_output).to('cpu')
        self.update_step = 0
    def sample_action(self, states):
        #🔴此处是有问题------------------------------------------------------
        new_states = []
        for state in states:
            new_states.append(state[np.newaxis, :])
        new_states = [torch.tensor(state) for state in new_states]
        #+++++++++++++++++++++++++++++++++++++
        if_print_states_shape = True
        #-------------------------
        #BUG:batch_size = states.size(0)
        #更改为++++++++++++
        batch_size = len(states)
        #🔴👆这里我感觉迟早要出问题
        #BUG-----------------------------------
        #if if_print_states_shape is True:
        #    print(f"#1-@zmrP@sample_action@states形状: {states.size()}")
        #更改为++++++++++++++++++++++++++++++
        if if_print_states_shape is False:
            # 打印每个状态的形状，因为 states 是列表，所以逐一打印
            for idx, state in enumerate(states):
                print(f"#1-@zmrP@sample_action@states[{idx}]形状: {state.shape}")
        #🔴打印出形状
        #BUG:-------------------原因：GRU可以处理None
        #BUG:self.hidden_state_SamAct = self.hidden_state_SamAct if self.hidden_state_SamAct is not None else torch.zeros(batch_size, self.rnn_hidden_dim)
        value, probs, hh = self.model(new_states,self.hidden_state_SamAct)
        self.hidden_state_SamAct = hh
        #🔴六个智能体共用一个self.hidden_state?????
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action.detach().numpy().item(), log_probs
    def get_random_actions(self, states):
        action_masks = {key: np.argwhere(states[key][-1] == 1.).flatten() for key in states}
        return {key: random.choices(action_masks[key].tolist(), k=1)[0] for key in action_masks}
    def get_actions(self, states_dic):
        actions = {}
#================================
        self.log_probs = {}
        #❓这里可以修改代码进行优化
#===========🟠👆这里已经是新增了属性self.log_probs
        #key_ll = []
        for key in states_dic.keys():
            states = states_dic[key]
            actions[key], self.log_probs[key] = self.sample_action(states)
            #🔴👆这里不能因为上一个states里没有动作，就不做动作啊
            #key_ll.append(key)
        #print(key_ll)
        return actions
    def _compute_returns(self, rewards,dones,gamma=0.99):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(np.array(returns)).unsqueeze(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    #def save_model(self, path):
        #Path(path).mkdir(parents=True, exist_ok=True)
        #torch.save(self.model.state_dict(), f"{path}/{step}")
        #torch.save(self.model.state_dict(), f"{path}")
    def evaluate(self, states, actions):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        #BUG:-------------------原因：GRU可以处理None
        #self.hidden_state_Eva = self.hidden_state_Eva if self.hidden_state_Eva is not None else torch.zeros(batch_size, self.rnn_hidden_dim)
        value, probs, hh = self.model(states,self.hidden_state_Eva)
        self.hidden_state_Eva = hh
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions.squeeze(dim=1))
        entropys = dist.entropy()
        return value, log_probs, entropys
    def sgd_iter(self, states, actions, returns, old_log_probs):
        '''使用 sgd_iter 方法对数据进行随机梯度下降迭代.
        
        这实质上是把整个数据集分成小批次，并对每批数据进行更新。
        '''
        batch_size = actions.shape[0]
        mini_batch_size = 32
        for _ in range(batch_size//mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield [states[i][rand_ids,:] for i in range(8)], actions[rand_ids,:], returns[rand_ids,:], old_log_probs[rand_ids,:]
    def update(self, *,stats_recorder=None,all_train_step):
        states, next_states, actions, rewards, dones, old_log_probs = self.memory.sample()
        old_log_probs = torch.cat(old_log_probs,dim=0).unsqueeze(dim=1)
        #🟠迷惑性代码👆:用来给内层加括号的-->作用:
        #old_log_probs = [tensor([1, 2, 3]), tensor([4, 5]), tensor([6, 7, 8, 9])]
        #torch.cat(old_log_probs,dim=0) = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        #.unsqueeze(dim=1) = ------>效果等价于🟢.view(-1,1)
        #tensor([[1],
        #[2],
        #[3],
        #[4],
        #[5],
        #[6],
        #[7],
        #[8],
        #[9]])
        #BUG🟢改进方法：
        if states is None: 
            return
        states = [torch.tensor(np.array(state),dtype=torch.float32) for state in states]
        next_states = [torch.tensor(np.array(next_state),dtype=torch.float32) for next_state in next_states]
        actions = torch.tensor(actions,dtype=torch.float32).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards),dtype=torch.float32)
        dones = torch.tensor(np.array(dones),dtype=torch.float32)
        returns = self._compute_returns(rewards, dones)
        for _ in range(2):
            for states, actions, returns, old_log_probs in self.sgd_iter(states, actions, returns, old_log_probs):
                values, log_probs, entropys = self.evaluate(states, actions)
                advantages = returns - values.detach()
                ratio = torch.exp(log_probs.unsqueeze(dim=1) - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropys.mean()
                critic_loss = nn.MSELoss()(returns,values)
                tot_loss = actor_loss + 0.5 * critic_loss
                self.model.opt.zero_grad()
                tot_loss.backward()
                self.model.opt.step()
                if stats_recorder is not None:
                    self.update_step += 1
                    all_train_step+=1
                    #stats_recorder.add_policy_loss(tot_loss.item(), self.update_step)
                    stats_recorder.add_policy_loss(tot_loss.item(),all_train_step)
                    #stats_recorder.add_value_loss(critic_loss.item(),  self.update_step)
                    stats_recorder.add_value_loss(critic_loss.item(),all_train_step)
                    #if self.update_step % 100 == 0:
                        #self.save_model('./output/bc_model', self.update_step)
                    #👆保存模型移动到train循环，传入all_train_step进行控制

class Exp:
    '''简单的数据容器,用于存储一次环境交互的结果.


    '''
    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)

class Memory():
    def __init__(self) -> None:
        self.buffer = deque(maxlen=10000)
    def push(self,exps):
        self.buffer.append(exps)
        #🟠exps应该是一个类👆在提取相应状态时，要用；
            #.state .action .reward .truncated .log_probs
    def handle(self, exps):
        '''提取并处理信息.
        
        从环境来的经验列表，提取并处理为能给神经网络的信息
        '''
        states = [[] for _ in range(8)] 
        #🔵👆注意state/next_state是包含8个状态的列表state_action_mask
        for exp in exps:
            for i in range(8):
                states[i].append(exp.state[i])
        for i in range(8):
            states[i] = np.array(states[i])
            #其实就是转np数组，但是np数组要求其中每一个列表都必须相同，因此只能部分转
        
        next_states = [[] for _ in range(8)]
        for exp in exps:
            for i in range(8):
                next_states[i].append(exp.next_state[i])
        #🔴❓是否缺少 next_states[i] = np.array(next_states[i]) ?
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        dones = np.array([exp.truncated for exp in exps])
        old_log_probs =  [exp.log_probs for exp in exps]
        #--------------🟠👆根本差异的来源
        return states, next_states, actions, rewards, dones, old_log_probs
    def sample(self):
        if len(self.buffer) == 0: 
            return None
        exps = self.buffer.popleft()
        return self.handle(exps)

def create_env(id=1):
    env_config = {
        'id': id,
        'env_setting': 'win',
        'client_path': 'D:\\1_GitProject\\game_package_release',
        'rl_server_ip': '127.0.0.1',
        'rl_server_port': 6666,
        'game_server_ip': '47.111.110.225',
        'game_server_port': 18001,
        'machine_server_ip': '',
        'machine_server_port': 0,
        "user_name": "qmxxpmz02wf6e",
        'render': False,
    }
    wrapper = RLWrapper({})
    env = GymEnv(env_config, wrapper=wrapper)
    return env

class ZmrRecorder:
    def __init__(self,*,version) -> None:
        self.writers = {}
        self.writer_types = ['Play','Policy']
        for writter_type in self.writer_types: 
            self.writers[writter_type] = SummaryWriter(f'./tmp/hpn_v{version}_{writter_type}',flush_secs=5)
                                                                #👆这里版本注意修改
    
    def add_scalar(self,w_type,*,tag,scalar_value,global_step): 
        self.writers[w_type].add_scalar(tag=tag, scalar_value=scalar_value, global_step = global_step)
    
    def add_rewards(self, rewards,all_ep_step): #🟢注意:all_ep_step应该为cnt*n+step
        for key in rewards.keys():
            self.add_scalar('Play',tag=f'rewards/reward_{key}',scalar_value=rewards[key],global_step=all_ep_step)
    def add_policy_loss(self, loss,all_ep_step):
        self.add_scalar('Policy',tag=f'loss/policy_loss',scalar_value=loss,global_step=all_ep_step)
    def add_value_loss(self, loss, all_ep_step):
        self.add_scalar('Policy',tag=f'loss/value_loss',scalar_value=loss,global_step=all_ep_step)
    def close(self):
        for writer in self.writters.values():
            writer.close()

#&&&&&是否正式运行
is_play = False
#&&&&&
def v2_train(*,env, policy,stats_recorder=None,model_version):
    model_path = f'./tmp/model/hpn_v{model_version}_model'
    #👆合并保存位置^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if os.path.exists(model_path):
        model_parameters = torch.load(model_path)
        policy.model.load_state_dict(model_parameters)
    else:
        print(f"Error:权重文件不存在{model_path}")
    if is_play: 
        policy.model.eval()
    ep_cnt = 0
    all_ep_step= 0 
    all_train_step = 0
    if_tmp = 0
    key_l = {0:0,1:0,2:0,3:0,4:0,5:0}
    for i in range(1000000):
        exps = []
        states, infos = env.reset()
        ep_rewards = defaultdict(int)
        ep_step = 0
        while True:
            actions = policy.get_actions(states)
            #if ep_cnt < 50:
            #    actions = policy.get_random_actions(states)
            next_states , rewards, dones, truncated, infos = env.step(actions)
            ep_step += 1
            all_ep_step +=1
            share_keys = list(set(states.keys()) & set(next_states.keys()) & set(actions.keys()) & set(rewards.keys()) & set(truncated.keys()))
            #🟠👆这个share_keys和下面的for in循环是做什么的？
            #返回的是具有完整信息的['0', '1', '2']
            #确保我们处理的每个智能体在当前时间步都有完整的信息
            #换句话说，我们不希望处理那些在某些字典中缺失数据的智能体
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            if_use_states_key = False

            if if_use_states_key:
                for key in states.keys():
                    try:
                        if states[key][0][14]== float(2):
                            print('Ball fallen========= ({0},{1},{2})'.format(states[key][0][5],states[key][0][6],states[key][0][7]))
                    except:
                        continue
                    key_l[key]+=1
                if all_ep_step%50==0:
                    #print(np.array(key_l.values)/sum(key_l.values))
                    print(np.array(list(key_l.values())) / sum(key_l.values()))
                if if_tmp == 0:
                    pprint(states[key])
                    print("--------------------------")
            if_tmp = 1
            #pprint(rewards)
            #pprint(infos)
            #pprint(rewards)
            #pprint(actions)
            #print("-----------------------------------------")
            #=====================================================================================================
            for key in rewards.keys():
                ep_rewards[key] += rewards[key]
            if truncated['__all__'] or ep_step >= 120:
                ep_step = 0
                ep_cnt += 1
                #stats_recorder.add_rewards(ep_rewards, ep_cnt)
            stats_recorder.add_rewards(rewards,all_ep_step)

            for key in share_keys:
                state = states[key]
                next_state = next_states[key]
                action = actions[key]
                reward = rewards[key]
                done = truncated[key]
                truncat = truncated[key]
    #========================================================================
                log_probs = policy.log_probs[key]
                #(Policy.get_actions)self.log_probs = {}
                #每次选择动作后，都会重置
    #==================================🟠👆注意这个log_probs是从policy里出来的
                exp = Exp(state=state,next_state=next_state,action=action,reward=reward,done=done,log_probs=log_probs,truncated=truncat)
                exps.append(exp)
            if len(exps) >= 512:
                policy.memory.push(exps)
                #policy.update(stats_recorder = stats_recorder,all_train_step=all_train_step)
                exps = []
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                print(f'{all_train_step}SAVE_MODEL+++++++++++++++++++++++++++')
                #policy.save_model(f'./tmp/model/hpn_v{model_version}_model')
                #torch.save(policy.model.state_dict(), f'./tmp/model/hpn_v{model_version}_model')
                #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # 确保路径存在
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                # 保存模型的状态字典
                torch.save(policy.model.state_dict(), model_path)

            if dones['__all__']:
                break
            states = next_states


def v2_main(*,version,model_version):
    env = create_env()
    policy = zmrPolicy()
    stats_recorder = ZmrRecorder(version=version)
    v2_train(env=env,policy=policy,stats_recorder=stats_recorder,model_version=model_version)
    stats_recorder.close()

if __name__ == '__main__':
    v2_main(version=1,model_version=1)