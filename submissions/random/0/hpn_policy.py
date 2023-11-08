#import sys,os
#import gymnasium as gym
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#附属工具====================================================================================================
#a.激活函数选择工具 ----------------------------------------------------------------------------------------------------
def get_activation_func(name, hidden_dim):
    """
    'relu'
    'tanh'
    'leaky_relu'
    'elu'
    'prelu'
    :param name:
    :return:
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif name == "elu":
        return nn.ELU(alpha=1., inplace=True)
    elif name == 'prelu':
        return nn.PReLU(num_parameters=hidden_dim, init=0.25)
#b.动作嵌入层-----------------------------------------------------------------------------------------
class ActionEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        num_actions =  52
        self.embedding = nn.Embedding(num_actions, embedding_dim)

    def forward(self, action_indices):
        return self.embedding(action_indices)
#c.多头合并层-----------------------------------------------------------------------------------------
class Merger_1(nn.Module):
    def __init__(self, head, fea_dim):
        '''合并[batch_size, n_heads, fea_dim]->[batch_size, fea_dim]
        
        '''
        super().__init__()
        self.head = head
        self.fea_dim = fea_dim
        if head > 1:
            self.weight = nn.Parameter(torch.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''

        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        '''
        if self.head > 1:
            batch_size = x.size(0)
            weight = self.weight.expand(batch_size, -1, -1)
            weighted_x = self.softmax(weight) * x
            return torch.sum(weighted_x, dim=1)
        else:
            return x.squeeze(1)

class Merger_2(nn.Module):
    def __init__(self, head, fea_dim):
        '''合并 [batch_size, 1, n_heads, fea_dim]->[batch_size, fea_dim]
        
        '''
        super().__init__()
        self.head = head
        self.fea_dim = fea_dim
        if head > 1:
            self.weight = nn.Parameter(torch.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        #BUG:
        if self.head > 1:
            batch_size = x.size(0)
            weight = self.weight.expand(batch_size, -1, -1)
            #BUG:🔵小修改，x的view的维度
            #return torch.sum(self.softmax(weight) * x.view(batch_size, self.head, -1), dim=1)
            return torch.sum(self.softmax(weight) * x.view(batch_size, self.head, self.fea_dim), dim=1)
        else:
            return torch.squeeze(x, dim=1)
#c.线性层&嵌入层----------------------------------------------------------------------------------------
def embedding_layer(input_size, num_embeddings , embedding_dim, **kwargs):
    class EmbeddingLayer(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kwargs):
            super(EmbeddingLayer, self).__init__()
            self.layer = nn.Embedding(num_embeddings=num_embeddings+1, embedding_dim=embedding_dim, **kwargs)
                                                    #👆注意潜在越界问题num_embeddings+1
        def forward(self, x: torch.Tensor):
            return self.layer(x)
    layer = EmbeddingLayer(num_embeddings, embedding_dim, **kwargs)
    output_size = [None, embedding_dim]
    return layer, output_size
def linear_layer(input_size, layer_dim):
    input_dim = input_size[1]
    output_size = [None, layer_dim]
    layer = nn.Sequential(nn.Linear(input_dim, layer_dim), nn.ReLU())
    return layer, output_size
#d.Agent嵌入层
class AgentEmbedding(nn.Module):
    '''基础查表操作
    '''
    def __init__(self):
        super().__init__()
        self.agent_state_len = 73
        self.my_character_type_embed_layer, self.my_character_type_embed_layer_out_dim = embedding_layer([None], 100, 16)
        self.my_role_type_embed_layer, self.my_role_type_embed_layer_out_dim = embedding_layer([None], 8 ,8)
        self.my_buff_type_embed_layer, self.my_buff_type_embed_layer_out_dim = embedding_layer([None], 50, 6)
        #self.agent_state_dim = 16+8+6-3 + self.agent_state_len
        #---------------------------
        #self.agent_state_dim = -3 + self.agent_state_len #🟠-3可能出现问题
        #🔵++++++++++++++++++++++++++++正确计算嵌入的总维度
        self.agent_state_dim = self.my_character_type_embed_layer_out_dim[1] \
                               + self.my_role_type_embed_layer_out_dim[1] \
                               + self.my_buff_type_embed_layer_out_dim[1] \
                               + self.agent_state_len\
                               - 3 #这个-3从何而来?
        #🟠-3原因是my_character_type/my_role_type/my_buff_type是属于self.agent_state_len的其中三个
        #当把这三个转为特征向量后，就要从原来的self.agent_state_len中减去3了，这就是-3的原因
        #因此，如下的forward传入x的长度就是73，而不处理的部分则是 my_states = x[:,3:73].float()

    def forward(self,x):
        my_character_type = x[:, 0].long()
        my_role_type = x[:, 1].long()
        my_buff_type = x[:, 2].long()
        
        my_character_type = self.my_character_type_embed_layer(my_character_type)
        my_role_type = self.my_role_type_embed_layer(my_role_type)
        my_buff_type = self.my_buff_type_embed_layer(my_buff_type)
        
        my_states = x[:,3:].float()

        return my_character_type,my_role_type,my_buff_type,my_states

#====================================================================================================
#1.定义超网络-----------------------------------------------------------------------------------------
class Hypernet(nn.Module):
    def __init__(self,input_dim,
                     hidden_dim,
                    main_input_dim,main_output_dim,
                    n_heads,
                    *,activation_func_name,
                    use_bias = False):
        '''权重生成网络

        :param input_dim: 特征信息(HPN输入)
        :param main_input_dim: 主要输入(HPN输出)
        :param main_output_dim: 主要输出(HPN输出)
        :param  n_heads: 多头
        
        :return [batch_size,main_input_dim, main_output_dim * self.n_heads]: 

        使用方法
        forward
        '''
        super().__init__()
        self.main_input_dim = main_input_dim
        self.main_output_dim = main_output_dim
        self.output_dim = main_input_dim * main_output_dim
        self.activation_func_name = activation_func_name
        self.n_heads = n_heads
        self.use_bias = use_bias
        
        #超网络层
        self.multihead_weight_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation_func(self.activation_func_name, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim * self.n_heads),
        )
        if self.use_bias == True:
            #偏置层
            self.multihead_bias_nn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                get_activation_func(self.activation_func_name, hidden_dim),
                nn.Linear(hidden_dim, self.main_output_dim * self.n_heads),
            )

    def forward(self, x):#[batch_size,main_input_dim * main_output_dim * self.n_heads]
        #->[batch_size,main_input_dim, main_output_dim * self.n_heads]
        weights = self.multihead_weight_nn(x).view([-1, self.main_input_dim, self.main_output_dim * self.n_heads])
        
        if self.use_bias == True:
            #->[batch_size, main_output_dim * n_heads]
            biases = self.multihead_bias_nn(x).view([-1, self.main_output_dim * self.n_heads])
            
            return weights, biases
        else:
            return weights
        
#2.Global状态处理层-----------------------------------------------------------------------------------------------
class GlobalLayer(nn.Module):
    def __init__(self,output_dim):
        super().__init__()
        self.global_state_len = 30
        self.linear_layer, self.linear_layer_out_dim = linear_layer([None, self.global_state_len], output_dim) #64
    def forward(self, x):
        x = x.float()
        x = self.linear_layer(x)
        return x    
#3.Self状态处理层-------------------------------------------------------------------------------------------------
class SelfLayer(nn.Module):
    def __init__(self,output_dim,hidden_dim=None,embed_net=None):
        super().__init__()
        self.input_dim=16+8+6-3+73
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_net = embed_net if embed_net is not None else AgentEmbedding()
        self.own = nn.Linear(self.input_dim, self.output_dim, bias=True)
    
    def init_hidden(self):
        '''BUG:🟠目前尚未使用隐藏层
        
        '''
        return self.own.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def forward(self,x):
        #(16             ,8           ,6           ,70      )
        my_character_type,my_role_type,my_buff_type,my_states = self.embed_net(x)
        input_tensor = torch.cat([my_character_type, my_role_type, my_buff_type,my_states], dim=1).float()
        output = self.own(input)
        return input_tensor,output #BUG:这里需要把智能体状态给输出了
        

#4.Agent状态处理层------------------------------------------------------------------------------------------------
class AgentLayer(nn.Module):
    '''Agent状态处理
    :param embed_net :为所有Agent使用同一个嵌入层转换为特征向量
    :param hpn_net :为所有Agent使用同一个HPN层来生成矩阵
        
    '''
    def __init__(self, output_dim, n_heads,
                 embed_net=None, hpn_net=None):
        super().__init__()
        #RNN
        self.output_dim = output_dim
        self.n_heads = n_heads

        #使用Embedding层
        self.embed_net = embed_net if embed_net is not None else AgentEmbedding()
        #使用HPN层
        self.hpn_net = hpn_net if hpn_net is not None else Hypernet(
            input_dim=16+8+6-3+73,
            hidden_dim=256,
            main_input_dim=16+8+6-3+73,
            main_output_dim=output_dim,  # 确保 main_output_dim 与 output_dim 一致
            n_heads=n_heads,
            activation_func_name='relu'
        )

    def forward(self, x):
        #(16             ,8           ,6           ,70      )
        my_character_type,my_role_type,my_buff_type,my_states = self.embed_net(x)
        
        #(1)根据特征生成矩阵权重
        #100
        hyper_input = torch.cat([my_character_type, my_role_type, my_buff_type,my_states], dim=1).float()
        
        #[6]->[batch_size,main_input_dim, main_output_dim * self.n_heads]
        input_w_agent, input_b_agent = self.hpn_net(hyper_input)
        
        #(2)进行矩阵运算
        #[batch_size,main_input_dim] * [batch_size,main_input_dim, main_output_dim * n_heads] ->[batch_size, n_heads, output_dim * n_heads]
        
        #output_agent = torch.matmul(matmul_input.unsqueeze(1), input_w_agent).squeeze(1)
        #output_agent += input_b_agent.unsqueeze(1)
        #output_agent = output_agent.view(-1, self.n_heads, self.output_dim)
        
        output_agent = torch.matmul(hyper_input.unsqueeze(1), input_w_agent)
        output_agent += input_b_agent
        output_agent = output_agent.view(
            -1, self.n_heads, self.output_dim #BUG:需要确保 main_output_dim = output_dim，否则这个重塑会出错
        )#[batch_size,n_heads,output_dim]

        return output_agent
#4.----------------------------------------------------------------------------------------------

#=================================================================================================    
class HPNPolicy(nn.Module):
    def __init__(self,
                 hpn_hidden_dim,rnn_hidden_dim,
                 n_heads_input,n_heads_output):
        super().__init__()
        #形状信息
        self.hpn_hidden_dim = hpn_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim if rnn_hidden_dim is not None else 128 #🟠BUG:玄学调参
        #BUG:细分---------------------self.n_heads = n_heads #切记不可太高，否则....
        self.n_heads_input = n_heads_input #这是合并输入
        self.n_heads_output = n_heads_output #这是技能动作输出的多头
        self.use_bias = True #是output专有动作时候使用
        
        #(1)states_embedding层--所有人共用一个 转为特征向量
        self.agent_embedding_net = AgentEmbedding()
        #actions_embedding层--用于合并动作
        self.action_embedding_net = ActionEmbedding(embedding_dim=128)

        #(2)HPN层--一共有两套网络 敌人和队友
        self.hyper_input_w_ally = Hypernet( #100的输入层，256的隐藏层，如果output_dim=128，则最终的超网络要输出的权重高达...
            input_dim=16+8+6-3+73,     #100*128*5=64000的输出层
            #🟠再提醒一次，16+8+6-3+73=100，为什么是-3已经解释过了
            hidden_dim=self.hpn_hidden_dim,
            main_input_dim=16+8+6-3+73,
            main_output_dim= self.hpn_hidden_dim,  # 确保 hpn_hidden_dim 与 output_dim 一致
            n_heads= self.n_heads_input,
            activation_func_name='relu',
            use_bias = True
        )
        self.hyper_input_w_anemy = Hypernet( 
            input_dim=16+8+6-3+73,     
            hidden_dim=self.hpn_hidden_dim,
            main_input_dim=16+8+6-3+73,
            main_output_dim= self.hpn_hidden_dim,
            n_heads= self.n_heads_input,
            activation_func_name='relu',
            use_bias = True
        )
        
        #(3)合并层--用于合并HPN产生的多头
        #[bs, n_head, fea_dim]->[bs, fea_dim]
        self.unify_input_heads = Merger_1(self.n_heads_input, self.rnn_hidden_dim)
        #[batch_size, 1, n_heads, fea_dim]->[bs, fea_dim]
        self.unify_output_heads = Merger_2(self.n_heads_output, 52-12)
        #👆注意Merger_1和Merger_2是处理不同维度的合并，如上注释所示
        
        #(4)RNN层
        #使用 nn.GRUCell处理单个时间步长的输入
        #
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        
        #A-公有动作层
        self.output_normal_actions = nn.Linear(self.rnn_hidden_dim, 12) 
        
        #B-专有技能层
        self.hyper_output_w_action = Hypernet( 
            input_dim=16+8+6-3+73,    
            hidden_dim=self.hpn_hidden_dim,
            main_input_dim= self.rnn_hidden_dim,
            main_output_dim= 52-12, #动作空间为52，其中公有动作12个，专有动作40个
            n_heads= self.n_heads_output,
            activation_func_name='relu',
            use_bias = False #🟠再次强调，这里的use_bias是False，意味着偏置是用hyper_output_b_action专门计算的
        )
        self.hyper_output_b_action = Hypernet( #专门用来计算偏置
            input_dim=16+8+6-3+73,     
            hidden_dim=self.hpn_hidden_dim,
            main_input_dim= self.rnn_hidden_dim,
            main_output_dim= 52-12,
            n_heads= self.n_heads_output,
            activation_func_name='relu',
            use_bias = False #🟠这里也是False
        ) 
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #(1)Global层
        self.global_state_layer = GlobalLayer(rnn_hidden_dim) #直接线性层
        
        #(2)Self层 分为critic和actor网络
        self.self_state_layer = SelfLayer(rnn_hidden_dim)
        self.self_state_layer_2 = AgentLayer(rnn_hidden_dim,self.n_heads_input,
                                            self.agent_embedding_net,self.hyper_input_w_ally)

        #(3)ally层 -- 其中没有新增任何神经网络-相当于集成
        self.ally0_state_layer = AgentLayer(rnn_hidden_dim,self.n_heads_input,
                                            self.agent_embedding_net,self.hyper_input_w_ally)
        self.ally1_state_layer = AgentLayer(rnn_hidden_dim,self.n_heads_input,
                                            self.agent_embedding_net,self.hyper_input_w_ally)

        #(4)enemy层
        self.enemy0_state_layer = AgentLayer(rnn_hidden_dim,self.n_heads_input,
                                            self.agent_embedding_net, self.hyper_input_w_anemy)
        self.enemy1_state_layer = AgentLayer(rnn_hidden_dim,self.n_heads_input,
                                            self.agent_embedding_net, self.hyper_input_w_anemy)
        self.enemy2_state_layer = AgentLayer(rnn_hidden_dim,self.n_heads_input,
                                            self.agent_embedding_net, self.hyper_input_w_anemy)

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #定义critic层
        self.critic_input_dim  = 3 * self.rnn_hidden_dim
        self.critic_hidden_dim = 5 * self.rnn_hidden_dim
        self.critic_layer = nn.Sequential(
            nn.Linear(self.critic_input_dim, self.critic_hidden_dim ), 
            nn.ReLU(), 
            nn.Linear(self.critic_hidden_dim , 1), 
            nn.ReLU()
        )
        
        # 定义优化器
        self.opt = optim.Adam(self.parameters(), lr=1e-3) #🔴如何初始化所有参数?
    
    def forward(self,states,hidden_state):
        global_feature = states[0].float()
        self_feature = states[1]
        ally0_feature = states[2]
        ally1_feature = states[3]
        enemy0_feature = states[4]
        enemy1_feature = states[5]
        enemy2_feature = states[6]
        if len(states) > 7: #action_mask动作掩码
            action_mask = states[7].float()
        
        #(1)Global层
        global_embedding = self.global_state_layer(global_feature)
        
        #(2)Self层 
        self_feats, self_embedding = self.self_state_layer(self_feature)
        #👆这里返回的self_feats是用于给self.hyper_output_w_action的输入

        #(3)ally层 
        ally0_feature = self.ally0_state_layer(ally0_feature)
        ally1_feature = self.ally1_state_layer(ally1_feature)
        ally_embedding = self.unify_input_heads( 
            ally0_feature + ally1_feature
        )

        #(4)enemy层
        enemy0_feature = self.enemy0_state_layer(enemy0_feature)
        enemy1_feature = self.enemy1_state_layer(enemy1_feature)
        enemy2_feature = self.enemy2_state_layer(enemy2_feature)
        enemy_embedding = self.unify_input_heads(
            enemy0_feature + enemy1_feature + enemy2_feature
        )

        #a.合并层
        #BUG:此处使用加法层==========================================================================================
        embedding = global_embedding + self_embedding + ally_embedding + enemy_embedding
        
        self_embedding_2 = self.self_state_layer_2(self_feature)
        ally_embedding_2 = ally_embedding + self.unify_input_heads(self_embedding_2)
        embedding_critic = torch.cat([global_embedding,ally_embedding_2,enemy_embedding],dim = 1).float()

        #==============================================Actor网络====================================================
        #b.激活和RNN
        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim) #首次运行使用零状态: h_in = torch.zeros(batch_size, rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # [bs, rnn_hidden_dim]

        #c.计算公有动作的价值
        q_normal = self.output_normal_actions(hh).view(-1, self.n_agents, 12)  # [bs, n_agents, 12]

        #d.计算专有动作价值+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #Part1-生成权重
        # agent_feats为[batch_size, agent_feature_dim]
        # 初始输出-> [batch_size, rnn_hidden_dim * 40 * n_heads]
        output_w_special = self.hyper_output_w_action(self_feats) #🟠只返回一个对象
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output_w_special = output_w_special.view(-1, 40 * self.n_heads_output, self.rnn_hidden_dim).transpose(1, 2)

        #Part2-生成偏置
        output_b_special = self.hyper_output_b_action(self_feats).view( -1 ,40 * self.n_heads_output)  
        
        # Part3-计算Q值------通过矩阵乘法计算每个专有动作的Q值
        # [batch_size, 1, rnn_hidden_dim] * [batch_size, rnn_hidden_dim, 40 * n_heads] = [batch_size, 1, 40 * n_heads]
        #BUG:错误的q_values计算---------------------------------------------------------------------------
        q_values = torch.matmul(hh.unsqueeze(1), output_w_special) #BUG:hh.unsqueeze(1)??????

        if self.use_bias:
            # 增加一个维度使偏置与q_values的形状匹配
            q_values += output_b_special.unsqueeze(1) #->[batch_size, 1, 40 * n_heads]
        
        #BUG:多余的平均值合并--------------------------------------------------
        #-->[batch_size, 1, 40 * n_heads]->[batch_size, 1, 40, n_heads]
        #q_values_unified = q_values.view(-1, 1, 40, self.n_heads_output).mean(dim=-1)
        
        #使用复杂的权重矩阵合并+++++++++++++++++++++++++++++++++++++++++++++++++
        #->[batch_size, 1, 40, n_heads]-->[batch_size, 40]
        q_values = self.unify_output_heads(q_values)
        # 这里需要确保 unify_output_heads 方法输出正确的形状

        
        q = torch.cat([q_normal,q_values],dim=-1)
        #转为概率
        #logits_p = F.softmax(q, dim=-1)

        #==============================================Critic网络====================================================
        value = self.critic_layer(embedding_critic)
        
        if len(states) > 7:  # 如果有动作掩码
            large_negative = torch.finfo(q.dtype).min if q.dtype == torch.float32 else -1e9
            # 应用掩码，未掩码的保持原值，掩码的设置为非常小的值
            mask_q = q * action_mask + (1 - action_mask) * large_negative
            # 对调整后的logits应用softmax，转换为概率
            probs = nn.functional.softmax(mask_q, dim=-1)
            return value.float(), probs.float(), hh
        else:
            #BUG:如果没有掩码，直接对原始Q值应用softmax转换为概率
            probs = nn.functional.softmax(q, dim=-1)
            return value.float(), probs.float(), hh
        

#=================================================================================================    
#
#
#
#=================================================================================================
#1.BC离线训练环境
class TrainHPC(nn.Module):
    def __init__(self):
        pass
#==================================================================================================
