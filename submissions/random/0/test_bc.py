import sys,os;sys.path.append(os.getcwd())
import torch
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter 
from baselines.common.utils import all_seed
from test_bc_collector import BCDataset, collate_fn
from test_bc_utils import download_human_data
#from baselines.common.model import Model


from hpn_linear import *

class Config:
    def __init__(self) -> None:
        self.seed = 0
        self.n_epoch = 10
        self.data_path = "D:\\1_GitProject\\3_1_DunkCityDynasty\\human_data"
        self.train_batchsize = 1
        self.test_batchsize = 4
        self.dataloader_n_workers = 4
        self.lr = 1e-4
        self.device = torch.device('cuda')

def train(cfg):
    #download_human_data(cfg.data_path)
    all_seed(cfg.seed)
    train_dataset = BCDataset(cfg.data_path,is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batchsize, shuffle=True, num_workers=cfg.dataloader_n_workers,collate_fn=collate_fn,)
    test_dataset = BCDataset(cfg.data_path,is_train=False)
    #----------------------------------------------
    # 主要调参
    hpn_hidden_dim = 128
    rnn_hidden_dim = 128
    n_heads_input = 2
    n_heads_output = 2

    hidden_state_SamAct = None
    hidden_state_Eva = None
    #----------------------------------------------
    model = HPNPolicy(hpn_hidden_dim, rnn_hidden_dim,
                            n_heads_input, n_heads_output,
                            net_mode ='bc').to(cfg.device)
    model_path = f'./hpn_linear_1_output/hpn_linear_1_bc_model'
    #👆合并保存位置^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if os.path.exists(model_path):
        model_parameters = torch.load(model_path)
        model.load_state_dict(model_parameters)
    else:
        print(f"Error:权重文件不存在{model_path}")
    #model_parameters = torch.load("./hpn_output/hpn_bc_model")
    #model.load_state_dict(model_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    tb_writer = SummaryWriter(f"./hpn_linear_1_output/logs/")
    step = 0
    for i_epoch in range(cfg.n_epoch):
        for i_batch, (global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state, action, weight) in enumerate(train_dataloader):
            #model_parameters = torch.load("./hpn_output/hpn_bc_model")
            #model.load_state_dict(model_parameters)
            # squeeze when batch_size = 1
            global_state = global_state.squeeze(0).to(cfg.device)
            self_state = self_state.squeeze(0).to(cfg.device)
            ally0_state = ally0_state.squeeze(0).to(cfg.device)
            ally1_state = ally1_state.squeeze(0).to(cfg.device)
            enemy0_state = enemy0_state.squeeze(0).to(cfg.device)
            enemy1_state = enemy1_state.squeeze(0).to(cfg.device)
            enemy2_state = enemy2_state.squeeze(0).to(cfg.device)
            action = action.squeeze(0).to(cfg.device)
            weight = weight.squeeze(0).to(cfg.device)
            # Forward
            _, logits = model([global_state, self_state, ally0_state, ally1_state, enemy0_state, enemy1_state, enemy2_state],None)#🔴🐞hidden_state缺失
            m = Categorical(logits=logits)
            loss = -m.log_prob(action) * weight
            loss = loss.mean()

            # Backward
            optimizer.zero_grad()
            loss.backward() if i_epoch == cfg.n_epoch - 1 else loss.backward(retain_graph=True)
            optimizer.step()
            if i_batch % 500 == 0:
                test_states, test_actions, _ = test_dataset.sample(cfg.test_batchsize)
                test_accuracy = []
                for test_state, test_action in zip(test_states, test_actions):
                    if len(test_action) == 0:
                        continue 
                    with torch.no_grad():
                        test_state = [inner_test_state.to(cfg.device) for inner_test_state in test_state]
                        test_action = test_action.to(cfg.device)
                        _, test_pred_logits = model(test_state,None)
                        test_pred_action = test_pred_logits.argmax(-1)
                        test_accuracy.append((test_pred_action == test_action).float().mean().detach().cpu().numpy())
                test_accuracy = sum(test_accuracy) / len(test_accuracy)
                step += 1
                print(f'Epoch: {i_epoch}, Batch: {i_batch}, Loss: {loss.item()}, test_acc: {test_accuracy}')
                tb_writer.add_scalar('loss', loss, step)
                tb_writer.add_scalar('acc', test_accuracy, step)
                torch.save(model.state_dict(), f"./hpn_linear_1_output/hpn_linear_1_bc_model")


if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    cfg = Config()
    train(cfg)