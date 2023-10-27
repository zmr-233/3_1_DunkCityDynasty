import json
#with open('./1_test.json', 'r') as f:
#        state_infos = json.load(f)
states_dic={
    #states->new_states[np.newaxis,:]
    "0":[   
            #forward> states:
            [
                #"global_state": 
                {
                    "attack_remain_time":0.0,
                    "match_remain_time": 179.82485961914062,
                    "is_home_team": 1.0,
                    "home_score": 0.0,
                    "away_score": 0.0,
                    #-->27行
                },
                #"self_state": 
                {
                    #-->41行
                    "character_id": 1.0,
                    "position_type": 3.0,
                    "buff_key": 0.0,
                    "buff_value": 0.0,
                    "stature": 206.0,
                    "rational_shoot_distance": 10.0,
                    "position_x": -1.75,
                },
                #"ally_0_state": 
                {},
                #"ally_1_state": 
                {},
                #"enemy_0_state": 
                {},
                #"enemy_1_state": 
                {},
                #"enemy_2_state": 
                {},
                #global_feature = states[0].float()
                #self_feature = states[1]
                #ally0_feature = states[2]
                #ally1_feature = states[3]
                #enemy0_feature = states[4]
                #enemy1_feature = states[5]
                #enemy2_feature = states[6]   
            ],
            [],
            [],
            #batch_size
        ],
}

if __name__ == '__main__':
    import keyword
    import torch
    from torch.utils.tensorboard import SummaryWriter

    # 初始化SummaryWriter
    writer = SummaryWriter('runs/embedding_demo')

    # 创建元数据
    meta = []
    while len(meta) < 100:
        meta = meta + keyword.kwlist
    meta = meta[:100]

    for i, v in enumerate(meta):
        meta[i] = v + str(i)

    # 创建图像标签
    label_img = torch.rand(100, 3, 10, 32)
    for i in range(100):
        label_img[i] *= i / 100.0

    # 添加嵌入到writer
    writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img, global_step=0)
    writer.add_embedding(torch.randn(100, 5), label_img=label_img, global_step=1)
    writer.add_embedding(torch.randn(100, 5), metadata=meta, global_step=2)

    # 关闭writer
    writer.close()
