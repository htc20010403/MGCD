import torch
import DL_junyi
from model import KANCD_group
import pandas as pd
import os
import os.path as osp
import params
from torch.utils.data import TensorDataset, DataLoader
import json
if params.dataset == 'data/junyi/':
    src, tgt = DL_junyi.CD_DL()
device = 'cuda:0'
DATA_PATH = os.path.join(os.getcwd(),'data/junyi')
def my_collate_group(batch):
    input_stu_ids, input_topic_ids, input_topic_embs,ys = [], [], [] ,[]
    for log in batch:
        y = log['correct_rate']
        if "topic_id" not in log:
            topic_emb = [1.0] * params.topic_num
        else:
            topic_emb = [0.] * params.topic_num
            topic_emb[log['topic_id']] = 1.0
        input_stu_ids.append(log['user_id'])
        input_topic_ids.append(log['topic_id'])
        input_topic_embs.append(topic_emb)
        ys.append(y)

    return torch.LongTensor(input_stu_ids), torch.LongTensor(input_topic_ids), torch.Tensor(input_topic_embs),torch.Tensor(ys)
def load_group_data():
    group_test_dataset="group_data_test_10_13.json"
    group_train_dataset="group_data_train_10_13.json"
    with open(params.dataset+group_train_dataset) as i_f:
        src_dataset = json.load(i_f)
    with open(params.dataset+group_test_dataset) as i_f:
        tgt_dataset = json.load(i_f)
    src_DL = DataLoader(dataset=src_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=my_collate_group)
    tgt_DL = DataLoader(dataset=tgt_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=my_collate_group)
    return src_DL, tgt_DL
src_group,tgt_group=load_group_data()

def KaNCD_group_main():
    cdm = KANCD_group.KaNCD(params.en,params.un,params.kn, params.latent_dim)
    e, auc, acc, rmse = cdm.train_group(train_data=src, test_data=tgt, epoch=200, device=device, lr=0.001,
                                  train_group_data=src_group, test_group_data=tgt_group)
    with open('result/KaNCD.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, auc= %f\n' % (e, acc, auc))

if __name__ == '__main__':
    KaNCD_group_main()
