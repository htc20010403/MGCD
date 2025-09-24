
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
# from EduCDM import CDM
import sys
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import random
import params
class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

dataset = 'data/junyi/'
device='cuda:0'
def get_gauss_rand():
    random_number = random.gauss(0, 1)
    while random_number < 0 or random_number > 1:
        random_number = random.gauss(0, 1)
    return random_number
with open(dataset + 'grain.txt') as f:
    f.readline()
    tn, an = f.readline().split(',')
    tn, an = int(tn), int(an)
with open(dataset + 'name_topic.txt', 'r') as f:  # name topic
    dict_nt = {}
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        dict_nt[int(line[0])] = int(line[1])
with open(dataset + 'topic_area.txt', 'r') as f:  # topic area
    dict_ta = {}
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        dict_ta[int(line[0])] = int(line[1])
class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        nn.init.xavier_normal_(self.knowledge_emb)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.prednet_full4 = PosLinear(params.topic_num, self.prednet_len1)
        self.prednet_full5 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.prednet_full6 = PosLinear(self.prednet_len2, 1)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)

            self.k_diff_full_group = nn.Linear(self.emb_dim, 1)
            self.stat_full_group = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)

            self.k_diff_full_group = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full_group = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

            self.k_diff_full1_group = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2_group = nn.Linear(self.emb_dim, 1)
            self.stat_full1_group = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2_group = nn.Linear(self.emb_dim, 1)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 加权部分
        # homoGCD
        self.h_homo = nn.Parameter(torch.rand(256, 1))
        nn.init.xavier_normal_(self.h_homo)
        self.W_homo = nn.Linear(1, 256)

        # homoGCD
        self.h_homo_know = nn.Parameter(torch.rand(256, 1))
        nn.init.xavier_normal_(self.h_homo_know)
        self.W_homo_know = nn.Linear(1, 256)

        # 粗粒度部分
        self.high_know = nn.Embedding(params.topic_num, self.emb_dim)  # 共有的粗粒度部分

        # 隐藏部分

        self.stu_hidden = nn.Embedding(self.student_n, 1)

        # 切分部分

        self.k_v_share = nn.Linear(self.emb_dim, self.emb_dim)
        self.k_v_special = nn.Linear(self.emb_dim, self.emb_dim)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        init.eye_(self.k_v_special.weight)  # 权重 = I
        # init.zeros_(self.k_v_special.bias)
        # nn.init.xavier_normal_(self.knowledge_emb)
        topic_name_list = []
        for i in range(835, 835 + 40):
            linshi = []
            for j in range(835):
                if j not in dict_nt:
                    continue
                if dict_nt[j] == int(i):
                    linshi.append(j)
            topic_name_list.append(linshi)
        self.topic_name_list = topic_name_list
        area_topic_list = []
        for i in range(835 + 40, 835 + 48):
            linshi = []
            for j in range(835, 835 + 40):
                if j not in dict_ta:
                    continue
                if dict_ta[j] == i:
                    linshi.append(j - 835)
            area_topic_list.append(linshi)
        self.area_topic_list = area_topic_list

    def forward_multi(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(input_exercise)
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # prednet
        input_knowledge_point = self.select_knowledge_exer(input_knowledge_point)
        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)
    def select_knowledge_exer(self, kq):
        # 事实上已经有了，那直接筛选就行
        # 指定想要保留的列索引
        keep_cols = list(range(835))

        # 创建一个掩码矩阵
        mask = torch.zeros_like(kq)
        mask[:, keep_cols] = 1
        kq = kq * mask
        return kq
    def pass_master_to_high_homo(self, posterior,method=None,stu_hidden_factor=None):  # 把835扩充到883
        if method==None:
            high_level = torch.rand(posterior.shape[0], 48).to(device)
            posterior = posterior.to(device)
            for i in range(835, 835 + 40):
                linshi = self.topic_name_list[i - 835]
                if len(linshi) == 0:
                    high_level[:, i - 835, :] = get_gauss_rand()  # 随机一个掌握
                else:
                    # high_level[:, i - 835:i - 835 + 1, :] = self.average_of_row(posterior, linshi)
                    high_level[:, i - 835:i - 835 + 1] = self.average_of_row_homo(posterior, linshi)
            return high_level[:, :40]
        elif method=="master":
            high_level = torch.rand(posterior.shape[0], 48).to(device)
            posterior = posterior.to(device)
            for i in range(835, 835 + 40):
                linshi = self.topic_name_list[i - 835]
                if len(linshi) == 0:
                    high_level[:, i - 835, :] = get_gauss_rand()  # 随机一个掌握
                else:
                    # high_level[:, i - 835:i - 835 + 1, :] = self.average_of_row(posterior, linshi)
                    high_level[:, i - 835:i - 835 + 1] = self.average_of_row_homo(posterior, linshi,"master")
            return high_level[:, :40]
        else:
            # print(1)
            high_level = torch.rand(40, posterior.shape[1]).to(device)
            posterior = posterior.to(device)
            for i in range(835, 835 + 40):
                linshi = self.topic_name_list[i - 835]
                if len(linshi) == 0:
                    high_level[i - 835, :] = get_gauss_rand()  # 随机一个掌握
                else:
                    # high_level[i - 835:i - 835 + 1, :] = self.average_of_row(posterior, linshi)
                    high_level[i - 835:i - 835 + 1, :] = self.average_of_row(posterior, linshi)
            return high_level

    def average_of_row_homo(self,tensor,indices,method=None,stu_hidden_factor=None):
        all_emb=tensor[:,indices]
        all_emb=all_emb.unsqueeze(-1)
        tensor_1=torch.relu(self.W_homo(all_emb))
        expanded_homo=self.h_homo.view(1,1,256)
        tensor_2=tensor_1*expanded_homo
        tensor_2=tensor_2.sum(dim=2,keepdim=True)
        weights=torch.softmax(tensor_2,dim=1)
        final_emb=(weights*all_emb).sum(dim=1,keepdim=True)
        final_emb=final_emb.squeeze(-1)
        return final_emb

    def k_diff_group_homo(self,emb):
        first_emb=self.pass_master_to_high_homo(emb)
        # print("end")
        temp_difficulties = torch.zeros(self.knowledge_n, 1)  # 变成835*128
        for i in range(self.exer_n):
            if i in dict_nt:
                temp_difficulties[i] = first_emb[i, dict_nt[i] - 835]
            else:
                temp_difficulties[i] = get_gauss_rand()
        # print(temp_difficulties.size())
        temp_difficulties=temp_difficulties.transpose(0,1)
        # temp_difficulties=temp_difficulties.view(1,temp_difficulties.shape[0],temp_difficulties.shape[1])
        final_emb=self.pass_master_to_high_homo(temp_difficulties)
        final_emb=final_emb.transpose(0,1)
        # print(final_emb.size())
        return final_emb

    def average_of_row_homo_know(self,tensor,indices):
        all_emb = tensor[indices]
        all_emb = all_emb.unsqueeze(-1)
        tensor_1 = torch.relu(self.W_homo_know(all_emb))
        expanded_homo = self.h_homo_know
        tensor_2 = torch.matmul(tensor_1, expanded_homo)
        tensor_2 = tensor_2.sum(dim=1, keepdim=True)
        weights = torch.softmax(tensor_2, dim=0)
        # print(weights)
        final_emb = (weights * all_emb).sum(dim=0, keepdim=True)
        # print(final_emb.size())
        final_emb=final_emb.squeeze(2)
        return final_emb

    def average_of_row(self, tensor, indices):#平均加权
        tensor = tensor.to(device)
        output_data = tensor[indices,:].mean(dim=0, keepdim=True)
        # print(output_data.size())
        return output_data

    def e_disc(self, emb):
        middle_emb = torch.rand(params.topic_num, 1).to(device)
        emb = emb.to(device)
        for i in range(params.topic_num):
            linshi = self.topic_name_list[i]
            if len(linshi) == 0:
                middle_emb[i] = get_gauss_rand()  # 随机一个掌握
            else:
                output_data = emb[linshi].mean(dim=0, keepdim=True)
                # print(output_data.size())
                middle_emb[i] = output_data
        # print(middle_emb.size())
        # print(middle_emb)
        return middle_emb
    # def e_disc(self, emb):
    #     middle_emb = torch.rand(40, 1).to(device)
    #     emb = emb.to(device)
    #     for i in range(40):
    #         linshi = self.topic_name_list[i]
    #         if len(linshi) == 0:
    #             middle_emb[i] = get_gauss_rand()  # 随机一个掌握
    #         else:
    #             x = emb[linshi]
    #             softmax_weights = F.softmax(x, dim=0)
    #             middle_emb[i] = torch.sum(softmax_weights * x)
    #     return middle_emb

    def orthogonal_loss(self,E_s, E_t):
        inner_product = torch.matmul(E_s.T, E_t)
        return torch.norm(inner_product)

    def forward_group_homo(self,stu_id,topic_emb):
        stu_emb = self.student_emb(stu_id)
        input_exercise = torch.tensor(list(range(self.exer_n)), dtype=torch.int).to(device)
        exer_emb = self.exercise_emb(input_exercise)
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb
        # print(knowledge_emb.size())

        # print(x0.size())
        knowledge_share = self.k_v_share(knowledge_emb)
        knowledge_special = self.k_v_special(knowledge_emb)

        cu_knowledge = self.pass_master_to_high_homo(knowledge_share, "knowledge")
        # print(cu_knowledge.size())
        # print(knowledge_special[:835,:].size())
        # print(cu_knowledge.size())
        self.high_tensor = torch.cat((knowledge_special[:835, :], cu_knowledge), dim=0)
        # 拼接两个张量
        knowledge_low_emb = torch.cat((self.high_tensor, knowledge_emb[-8:, :]), dim=0)  # 形状为 (46, 128)
        # print(knowledge_low_emb.size())

        knowledge_emb=knowledge_low_emb
        # print(knowledge_emb.size())

        knowledge_emb_exer = knowledge_emb.unsqueeze(0).repeat(self.exer_n, 1, 1)
        knowledge_emb = knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        input_exercise = torch.tensor(list(range(self.exer_n)), dtype=torch.int).to(device)

        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1_group(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2_group(stat_emb)).view(batch, -1)
        stu_hidden_factor=[]
        # print(stat_emb.size()) 128*883
        #算了，还是修饰一下stat_emb吧，更符合我的代码，不管怎样，都变成128*40的就行了 128*883
        stat_emb_group=self.pass_master_to_high_homo(stat_emb,"master")
        # print(stat_emb_group.size())
        # batch, dim = exer_emb.size()
        exer_emb = exer_emb.unsqueeze(1).expand(-1, self.knowledge_n, -1)
        # print(exer_emb.size())
        # print(knowledge_emb_exer.size())

        #倒也是好改了，直接把exer_emb给聚合一下，放进去算了，下面每一个全连接层都得换一下。
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb_exer)).view(self.exer_n, -1)
        elif self.mf_type == 'ncf1':

            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb_exer), dim=-1))).view(self.exer_n, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        # print(k_difficulty.size())
        k_difficulty_group=self.k_diff_group_homo(k_difficulty)
        # print(k_difficulty_group.size())
        k_difficulty_group=k_difficulty_group.repeat(batch,1).view(batch,params.topic_num)
        # print(k_difficulty_group.size())
        #这个更简单，直接也换成一个直接的，或者接一个homo
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))
        e_discrimination_group=self.e_disc(e_discrimination)
        e_discrimination_group=e_discrimination_group.repeat(batch,1).view(batch,params.topic_num)
        # print(e_discrimination_group.size())

        input_x = e_discrimination_group * (stat_emb_group - k_difficulty_group) * topic_emb
        # print(input_x.size())
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full4(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        zhengjiao_loss = self.orthogonal_loss(knowledge_share, knowledge_special)
        zhengjiao_loss = zhengjiao_loss * 0.1
        return output_1.view(-1),zhengjiao_loss


class KaNCD:
    def __init__(self, exer_n,student_n,knowledge_n,dim):
        super(KaNCD, self).__init__()
        mf_type = 'gmf'
        self.net = Net(exer_n, student_n, knowledge_n, mf_type, dim)

    def train_group(self, train_group_data,test_group_data,train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.net = self.net.to(device)
        self.net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        best_epoch = 0
        best_auc = 0.
        acc1 = 0.
        best_f1 = 0.
        rmse1 = 1.
        best_mae = 1.
        best_rmse = 1.

        best_multi_auc = 0.
        best_multi_acc = 0.
        best_multi_rmse = 1.
        best_multi_epoch = 0

        MSE = nn.MSELoss()

        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i, file=sys.stdout):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.net.forward_multi(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average stu loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            # if test_data is not None:
            #     auc, acc, rmse,f1= self.eval_multi(test_data)
            #     print("[Epoch %d] auc: %.6f, acc: %.6f, rmse: %.6f'" % (epoch_i, auc, acc,rmse))
            #     if auc > best_auc:
            #         best_multi_epoch = epoch_i
            #         best_multi_rmse = rmse
            #         best_multi_acc = acc
            #         best_multi_auc = auc
            #         self.save("params/KANCD_1_4.params")
            # print(
            #     'BEST epoch<%d>, auc: %.6f, acc: %.6f, rmse: %.6f' % (best_multi_epoch, best_multi_auc, best_multi_acc,best_multi_rmse))
            # print("这是细粒度！！！！！！")
            self.net = self.net.to(device)
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_group_data, "Epoch %s" % epoch_i, file=sys.stdout):
                batch_count += 1
                user_id, topic_id, topic_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                topic_id: torch.Tensor = topic_id.to(device)
                topic_emb: torch.Tensor = topic_emb.to(device)
                y: torch.Tensor = y.to(device)
                # pred = self.model.forward_group_homo(user_id,topic_emb)
                pred, zhengjiao_loss = self.net.forward_group_homo(user_id, topic_emb)  # 记得看看评估的改了没有！！！
                loss = MSE(pred, y)
                loss += zhengjiao_loss
                # loss.backward()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.mean().item())
                # epoch_losses.append(loss.mean().item())
            # optimizer.step()
            # optimizer.zero_grad()

            print("[Epoch %d] average group_loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_group_data is not None:
                rmse, mae = self.eval_group_homo(test_group_data, device=device)
                print("[Epoch %d] rmse: %.6f, mae: %.6f" % (epoch_i, rmse, mae))
                if rmse < best_rmse:
                    best_epoch = epoch_i
                    best_rmse = rmse
                    best_mae = mae
                    # self.save("params/KANCD_group_1_4.params")
            print(
                'BEST epoch<%d>, rmse: %.6f, mae: %.6f' % (best_epoch, best_rmse, best_mae))
            print("这是homo+第一种训练方法+全数据集+正交loss+知识点单独参数+special和平均拼接+平均区分度+gmf！！！！！！")

        return best_epoch, best_auc, acc1

    def eval_group_homo(self, test_data, device="cpu"):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating", file=sys.stdout):
            user_id, topic_id, topic_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            topic_id: torch.Tensor = topic_id.to(device)
            topic_emb: torch.Tensor = topic_emb.to(device)
            # print(topic_emb)
            # pred= self.model.forward_group_homo(user_id,topic_emb)

            pred,loss=self.net.forward_group_homo(user_id,topic_emb)

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        return rmse, mae

    def eval_multi(self, test_data, device="cpu"):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating", file=sys.stdout):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.net.forward_multi(user_id, item_id, knowledge_emb)

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), rmse, f1_score(y_true,
                                                                                                              np.array(
                                                                                                                  y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)

    def advantage(self, test_data, device="cpu"):
        self.net = self.net.to(device)
        self.net.eval()
        label, feature = [], []
        for batch_data in tqdm(test_data, "Get advantage", file=sys.stdout):
            user_id, item_id, kq, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            kq = kq.to(device)
            advantage = self.net.advantage(user_id, item_id, kq)

            feature.extend(advantage.detach().cpu().tolist())
            label.extend(y.tolist())
        return feature, label

    def pro_case(self, test_data, device="cpu"):
        self.net = self.net.to(device)
        self.net.eval()
        feature, feature_s = [], []
        for batch_data in tqdm(test_data, "Get some pro", file=sys.stdout):
            user_id, item_id, kq, _ = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            kq = kq.to(device)
            pro, pro_s = self.net.pro_case(user_id, item_id, kq)

            feature.extend(pro.detach().cpu().tolist())
            feature_s.extend(pro_s.detach().cpu().tolist())

        return feature, feature_s

    def student_v(self, device="cpu"):
        self.load("params/ncdm.params")
        stu_v = self.net.params(device)
        return stu_v

    def exer_v(self, device="cpu"):
        self.load("params/ncdm.params")
        exer_v = self.net.exer_v(device)
        return exer_v
