
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

import pandas as pd
import torch


from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
from data_loader_DC_CDM import TrainDataLoader, ValTestDataLoader
from model_NCD_DC import Net
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')  # 或者使用 'TkAgg'
import matplotlib.pyplot as plt

# can be changed according to config.txt
exer_n = 81
knowledge_n = 11
student_n = 116805
# can be changed according to command parameter
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
epoch_n = 5


#新增
# 预加载所有学生的 One-Hot 编码
def preload_one_hot_encodings():
    one_hot_dict = {}
    for _, row in df_one_hot.iterrows():
        stu_id = row['STU_ID']
        one_hot_tensor = torch.FloatTensor(row[-6:].values.astype(float))
        one_hot_dict[stu_id] = one_hot_tensor
    return one_hot_dict


# 修改 get_one_hot_by_stu_ids 函数
def get_one_hot_by_stu_ids(stu_ids):
    one_hot_tensors = []
    for stu_id in stu_ids:
        stu_id = stu_id.item()
        if stu_id in one_hot_dict:
            one_hot_tensors.append(one_hot_dict[stu_id])
        else:
            print(f"No data found for STU_ID: {stu_id}")
            one_hot_tensors.append(None)
    return one_hot_tensors


def kl_divergence(mu, logvar, prior_mean, prior_std):
    prior_var = prior_std.pow(2)
    var = logvar.exp()
    kld = -0.5 * torch.sum(1 + logvar - torch.log(prior_var) - ((mu - prior_mean).pow(2) + var) / prior_var)
    return kld

def vae_loss_function(recon_x, x, mu, logvar, prior_mean, prior_std):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = kl_divergence(mu, logvar, prior_mean, prior_std)
    return BCE + KLD

# 梯度裁剪函数
def clip_gradients(model, max_norm=1.0):
    for param in model.parameters():
        if param.grad is not None:
            torch.nn.utils.clip_grad_norm_(param, max_norm)

# 使用 KDE 进行采样
def kde_sample(data, num_samples):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)
    samples = kde.sample(num_samples)
    return samples

# 从训练数据中获取潜在变量 z
def get_latent_variables(model, data_loader):
    model.eval()
    latent_vars = []
    with torch.no_grad():
        for input_stu_ids, input_exer_ids, input_knowledge_embs, labels in data_loader:
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            stu_emb = model.student_emb(input_stu_ids)
            mu, logvar = model.vae.encode(stu_emb)
            z = model.vae.reparameterize(mu, logvar)
            latent_vars.append(z.cpu().numpy())
    return np.concatenate(latent_vars, axis=0)

def train(student_scores):
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002) #飞哥的为lr=0.002 换一下 是因为 出现了梯度爆炸 loss为Nan的情况

    print('training model...')

    loss_function = nn.NLLLoss() #负对数似然损失 也可以做多分类
    for epoch in range(epoch_n):
        data_loader.reset()  # reset() 应该是重新初始化指针
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch() # 按论文中的数据形式 格式化每一批数据
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            # 新增

            one_hot_input = get_one_hot_by_stu_ids(input_stu_ids)
            group_labels = torch.tensor([torch.argmax(encoding).item() for encoding in one_hot_input])
            one_hot_tensor = torch.stack(one_hot_input)
            one_hot_tensor = one_hot_tensor.to(device)
            group_labels = group_labels.to(device)

            output_1,output_emb,output_xianyanfenbu= net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, one_hot_tensor,group_labels) # 开始训练
            # output_1= net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,one_hot_tensor)  # 开始训练
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            output_0_emb = torch.ones(output_emb.size()).to(device) - output_emb
            output_emb = torch.cat((output_0_emb, output_emb), 1)

            output_0_xianyanfenbu= torch.ones(output_xianyanfenbu.size()).to(device) - output_xianyanfenbu
            output_xianyanfenbu = torch.cat((output_0_xianyanfenbu, output_xianyanfenbu), 1)


            loss1 = loss_function(torch.log(output), labels)
            loss_emb = loss_function(torch.log(output_emb), labels)
            loss_xianyanfenbu= loss_function(torch.log(output_xianyanfenbu), labels)
            loss = loss1+loss_emb+loss_xianyanfenbu
            loss.backward()
            # clip_gradients(net) # 新增一个梯度裁剪 ，看看能不能解决梯度爆炸问题
            optimizer.step()
            net.apply_clipper()  # 保持单调性增加

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # validate and save current model every epoch
        rmse, auc = validate(net, epoch)
        save_snapshot(net, 'model/Science_model_NCD_DC_epoch' + str(epoch + 1))

def validate(model, epoch):
    data_loader = ValTestDataLoader('test')      # validation代表使用验证集验证 空 直接使用测试集
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
       #新增
        one_hot_input = get_one_hot_by_stu_ids(input_stu_ids)
        group_labels = torch.tensor([torch.argmax(encoding).item() for encoding in one_hot_input])
        one_hot_tensor = torch.stack(one_hot_input)
        one_hot_tensor = one_hot_tensor.to(device)
        group_labels = group_labels.to(device)

        output,_,_= net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,one_hot_tensor,group_labels) # 验证时不使用KDE采样
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/Science_model_NCD_DC_val.txt', 'a', encoding='utf8') as f:
        f.write('1, epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    # global student_n, exer_n, knowledge_n, device
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    # 读取每位学生的平均分
    average_scores_list = []
    with open('./data/average_scores.txt', 'r', encoding='utf8') as f:
        for line in f:
            average_scores_list.append(float(line.strip()))



    # 读取文件
    df_one_hot = pd.read_csv('./data/one_hot_encoded_scores.csv')
    # 在训练开始时预加载
    one_hot_dict = preload_one_hot_encodings()
    train(average_scores_list)
