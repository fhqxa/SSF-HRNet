#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import random
import argparse
import scipy as sp
import scipy.stats
from resnet import resnet12, RR
from renet import RENet

# device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=512) # 512
parser.add_argument("-p", "--parent_relation_dim", type=int, default=128)  # 128
parser.add_argument("-r", "--fine_relation_dim", type=int, default=128)  #128
parser.add_argument("-w", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=10)
parser.add_argument("-e", "--episode", type=int, default=300000)# 300000
parser.add_argument("-t", "--test_episode", type=int, default=600)  # 600
parser.add_argument("-l", "--learning_rate", type=float, default=0.1)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)

# parser.add_argument("-self_method", "--self_method", default='scr')

args = parser.parse_args()
FEATURE_DIM = args.feature_dim
FINE_RELATION_DIM = args.fine_relation_dim
PARENT_RELATION_DIM = args.parent_relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

# 用于提取模型特征
class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        #out = out.view(out.size(0),-1)
        return out4  # 64

# 用于计算两个拼接在一起的特征的相关性系数
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        # out = self.fc2(out)
        return out

# 模型每一层的参数初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def get_new_labels(labels,depend_labels):  # 获得对应的节点标签
    
    new_labels = []
    for c in labels:
        label = []
        for j in range(len(depend_labels)):
            if c == depend_labels[j]:
                label.append(1)
            else:
                label.append(0)
        new_labels.append(label)
    return new_labels

def get_predict_labels(labels,depend_labels):  # 获得预测的父子节点

    predict_labels = [depend_labels[i] for i in labels]

    return predict_labels


def main(args):
    # Step 1: init data folders
# 将数据集划分为训练集和测试集
    print("init data folders")
    # init character folders for dataset construction
    # there are 351 metatrain_folders paths   metatest_folders  97
    metatrain_folders,metatest_folders = tg.tired_imagenet_folders()

    # Step 2: init neural networks
    print("init neural networks")


    feature_encoder = resnet12()
    parent_relation_network = RR(PARENT_RELATION_DIM)
    fine_relation_network = RR(FINE_RELATION_DIM)
    
    mse = nn.MSELoss().cuda(GPU)

    feature_encoder.apply(weights_init)
    parent_relation_network.apply(weights_init)
    fine_relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    parent_relation_network.cuda(GPU)
    fine_relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.SGD(feature_encoder.parameters(), momentum=0.9, lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)

    parent_relation_network_optim = torch.optim.SGD(parent_relation_network.parameters(), momentum=0.9, lr=LEARNING_RATE)
    parent_relation_network_scheduler = StepLR(parent_relation_network_optim, step_size=100000, gamma=0.5)

    fine_relation_network_optim = torch.optim.SGD(fine_relation_network.parameters(), momentum=0.9, lr=LEARNING_RATE)
    fine_relation_network_scheduler = StepLR(fine_relation_network_optim, step_size=100000, gamma=0.5)

    print("Training...")

    last_accuracy = 0
    last_episode = 0
    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        parent_relation_network_scheduler.step(episode)
        fine_relation_network_scheduler.step(episode)

        task = tg.TiredImagenetTask(metatrain_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)

        sample_dataloader = tg.get_tired_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)

        batch_dataloader = tg.get_tired_imagenet_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True)

        samples, sample_coarse_labels, sample_labels = sample_dataloader.__iter__().next()
        batches, batch_coarse_labels, batch_labels = batch_dataloader.__iter__().next()
        sample_features = feature_encoder(Variable(samples).cuda(GPU)).unsqueeze(0)
        batch_features = feature_encoder(Variable(batches).cuda(GPU))

        print("batch_features", batch_features)

        sample_features_ext = sample_features.repeat(BATCH_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relations_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 6, 6)
        fine_relations = fine_relation_network(relations_pairs).view(-1, CLASS_NUM)

        fine_relations = F.softmax(fine_relations, dim=1)
        _, fine_predcit = torch.max(fine_relations.data, 1)
        fine_reward = [1 if fine_predcit[j] == batch_labels[j] else 0 for j in range(batch_labels.size(0))]
        fine_accuracy = np.sum(fine_reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM

        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1), 1)).cuda(GPU)
        fine_loss = mse(fine_relations, one_hot_labels)

        sample_parents = {}

        for i in range(sample_labels.size(0)):
            if sample_coarse_labels[i].item() not in sample_parents.keys():
                sample_parents[sample_coarse_labels[i].item()] = [i]
            else:
                sample_parents[sample_coarse_labels[i].item()].append(i)

        batch_parents_num = len(sample_parents.keys())

        for i, (key, value) in enumerate(sample_parents.items()):
            sample_parents_features_ext = sample_features[0, value, :]

            sample_parents_features_ext = sample_parents_features_ext.unsqueeze(0)
            sample_parents_features_ext = torch.mean(sample_parents_features_ext, 1)

            if i == 0:
                sample_parents_features_exts = sample_parents_features_ext
            else:
                sample_parents_features_exts = torch.cat((sample_parents_features_exts, sample_parents_features_ext), 0)


        sample_parents_features_exts = sample_parents_features_exts.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1, 1)
        batch_features_exts = batch_features.unsqueeze(0).repeat(batch_parents_num, 1, 1, 1, 1)
        batch_features_exts = torch.transpose(batch_features_exts, 0, 1)
        parent_relation_pairs = torch.cat((sample_parents_features_exts, batch_features_exts), 2)
        parent_relation_pairs = parent_relation_pairs.view(-1, FEATURE_DIM*2, 6, 6)
        parents_relations = parent_relation_network(parent_relation_pairs).view(-1, batch_parents_num)


        parents_relations = F.softmax(parents_relations, dim=1)
        new_batch_parents_labels = torch.Tensor(get_new_labels(batch_coarse_labels, list(sample_parents.keys()))).view(-1, batch_parents_num).cuda(GPU)

        parent_loss = mse(parents_relations, new_batch_parents_labels)
        _, predict_parents_labels = torch.max(parents_relations.data, 1)
        batch_predict_parents_labels = torch.Tensor(get_predict_labels(predict_parents_labels, list(sample_parents.keys()))).view(-1, 1)

        parents_reward = [1 if batch_predict_parents_labels[j] == batch_coarse_labels[j] else 0 for j in range(len(batch_coarse_labels))]
        parent_accuracy = np.sum(parents_reward)/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS


        relations = torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).cuda(GPU)
        for i, (key, value) in enumerate(sample_parents.items()):
            for j in value:
                relations[:, j] = 1/2*(0.5*fine_relations[:, j] + 0.5*parents_relations[:, i])


        _,predcit = torch.max(relations.data, 1)
        reward = [1 if predcit[j] == batch_labels[j] else 0 for j in range(batch_labels.size(0))]
        train_accuracy = np.sum(reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM

        loss = fine_loss+0.1*parent_loss

        if (episode+1) % 100 == 0:
            print("train:%6d, fine_loss:%.6f, parent_loss:%.6f, loss:%.6f, fine_accuracy:%.4f, parent_accuracy:%.4f, train_accuracy:%.4f" %(episode+1, fine_loss, parent_loss, loss, fine_accuracy, parent_accuracy, train_accuracy))

        feature_encoder.zero_grad()
        parent_relation_network.zero_grad()
        fine_relation_network.zero_grad()  
        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)  # 解决梯度爆炸
        torch.nn.utils.clip_grad_norm_(parent_relation_network.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(fine_relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        parent_relation_network_optim.step()
        fine_relation_network_optim.step()


        if (episode+1)%5000== 0:


            print("Testing...")
            accuracies = []
            fine_accuracies = []
            parent_accuracies = []
            for test_episode in range(TEST_EPISODE):

                task = tg.TiredImagenetTask(metatest_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)

                sample_dataloader = tg.get_tired_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False )
                test_dataloader = tg.get_tired_imagenet_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True)
                sample_images, sample_coarse_labels, sample_labels = sample_dataloader.__iter__().next()
                test_images, test_coarse_labels, test_labels = test_dataloader.__iter__().next()

                sample_features = feature_encoder(Variable(sample_images).cuda(GPU)).unsqueeze(0)
                test_features = feature_encoder(Variable(test_images).cuda(GPU))
                
                sample_features_ext = sample_features.repeat(BATCH_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)

                c = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 6, 6)
                fine_relations = fine_relation_network(c).view(-1, CLASS_NUM)
                fine_relations = F.softmax(fine_relations, dim=1)

                _,fine_predcit = torch.max(fine_relations.data, 1)
                fine_reward =[1 if fine_predcit[j]==test_labels[j] else 0 for j in range(test_labels.size(0))]
                fine_accuracy = np.sum(fine_reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM
                fine_accuracies.append(fine_accuracy)

                for i in range(sample_labels.size(0)):
                    if sample_coarse_labels[i].item() not in sample_parents.keys():
                        sample_parents[sample_coarse_labels[i].item()] = [i]
                    else:
                        sample_parents[sample_coarse_labels[i].item()].append(i)
                test_parents_num = len(sample_parents.keys())
                
                for i, (key,value) in enumerate(sample_parents.items()):
                    
                    sample_parents_features_ext = sample_features[0,value,:]  # ?*64*19*19
                    sample_parents_features_ext = sample_parents_features_ext.unsqueeze(0) # 1*?*64*19*19
                    sample_parents_features_ext = torch.mean(sample_parents_features_ext,1)  # 1*64*19*19  # 获取每个父类特征的平均值

                    if i == 0:
                        sample_parents_features_exts = sample_parents_features_ext
                    else:
                        sample_parents_features_exts = torch.cat((sample_parents_features_exts,sample_parents_features_ext), 0) # ?*64*19*19

                sample_parents_features_exts = sample_parents_features_exts.unsqueeze(0).repeat(CLASS_NUM*BATCH_NUM_PER_CLASS, 1, 1,1,1)  # 75*?*64*19*19
                test_features_exts = test_features.unsqueeze(0).repeat(test_parents_num, 1, 1, 1, 1)
                test_features_exts = torch.transpose(test_features_exts, 0, 1)
                parent_relation_pairs = torch.cat((sample_parents_features_exts,test_features_exts), 2)
                parent_relation_pairs = parent_relation_pairs.view(-1, FEATURE_DIM*2, 6, 6)

                parents_relations = parent_relation_network(parent_relation_pairs).view(-1,test_parents_num)
                parents_relations = F.softmax(parents_relations,dim=1)
                
                _,predict_parents_labels = torch.max(parents_relations.data,1)
                test_predict_parents_labels = torch.Tensor(get_predict_labels(predict_parents_labels,list(sample_parents.keys()))).view(-1,1)
                parents_reward = [1 if test_predict_parents_labels[j]==test_coarse_labels[j] else 0 for j in range(len(test_coarse_labels))]   # 预测正确为1，否则为0

                parent_accuracy = np.sum(parents_reward)/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS 
                parent_accuracies.append(parent_accuracy)
            
                relations = torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).cuda(GPU)
                for i, (key, value) in enumerate(sample_parents.items()):
                    for j in value:
                        relations[:, j] = 1/2*(0.5*fine_relations[:, j] + 0.5*parents_relations[:, i])
                _, predcit = torch.max(relations.data, 1)
                reward =[1 if predcit[j]==test_labels[j] else 0 for j in range(test_labels.size(0))]
                test_accuracy = np.sum(reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM
                accuracies.append(test_accuracy)

                if (test_episode+1)%100 == 0:
                    print("test:%6d, fine_accuracy:%.4f, parent_accuracy:%.4f, test_accuracy:%.4f" %(test_episode+1,fine_accuracy,parent_accuracy,test_accuracy))


            test_accuracy, h = mean_confidence_interval(accuracies)
            parent_accuracy, parent_h = mean_confidence_interval(parent_accuracies)
            fine_accuracy, fine_h = mean_confidence_interval(fine_accuracies)

            print("test: fine_accuracy:%.4f, fine_h:%.4f, parent_accuracy:%.4f, parent_h:%.4f, test_accuracy:%.4f, test_h:%.4f, last_episode:%6d, last_accuracy:%.4f" %(fine_accuracy,fine_h,parent_accuracy,parent_h,test_accuracy,h,last_episode,last_accuracy))

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(feature_encoder.state_dict(), str("/home/dj212/ZYQ/HMRN-tiredimagenet-scr/test_result/tiredImageNet/5-way-1-shot/H_noscr_0C/tiredImageNet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(parent_relation_network.state_dict(), str("/home/dj212/ZYQ/HMRN-tiredimagenet-scr/test_result/tiredImageNet/5-way-1-shot/H_noscr_0C/tiredImageNet_parent_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(fine_relation_network.state_dict(), str("/home/dj212/ZYQ/HMRN-tiredimagenet-scr/test_result/tiredImageNet/5-way-1-shot/H_noscr_0C/tiredImageNet_fine_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(feature_encoder_optim.state_dict(), str("/home/dj212/ZYQ/HMRN-tiredimagenet-scr/test_result/tiredImageNet/5-way-1-shot/H_noscr_0C/tiredImageNet_feature_encoder_optim_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(parent_relation_network_optim.state_dict(), str("/home/dj212/ZYQ/HMRN-tiredimagenet-scr/test_result/tiredImageNet/5-way-1-shot/H_noscr_0C/tiredImageNet_parent_relation_network_optim_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(fine_relation_network_optim.state_dict(), str("/home/dj212/ZYQ/HMRN-tiredimagenet-scr/test_result/tiredImageNet/5-way-1-shot/H_noscr_0C/tiredImageNet_fine_relation_network_optim_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                with open('/home/dj212/ZYQ/HMRN-tiredimagenet-scr/test_result/tiredImageNet/5-way-1-shot/H_noscr_0C/jilu.txt', 'w') as f:
                    f.write(str(test_accuracy)+str(episode + 1))

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy
                last_episode = episode + 1
    
    print("best episode: ", last_episode, "best accuracy: ", last_accuracy)  # , "h: ", h


if __name__ == '__main__':
    main(args)
