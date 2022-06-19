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
import task_generator_test as tg
import os
import math
import random
import argparse
import scipy as sp
import scipy.stats
from resnet import resnet12,RR

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 512)  # 64
parser.add_argument("-p","--parent_relation_dim",type = int, default = 128)   # 8
parser.add_argument("-r","--fine_relation_dim",type = int, default = 128)  # 8
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)
parser.add_argument("-e","--episode",type = int, default= 20)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.1)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

# Hyper Parameters
args = parser.parse_args()
FEATURE_DIM = args.feature_dim
FINE_RELATION_DIM = args.fine_relation_dim
PARENT_RELATION_DIM = args.parent_relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS=args.sample_num_per_class
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
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        # out = self.fc2(out)
        return out

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

def get_new_sample_labels(labels):
    new_sample_labels = []
    for i in range(CLASS_NUM):
        idx = i*5
        new_sample_labels.append(labels[idx].item())
    return new_sample_labels

def get_new_labels(labels,depend_labels):  # 获得预测d 父子节点
    
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

    new_parents_labels = [depend_labels[i] for i in labels]

    return new_parents_labels

def get_predict_labels(labels,depend_labels):  # 获得预测的父子节点

    predict_labels = [depend_labels[i] for i in labels]

    return predict_labels

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.tired_imagenet_folders()


    # Step 2: init neural networks
    print("init neural networks")
    feature_encoder = resnet12()
    parent_relation_network = RR(PARENT_RELATION_DIM)
    fine_relation_network = RR(FINE_RELATION_DIM)
    mse = nn.MSELoss().cuda(GPU)# 调用损失函数

   
    feature_encoder.apply(weights_init)   #初始化参数
    parent_relation_network.apply(weights_init)
    fine_relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    parent_relation_network.cuda(GPU)
    fine_relation_network.cuda(GPU)
    
    feature_encoder_optim = torch.optim.SGD(feature_encoder.parameters(),lr=LEARNING_RATE)  # 优化器
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)  # 学习率计算
    parent_relation_network_optim = torch.optim.SGD(parent_relation_network.parameters(),lr=LEARNING_RATE)
    parent_relation_network_scheduler = StepLR(parent_relation_network_optim,step_size=100000,gamma=0.5)
    fine_relation_network_optim = torch.optim.SGD(fine_relation_network.parameters(),lr=LEARNING_RATE)
    fine_relation_network_scheduler = StepLR(fine_relation_network_optim,step_size=100000,gamma=0.5)

    #加载已训练模型
    if os.path.exists(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/SCR_NOH/tiredImageNet_feature_encoder_5way_5shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/SCR_NOH/tiredImageNet_feature_encoder_5way_5shot.pkl")))
        print("load feature encoder success")
    
    if os.path.exists(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/H_SCR_0C/tiredImageNet_parent_relation_network_5way_5shot.pkl")):
        parent_relation_network.load_state_dict(torch.load(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/H_SCR_0C/tiredImageNet_parent_relation_network_5way_5shot.pkl")))
        print("load parent relation network success")
    
    if os.path.exists(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/SCR_NOH/tiredImageNet_fine_relation_network_5way_5shot.pkl")):
        fine_relation_network.load_state_dict(torch.load(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/SCR_NOH/tiredImageNet_fine_relation_network_5way_5shot.pkl")))
        print("load fine relation network success")

    if os.path.exists(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/SCR_NOH/tiredImageNet_feature_encoder_optim_5way_5shot.pkl")):
        feature_encoder_optim.load_state_dict(torch.load(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/SCR_NOH/tiredImageNet_feature_encoder_optim_5way_5shot.pkl")))
        print("load feature encoder optim success")
    
    if os.path.exists(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/H_SCR_0C/tiredImageNet_parent_relation_network_optim_5way_5shot.pkl")):
        parent_relation_network_optim.load_state_dict(torch.load(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/H_SCR_0C/tiredImageNet_parent_relation_network_optim_5way_5shot.pkl")))
        print("load parent relation network optim success")

    if os.path.exists(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/SCR_NOH/tiredImageNet_fine_relation_network_optim_5way_5shot.pkl")):
        fine_relation_network_optim.load_state_dict(torch.load(str("/home/dj212/ZYQ/SSF-HRNET/test_result/tiredImageNet/5-way-5-shot/SCR_NOH/tiredImageNet_fine_relation_network_optim_5way_5shot.pkl")))
        print("load fine relation network optim success")



    total_accuracy = []
    total_fine_accuracy = []
    total_parent_accuracy = []
    for episode in range(EPISODE):

            print("Testing...")

            accuracies = []
            fine_accuracies = []
            parent_accuracies = []
            for test_episode in range(TEST_EPISODE):

                task = tg.TiredImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                sample_dataloader = tg.get_tired_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False )
                test_dataloader = tg.get_tired_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True )
                
                sample_images,sample_coarse_labels,sample_labels = sample_dataloader.__iter__().next()
                test_images,test_coarse_labels,test_labels = test_dataloader.__iter__().next()
                
                sample_labels = torch.Tensor(get_new_sample_labels(sample_labels)).view(-1,1).int()
                sample_coarse_labels = torch.Tensor(get_new_sample_labels(sample_coarse_labels)).view(-1,1).int()

                sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,6,6)
                sample_features = torch.mean(sample_features,1).unsqueeze(0).cuda(GPU) # 1*5*64*19*19
                test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 25*64*19*19

                sample_features_ext = sample_features.repeat(CLASS_NUM*BATCH_NUM_PER_CLASS,1,1,1,1)  # 75*?*64*19*19
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)  # ?*85*64*19*19
                test_features_ext = torch.transpose(test_features_ext,0,1)  # 85*?*64*19*19
                relation_pairs = torch.cat((sample_features_ext,test_features_ext),2) # 85*?*128*19*19
                relation_pairs = relation_pairs.view(-1,FEATURE_DIM*2,6,6)

                fine_relations = fine_relation_network(relation_pairs).view(-1,CLASS_NUM)
                fine_relations = F.softmax(fine_relations,dim=1)

                _,fine_predcit = torch.max(fine_relations.data,1)
                fine_reward =[1 if fine_predcit[j]==test_labels[j] else 0 for j in range(test_labels.size(0))]
                fine_accuracy = np.sum(fine_reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM
                fine_accuracies.append(fine_accuracy)

                sample_parents = {}  # 整合到父节点的数组下
                for i in range(sample_labels.size(0)):
                    if sample_coarse_labels[i].item() not in sample_parents.keys():
                        sample_parents[sample_coarse_labels[i].item()] = [i]
                    else:
                        sample_parents[sample_coarse_labels[i].item()].append(i)
                test_parents_num = len(sample_parents.keys())

                for i, (key,value) in enumerate(sample_parents.items()):

                    sample_parents_features_ext = sample_features[0,value,:]
                    sample_parents_features_ext = sample_parents_features_ext.unsqueeze(0)
                    sample_parents_features_ext = torch.mean(sample_parents_features_ext,1)

                    if i == 0:
                        sample_parents_features_exts = sample_parents_features_ext
                    else:
                        sample_parents_features_exts = torch.cat((sample_parents_features_exts,sample_parents_features_ext),0) # ?*64*19*19
                sample_parents_features_exts = sample_parents_features_exts.unsqueeze(0).repeat(CLASS_NUM*BATCH_NUM_PER_CLASS,1,1,1,1)  # 75*?*64*19*19
                test_features_exts = test_features.unsqueeze(0).repeat(test_parents_num,1,1,1,1)
                test_features_exts = torch.transpose(test_features_exts,0,1)
                parent_relation_pairs = torch.cat((sample_parents_features_exts,test_features_exts),2)
                parent_relation_pairs = parent_relation_pairs.view(-1,FEATURE_DIM*2,6,6)

                parents_relations = parent_relation_network(parent_relation_pairs).view(-1,test_parents_num)
                parents_relations = F.softmax(parents_relations,dim=1)

                _,predict_parents_labels = torch.max(parents_relations.data,1)
                test_predict_parents_labels = torch.Tensor(get_predict_labels(predict_parents_labels,list(sample_parents.keys()))).view(-1,1)
                parents_reward = [1 if test_predict_parents_labels[j]==test_coarse_labels[j] else 0 for j in range(len(test_coarse_labels))]   # 预测正确为1，否则为0

                parent_accuracy = np.sum(parents_reward)/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS
                parent_accuracies.append(parent_accuracy)

                relations = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM))
                for i,(key,value) in enumerate(sample_parents.items()):
                    for j in value:
                        relations[:,j] = 1/2*(0.5*fine_relations[:,j] + 0.5*parents_relations[:,i])

                _,predcit = torch.max(relations.data,1)
                reward =[1 if predcit[j]==test_labels[j] else 0 for j in range(test_labels.size(0))]
                test_accuracy = np.sum(reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM
                accuracies.append(test_accuracy)

                if (test_episode+1)%100 == 0:
                        print("test:%6d, fine_accuracy:%.4f, parent_accuracy:%.4f, test_accuracy:%.4f" %(test_episode+1,fine_accuracy,parent_accuracy,test_accuracy))


            test_accuracy = np.mean(accuracies)
            parent_accuracy = np.mean(parent_accuracies)
            fine_accuracy = np.mean(fine_accuracies)

            print("test: fine_accuracy:%.4f, parent_accuracy:%.4f,  test_accuracy:%.4f" %(fine_accuracy,parent_accuracy,test_accuracy))
            total_accuracy.append(test_accuracy)
            total_fine_accuracy.append(fine_accuracy)
            total_parent_accuracy.append(parent_accuracy)

    test_accuracy,h = mean_confidence_interval(total_accuracy)
    fine_accuracy,fine_h = mean_confidence_interval(total_fine_accuracy)
    parent_accuracy,parent_h = mean_confidence_interval(total_parent_accuracy)
   
    print("avg_test_accuracy:",test_accuracy,"avg_h:",h,"avg_fine_accuracy:",fine_accuracy,"fine_h:",fine_h,"avg_parent_accuracy:",parent_accuracy,"parent_h:",parent_h)


if __name__ == '__main__':
    main()
