#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:49:19 2020

@author: qinfang
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision

import cv2
#import matplotlib.pyplot as plt

import os

from scipy import io

class Conv5NN(nn.Module):
    def __init__(self):
        super(Conv5NN,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                                    nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                                    nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(128,256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                                    nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer4 = nn.Sequential(nn.Conv2d(256,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                                    nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer5 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                                    nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc = nn.Sequential(nn.Linear(1024,128),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                nn.Linear(128,32),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                nn.Linear(32,11)
                                )
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
def cnn(train_size = 350,
    val_size = 75,
    test_size = 75,
    extra_train_size = 700,
    extra_test_size = 150,
    extra_val_size = 150,
    non_digit_size = 1500,
    increase_perf_cnt_cutoff = 5,
    num_epochs = 10,
    tuning = False,
    default_lr = 0.001,
    default_batch_size = 128,
    simple_cnn_saved_name = 'svhn_11_cnn_tuned.pth',
    vgg_saved_name = 'svhn_11_vgg16_tuned.pth'):

    best_lr = default_lr
    best_batch_size = default_batch_size
    increase_perf_cnt = 0
    
    print("----- loading data ------")
    train_data = io.loadmat('train_32x32.mat')
    test_data = io.loadmat('test_32x32.mat')
    x_train = train_data['X'].astype(np.float32)/255.0
    y_train = (train_data['y'][:,0]).astype(np.int32)
    x_test = test_data['X'].astype(np.float32)/255.0
    y_test = (test_data['y'][:,0]).astype(np.int32)
    

    selected_train_index = []
    selected_test_index=[]
    selected_val_index=[]

    for i in range(1,11):
        a = np.where(y_train == i)[0][:train_size]
        selected_train_index.append(a)
        b = np.where(y_test == i)[0][:test_size]
        selected_test_index.append(b)
        c = np.where(y_test == i)[0][test_size:test_size+val_size]
        selected_val_index.append(c)
    train_idx = np.concatenate(selected_train_index)
    test_idx = np.concatenate(selected_test_index)
    val_idx = np.concatenate(selected_val_index)
    selected_xtrain = x_train[:,:,:,train_idx]
    selected_ytrain = y_train[train_idx,]
    selected_xtest = x_test[:,:,:,test_idx]
    selected_ytest =  y_test[test_idx,]
    selected_xval = x_test[:,:,:,val_idx]
    selected_yval = y_test[val_idx,]
    selected_xtrain = np.transpose(selected_xtrain,(3,2,0,1)).astype(np.float32)
    selected_xtest = np.transpose(selected_xtest,(3,2,0,1)).astype(np.float32)
    selected_xval = np.transpose(selected_xval,(3,2,0,1)).astype(np.float32)
    print("----- loading extra corner data ------")
    f_list = os.listdir("train")
    imgs = []
    for f in f_list:
        if f[-4:] == '.png':
            
            img = cv2.imread('train/'+f)
            h,w = img.shape[0],img.shape[0]
    
            if w > 64 and h > 64:
                imgs.append(img[:32,:32,:])
                if len(imgs) >= non_digit_size:
                    
                    break
                imgs.append(img[:32,-32:,:])
                if len(imgs) >= non_digit_size:
                    break
                imgs.append(img[-32:,-32:,:])
                if len(imgs) >= non_digit_size:
                    break
                imgs.append(img[-32:,:32,:])
                if len(imgs) >= non_digit_size:
                    break
            elif w > 64 and h >= 32:
                imgs.append(img[:32,:32,:])
                if len(imgs) >= non_digit_size:
                    break
                imgs.append(img[:32,-32:,:])
                if len(imgs) >= non_digit_size:
                    break
            elif h > 64 and w >=32:
                imgs.append(img[:32,:32,:])
                if len(imgs) >= non_digit_size:
                    break
                imgs.append(img[-32:,:32,:])
                if len(imgs) >= non_digit_size:
                    break
    print("loaded non-digit images")
    corners = np.transpose(np.stack(imgs),(0,3,1,2)).astype(np.float32)/255.0
    ids = np.random.choice(corners.shape[0], corners.shape[0],False)
    train_ids = ids[:int(corners.shape[0]*0.7)]
    test_ids = ids[int(corners.shape[0]*0.7):int(corners.shape[0]*0.85)]
    val_ids = ids[int(corners.shape[0]*0.85):]
    corners_train = corners[train_ids]
    corners_test = corners[test_ids]
    corners_val = corners[val_ids]
    corners_train_y = np.zeros(corners_train.shape[0])
    corners_test_y = np.zeros(corners_test.shape[0])
    corners_val_y = np.zeros(corners_val.shape[0])
    print("----- loading extra digit data ------")
    extra_data = io.loadmat('extra_32x32.mat')
    x_extra = extra_data['X'].astype(np.float32)/255.0
    y_extra = (extra_data['y'][:,0]).astype(np.int32)
    
    selected_extra_train_index = []
    selected_extra_test_index=[]
    selected_extra_val_index=[]
    for i in range(1,11):
        extra_id = np.where(y_extra == i)[0]
        a = extra_id[:extra_train_size]
        selected_extra_train_index.append(a)
        b = extra_id[extra_train_size:extra_train_size+extra_test_size]
        selected_extra_test_index.append(b)
        c = extra_id[extra_train_size+extra_test_size:extra_train_size+extra_test_size+extra_val_size]
        selected_extra_val_index.append(c)
    extra_train_idx = np.concatenate(selected_extra_train_index)
    extra_test_idx = np.concatenate(selected_extra_test_index)
    extra_val_idx = np.concatenate(selected_extra_val_index)
    extra_xtrain = x_extra[:,:,:,extra_train_idx]
    extra_ytrain = y_extra[extra_train_idx,]
    extra_xtest = x_extra[:,:,:,extra_test_idx]
    extra_ytest =  y_extra[extra_test_idx,]
    extra_xval = x_extra[:,:,:,extra_val_idx]
    extra_yval = y_extra[extra_val_idx,]  
    extra_xtrain = np.transpose(extra_xtrain,(3,2,0,1)).astype(np.float32)
    extra_xtest = np.transpose(extra_xtest,(3,2,0,1)).astype(np.float32)
    extra_xval = np.transpose(extra_xval,(3,2,0,1)).astype(np.float32)
    
    selected_xtrain = np.vstack((selected_xtrain, corners_train, extra_xtrain))
    selected_xtest = np.vstack((selected_xtest, corners_test, extra_xtest))
    selected_xval = np.vstack((selected_xval,corners_val,extra_xval))
    selected_ytrain = np.hstack((selected_ytrain, corners_train_y, extra_ytrain))
    selected_ytest = np.hstack((selected_ytest, corners_test_y, extra_ytest))
    selected_yval = np.hstack((selected_yval,corners_val_y,extra_yval))
    print(selected_xtrain.shape, selected_xtest.shape,selected_xval.shape,selected_yval.shape, selected_ytrain.shape, selected_ytest.shape)
    print("y train distribution")
    print(np.unique(selected_ytrain, return_counts=True))
    print("y validation distribution")
    print(np.unique(selected_yval,return_counts=True))
    print("y test distribution")
    print(np.unique(selected_ytest, return_counts=True))
    tensor_x_train = torch.Tensor(selected_xtrain)
    tensor_y_train = torch.LongTensor(selected_ytrain)
    
    train_dataset = data.TensorDataset(tensor_x_train,tensor_y_train) 
    tensor_x_val = torch.Tensor(selected_xval)
    tensor_y_val = torch.LongTensor(selected_yval)
    val_dataset = data.TensorDataset(tensor_x_val,tensor_y_val)
    tensor_x_test = torch.Tensor(selected_xtest)
    tensor_y_test = torch.LongTensor(selected_ytest)
    test_dataset = data.TensorDataset(tensor_x_test,tensor_y_test)

     
    cuda = torch.cuda.is_available() 
    if tuning:
        print("----- start tuning parameters for simple CNN modeling ------")
        
        learning_rates = [0.001, 0.0003]
        batch_sizes = [128, 1024]
        
        max_val_accu = 0
        val_accu_list = []
        lr_list = []
        batch_list = []
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print("learning rate:", lr, "; batch size:", batch_size)
                train_params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 4}
        
                val_params = {'batch_size': batch_size,
                      'shuffle':False,
                      'num_workers':4}
                
                train_dataloader = data.DataLoader(train_dataset,**train_params)
                val_dataloader = data.DataLoader(val_dataset,**val_params)
                model = Conv5NN()
                if cuda:
                    model = model.cuda()
                curr_max_val_accu = 0
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr)
                for epoch in range(num_epochs):
                    print("----- epoch",epoch,"------")
                    train_loss = 0
                    train_corrects = 0
                    val_loss = 0
                    val_corrects = 0
                    model.train()
                    for inputs,labels in train_dataloader:
                        
                        optimizer.zero_grad()
                        if cuda:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * inputs.size(0)
                        train_corrects += torch.sum(torch.max(outputs,1)[1] == labels)
                    train_loss, train_accu = train_loss/len(train_dataloader.dataset), train_corrects.data.cpu().numpy()/len(train_dataloader.dataset)
                    print(train_loss, train_accu)
                    
                    model.eval()
                    for inputs,labels in val_dataloader:
                        
                        if cuda:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
    
                        val_loss += loss.item() * inputs.size(0)
                        val_corrects += torch.sum(torch.max(outputs,1)[1] == labels)
                    val_loss, val_accu = val_loss/len(val_dataloader.dataset), val_corrects.data.cpu().numpy()/len(val_dataloader.dataset)
                    print(val_loss, val_accu)
                    if curr_max_val_accu < val_accu:
                        curr_max_val_accu = val_accu
                        increase_perf_cnt = 0
                    else:
                        increase_perf_cnt += 1
                        if increase_perf_cnt >= increase_perf_cnt_cutoff:
                            print('At epoch', epoch, "stop training due to no increase val accu in last five epoch")
                            break
                lr_list.append(lr)
                batch_list.append(batch_size)
                val_accu_list.append(curr_max_val_accu)
                if max_val_accu < curr_max_val_accu:
                    max_val_accu = curr_max_val_accu
                    best_batch_size = batch_size
                    best_lr = lr
        print("---- tuning records -----")
        print("batch size list")
        print(batch_list)
        print("learning rate list")
        print(lr_list)
        print("validation accuracy list")
        print(val_accu_list)
        
    print("best parameters:")
    print("best learning rate", best_lr)
    print("best batch size", best_batch_size)
    train_params = {'batch_size': best_batch_size,
          'shuffle': True,
          'num_workers': 4}
    
    val_params = {'batch_size': best_batch_size,
                  'shuffle':False,
                  'num_workers':4}
    test_params = {'batch_size': best_batch_size,
          'shuffle': False,
          'num_workers': 4}
 
    train_dataloader = data.DataLoader(train_dataset,**train_params)
    
    val_dataloader = data.DataLoader(val_dataset,**val_params)
    
    test_dataloader = data.DataLoader(test_dataset,**test_params)
    print("----- start simple CNN modeling ------")
    model = Conv5NN()

    if cuda:
        model = model.cuda()    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=best_lr)

    max_val_accu = 0
    train_loss_list = []
    train_accu_list = []
    test_loss_list = []
    test_accu_list = []
    val_loss_list = []
    val_accu_list = []
    print("----- start training simple CNN ------")
    for epoch in range(num_epochs):
        print("----- epoch",epoch,"------")
        train_loss = 0
        train_corrects = 0
        test_loss = 0
        test_corrects = 0
        val_loss = 0
        val_corrects = 0
        model.train()
        for inputs,labels in train_dataloader:
            
            optimizer.zero_grad()
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(torch.max(outputs,1)[1] == labels)
        train_loss, train_accu = train_loss/len(train_dataloader.dataset), train_corrects.data.cpu().numpy()/len(train_dataloader.dataset)
        train_loss_list.append(train_loss)
        train_accu_list.append(train_accu)
        print(train_loss, train_accu)
        model.eval()
        for inputs,labels in val_dataloader:
            
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(torch.max(outputs,1)[1] == labels)
        val_loss, val_accu = val_loss/len(val_dataloader.dataset), val_corrects.data.cpu().numpy()/len(val_dataloader.dataset)
        val_loss_list.append(val_loss)
        val_accu_list.append(val_accu)
        if max_val_accu <  val_accu:
            increase_perf_cnt = 0
            max_val_accu = val_accu
            print("new higher val accu:", val_accu)
            torch.save(model.state_dict(),simple_cnn_saved_name)
        else:
            increase_perf_cnt += 1
            if increase_perf_cnt >= increase_perf_cnt_cutoff:
                print('At epoch', epoch, "stop training due to no increase val accu in last five epoch")
                break
        model.eval()
        for inputs,labels in test_dataloader:
            
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(torch.max(outputs,1)[1] == labels)
        test_loss, test_accu = test_loss/len(test_dataloader.dataset), test_corrects.data.cpu().numpy()/len(test_dataloader.dataset)
        test_loss_list.append(test_loss)
        test_accu_list.append(test_accu)        
        print(test_loss, test_accu)
        
    print("---train loss history----")
    print(train_loss_list)
    print("---validation loss history---")
    print(val_loss_list)
    print("---test loss history----")
    print(test_loss_list)
    print("---train accu history----")
    print(train_accu_list)
    print("---validation accu history---")
    print(val_accu_list)
    print("---test accu history----")
    print(test_accu_list)
    
    model = Conv5NN()
    model.load_state_dict(torch.load(simple_cnn_saved_name,map_location='cpu'))
    
    print("accuracy for each class in training")
    model.eval()
    digit_correct_num = {}
    digit_num = {}
    for inputs,labels in train_dataloader:
        outputs = model(inputs)
        preds = torch.max(outputs,1)[1].squeeze().cpu().numpy().astype(int)
        labels = labels.squeeze().cpu().numpy().astype(int)
        for i, label in enumerate(labels):
            pred = preds[i]
            if pred == label:
                if label not in digit_correct_num:
                    digit_correct_num[label] = 1
                else:
                    digit_correct_num[label] += 1
            if label not in digit_num:
                digit_num[label] = 1
            else:
                digit_num[label] += 1
    print(digit_num)
    print(digit_correct_num)
    for d in digit_num:
       print("accu for", d, "is ",(0 if d not in digit_correct_num else digit_correct_num[d])/digit_num[d])

    print("accuracy for each class in validation")
    digit_correct_num = {}
    digit_num = {}
    for inputs,labels in val_dataloader:
        outputs = model(inputs)
        preds = torch.max(outputs,1)[1].squeeze().cpu().numpy().astype(int)
        labels = labels.squeeze().cpu().numpy().astype(int)
        for i, label in enumerate(labels):
            pred = preds[i]
            if pred == label:
                if label not in digit_correct_num:
                    digit_correct_num[label] = 1
                else:
                    digit_correct_num[label] += 1
            if label not in digit_num:
                digit_num[label] = 1
            else:
                digit_num[label] += 1
    print(digit_num)
    print(digit_correct_num)
    for d in digit_num:
       print("accu for", d, "is ",(0 if d not in digit_correct_num else digit_correct_num[d])/digit_num[d])       
       
    print("accuracy for each class in testing")
    digit_correct_num = {}
    digit_num = {}
    for inputs,labels in test_dataloader:
        outputs = model(inputs)
        preds = torch.max(outputs,1)[1].squeeze().cpu().numpy().astype(int)
        labels = labels.squeeze().cpu().numpy().astype(int)
        for i, label in enumerate(labels):
            pred = preds[i]
            if pred == label:
                if label not in digit_correct_num:
                    digit_correct_num[label] = 1
                else:
                    digit_correct_num[label] += 1
            if label not in digit_num:
                digit_num[label] = 1
            else:
                digit_num[label] += 1
    print(digit_num)
    print(digit_correct_num)
    for d in digit_num:
       print("accu for", d, "is ",(0 if d not in digit_correct_num else digit_correct_num[d])/digit_num[d])
    

       
    print("----- start vgg16 modeling ------")
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier._modules['0'] = nn.Linear(25088, 128, bias=True)
    model.classifier._modules['3'] = nn.Linear(128, 32, bias=True)
    model.classifier._modules['6'] = nn.Linear(32, 11, bias=True)

    if cuda:
        model = model.cuda()    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=best_lr)

    max_val_accu = 0
    train_loss_list = []
    train_accu_list = []
    test_loss_list = []
    test_accu_list = []
    val_loss_list = []
    val_accu_list = []
    print("----- start training vgg16 ------")
    for epoch in range(num_epochs):
        print("----- epoch",epoch,"------")
        train_loss = 0
        train_corrects = 0
        test_loss = 0
        test_corrects = 0
        val_loss = 0
        val_corrects = 0
        model.train()
        for inputs,labels in train_dataloader:
            
            optimizer.zero_grad()
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(torch.max(outputs,1)[1] == labels)
        train_loss, train_accu = train_loss/len(train_dataloader.dataset), train_corrects.data.cpu().numpy()/len(train_dataloader.dataset)
        train_loss_list.append(train_loss)
        train_accu_list.append(train_accu)
        print(train_loss, train_accu)
        model.eval()
        for inputs,labels in val_dataloader:
            
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(torch.max(outputs,1)[1] == labels)
        val_loss, val_accu = val_loss/len(val_dataloader.dataset), val_corrects.data.cpu().numpy()/len(val_dataloader.dataset)
        val_loss_list.append(val_loss)
        val_accu_list.append(val_accu) 
        if max_val_accu <  val_accu:
            increase_perf_cnt = 0
            max_val_accu = val_accu
            print("new higher val accu:", val_accu)
            torch.save(model.state_dict(),vgg_saved_name)
        else:
            increase_perf_cnt += 1
            if increase_perf_cnt >= increase_perf_cnt_cutoff:
                print('At epoch', epoch, "stop training due to no increase val accu in last five epoch")
                break
        model.eval()
        for inputs,labels in test_dataloader:
            
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(torch.max(outputs,1)[1] == labels)
        test_loss, test_accu = test_loss/len(test_dataloader.dataset), test_corrects.data.cpu().numpy()/len(test_dataloader.dataset)
        test_loss_list.append(test_loss)
        test_accu_list.append(test_accu)        
        print(test_loss, test_accu)

    print("---train loss history----")
    print(train_loss_list)
    print("---validation loss history---")
    print(val_loss_list)
    print("---test loss history----")
    print(test_loss_list)
    print("---train accu history----")
    print(train_accu_list)
    print("---validation accu history---")
    print(val_accu_list)
    print("---test accu history----")
    print(test_accu_list)
    model = torchvision.models.vgg16()
    model.classifier._modules['0'] = nn.Linear(25088, 128, bias=True)
    model.classifier._modules['3'] = nn.Linear(128, 32, bias=True)
    model.classifier._modules['6'] = nn.Linear(32, 11, bias=True)
    model.load_state_dict(torch.load(vgg_saved_name,map_location='cpu'))
    print("accuracy for each class in training")
    model.eval()
    digit_correct_num = {}
    digit_num = {}
    for inputs,labels in train_dataloader:
        outputs = model(inputs)
        preds = torch.max(outputs,1)[1].squeeze().cpu().numpy().astype(int)
        labels = labels.squeeze().cpu().numpy().astype(int)
        for i, label in enumerate(labels):
            pred = preds[i]
            if pred == label:
                if label not in digit_correct_num:
                    digit_correct_num[label] = 1
                else:
                    digit_correct_num[label] += 1
            if label not in digit_num:
                digit_num[label] = 1
            else:
                digit_num[label] += 1
    print(digit_num)
    print(digit_correct_num)
    for d in digit_num:
       print("accu for", d, "is ",(0 if d not in digit_correct_num else digit_correct_num[d])/digit_num[d])

    print("accuracy for each class in validation")
    digit_correct_num = {}
    digit_num = {}
    for inputs,labels in val_dataloader:
        outputs = model(inputs)
        preds = torch.max(outputs,1)[1].squeeze().cpu().numpy().astype(int)
        labels = labels.squeeze().cpu().numpy().astype(int)
        for i, label in enumerate(labels):
            pred = preds[i]
            if pred == label:
                if label not in digit_correct_num:
                    digit_correct_num[label] = 1
                else:
                    digit_correct_num[label] += 1
            if label not in digit_num:
                digit_num[label] = 1
            else:
                digit_num[label] += 1
    print(digit_num)
    print(digit_correct_num)
    for d in digit_num:
       print("accu for", d, "is ",(0 if d not in digit_correct_num else digit_correct_num[d])/digit_num[d])       
              
       
    print("accuracy for each class in testing")
    digit_correct_num = {}
    digit_num = {}
    for inputs,labels in test_dataloader:
        outputs = model(inputs)
        preds = torch.max(outputs,1)[1].squeeze().cpu().numpy().astype(int)
        labels = labels.squeeze().cpu().numpy().astype(int)
        for i, label in enumerate(labels):
            pred = preds[i]
            if pred == label:
                if label not in digit_correct_num:
                    digit_correct_num[label] = 1
                else:
                    digit_correct_num[label] += 1
            if label not in digit_num:
                digit_num[label] = 1
            else:
                digit_num[label] += 1
    print(digit_num)
    print(digit_correct_num)
    for d in digit_num:
       print("accu for", d, "is ",(0 if d not in digit_correct_num else digit_correct_num[d])/digit_num[d])
       
if __name__ == '__main__':    
    cnn(train_size = 3500,
        val_size = 750,
        test_size = 750,
        extra_train_size = 7000,
        extra_test_size = 1500,
        extra_val_size = 1500,
        non_digit_size = 15000,
        increase_perf_cnt_cutoff = 5,
        num_epochs = 100,
        tuning = True,
        default_lr = 0.001,
        default_batch_size = 128,
        simple_cnn_saved_name = 'svhn_11_cnn_tuned.pth',
        vgg_saved_name = 'svhn_11_vgg16_tuned.pth')
    
    print("CNN modeling experiment is done")

