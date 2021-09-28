#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:21:33 2020

@author: qinfang shen
"""

import torch
import numpy as np
import torch.nn as nn

import torchvision

import os
import cv2





OUTPUT_DIR = "graded_images"
vgg_saved_name = 'svhn_11_vgg16_tuned.pth'
model = torchvision.models.vgg16()
model.classifier._modules['0'] = nn.Linear(25088, 128, bias=True)
model.classifier._modules['3'] = nn.Linear(128, 32, bias=True)
model.classifier._modules['6'] = nn.Linear(32, 11, bias=True)
model.load_state_dict(torch.load(vgg_saved_name,map_location='cpu'))
model.eval()

def extract_roi_pyramids(img,delta=25, min_area=300, max_area=2000,scales=[1.3,1,0.7]):
    ih,iw = img.shape[:2]
    mser = cv2.MSER() 
    mser_inst = mser.create(_delta=delta,_min_area=min_area,_max_area=max_area)
    area,boxes = mser_inst.detectRegions(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

    bb_set = set()
    for box in boxes:
        x, y, w, h = box
        max_hw = max(w,h)
        x_center, y_center = int(x+0.5*w), int(y+0.5*h)
        for scale in scales:
            sw = int(max_hw * scale)
            x1 = max(0,int(x_center - 0.5 * sw))
            y1 = max(0,int(y_center - 0.5 * sw))
            x2 = min(iw,int(x_center + 0.5 * sw))
            y2 = min(ih,int(y_center + 0.5 * sw))
            if (x1,y1,x2,y2) not in bb_set:
                bb_set.add((x1,y1,x2,y2))

            x1 = max(0,int(x_center - 0.5 * sw * 0.75))
            y1 = max(0,int(y_center - 0.5 * sw * 1.5))
            x2 = min(iw,int(x_center + 0.5 * sw * 0.75))
            y2 = min(ih,int(y_center + 0.5 * sw * 1.5))
            if (x1,y1,x2,y2) not in bb_set:
                bb_set.add((x1,y1,x2,y2))   

    return np.array(list(bb_set))  

def digit_detect_process(image,
                         delta=25,
                         min_area=300,
                         max_area=2000,
                         pyramid_scales=[1.3,1],
                         iou_threshold=0.3,
                         fontsize=2,
                         blur=False,
                         orientation='h'):
    img = image.copy()
    if blur:
        img = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1)
    bb_roi_pyramids = extract_roi_pyramids(img,delta=delta, min_area=min_area, max_area=max_area,scales=pyramid_scales)
    if len(bb_roi_pyramids) < 1:
        return
    img_list = []
    for x1,y1,x2,y2 in bb_roi_pyramids:

        img2 = img[y1:y2,x1:x2,:]
        img2 = cv2.resize(img2,(32,32))
        img_list.append(img2)
    imgs = np.transpose(np.stack(img_list),(0,3,1,2)).astype(np.float32)/255.0

    img_t = torch.Tensor(imgs)
    outputs = torch.nn.Softmax(dim=1)(model(img_t.view(-1, 3, 32,32))).cpu().detach().numpy()
    preds = np.argmax(outputs,axis=1)
    nms_ids = torchvision.ops.nms(torch.Tensor(bb_roi_pyramids),torch.Tensor(np.max(outputs,axis=1)),iou_threshold).numpy()
    res_img = image.copy()
    for i in nms_ids:
        if preds[i] > 0:
            x1,y1,x2,y2 = bb_roi_pyramids[i]
            if orientation == 'h':
                cv2.putText(res_img,str(preds[i]%10), (x1+10,y1), cv2.FONT_HERSHEY_SIMPLEX,  
                       fontsize, (255,50,200), 4, cv2.LINE_AA)
            else:
                cv2.putText(res_img,str(preds[i]%10), (x2+10,int((y1+y2)*0.5)), cv2.FONT_HERSHEY_SIMPLEX,  
                       fontsize, (255,50,200), 4, cv2.LINE_AA)
            cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 255, 0), fontsize)
    return res_img       
            
            
if __name__ == '__main__':
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    img1 = cv2.imread("input1.png")
    img1_res = digit_detect_process(img1)
    cv2.imwrite(OUTPUT_DIR +"/1.png",img1_res)
    #print("--- 1.png has been processed---")
     
    img2 = cv2.imread("input2.png")
    img2_res = digit_detect_process(img2,delta=1,
                     min_area=1000,
                     max_area=2000,
                     pyramid_scales=[1.7,1.4],
                     iou_threshold=0.2,
                     fontsize=3,orientation='v')
    cv2.imwrite(OUTPUT_DIR +"/2.png",img2_res)
    #print("--- 2.png has been processed---")   
           
    img3 = cv2.imread("input3.png")
    img3_res = digit_detect_process(img3,delta=8,
                     min_area=50, 
                     max_area=200,
                     pyramid_scales=[2,1.5,1],
                     iou_threshold=0.2)
    cv2.imwrite(OUTPUT_DIR +"/3.png",img3_res)           
    #print("--- 3.png has been processed---")  


    img4 = cv2.imread("input4.png")
    img4_res = digit_detect_process(img4,delta=15,
                     min_area=700, 
                     max_area=1000,
                     pyramid_scales=[1.5,1.3],
                     iou_threshold=0.2,
                     fontsize=2,
                     orientation='v')
    cv2.imwrite(OUTPUT_DIR +"/4.png",img4_res) 
    #print("--- 4.png has been processed---")

    img5 = cv2.imread("input5.png")
    img5_res = digit_detect_process(img5,delta=15,
                     min_area=400, 
                     max_area=10000,
                     pyramid_scales=[1.3],
                     iou_threshold=0.2,
                     fontsize=5)
    cv2.imwrite(OUTPUT_DIR +"/5.png",img5_res) 
    #print("--- 5.png has been processed---")




