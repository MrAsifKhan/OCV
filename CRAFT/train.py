import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import h5py
import re
import water
#from test import test

import multiprocessing as mp
import torch

from math import exp
from dataset import FPC_dataset

from mseloss import Maploss
from validation import get_validation_metrics


from collections import OrderedDict
import craft_utils
from torch.optim.lr_scheduler import StepLR

from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool
from  ocr_fpc import DataExtraction as DataEx
import configuration as cf

parser_train = argparse.ArgumentParser(description='CRAFT Detection Training')

parser_train.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser_train.add_argument('--epochs', default=30, type = int,
                    help='number of epochs to use for trianing')                    
parser_train.add_argument('--batch_size', default=1, type = int,
                    help='batch size of training')
parser_train.add_argument('--lr', default=0.0001, type=float,
                    help='initial learning rate')              
parser_train.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser_train.add_argument('--weight_decay', default=5e-6, type=float,
                    help='Weight decay for SGD')
parser_train.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser_train.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser_train.add_argument('--validation_intervel', default=5, type=int,
                    help='number of epoches after which the validation has to be performed')
parser_train.add_argument('--iou_threshold', default=0.70, type=float,
                    help='threshold for detecting ious')
parser_train.add_argument('--text_threshold', default=0.6, type=float,
                    help='text confidence threshold')
parser_train.add_argument('--link_threshold', default=0.6, type=float,
                    help='link confidence threshold')
parser_train.add_argument('--low_text', default=0.40, type=float,
                    help='text low bound threshold')
parser_train.add_argument('--save_interval', default=20, type=float,
                    help='interval to save the model')
                                       
args = parser_train.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(42)
torch.cuda.empty_cache()
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** step)
    print("*Adjusted Learning rate : ",lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#scheduler = StepLR(optimizer, step_size=2, gamma = args.gamma)
if __name__ == '__main__':
    mp.set_start_method('spawn')
    split = 0.85
    net = CRAFT()
    if args.resume:
        net.load_state_dict(copyStateDict(torch.load(args.resume)))
        print(f'Resuming Training from checkpoint {args.resume} ....')
    else:
        net.load_state_dict(copyStateDict(torch.load('./pretrained/craft_mlt_25k.pth')))
    net = net.to(device)
    
    net = torch.nn.DataParallel(net).to(device)

    print(f"Starting training script for CRAFT using {device} and parameters \n{args}")
    train_df, validation_df = DataEx().get_df(split)
    print(f"Training with {len(train_df)} training images and {len(validation_df)} validation images ")
    
    train_dataloader = FPC_dataset(net, train_df, "train", target_size = 1024)
    train_loader = torch.utils.data.DataLoader(
        train_dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True)

    validation_dataloader = FPC_dataset(net, validation_df, "validation", target_size = 1024)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataloader,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        pin_memory=True)

    cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    criterion = Maploss()
    #criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    step_index = 0
    loss_time  = 0
    best_val   = None
    loss_value = 0
    compare_loss = 1
    best_IOU = 0.4
    for epoch in range(args.epochs):
        train_time_st = time.time()
        loss_value = 0
        
        st = time.time()

        for index, (real_images, real_gh_label, real_gah_label, real_mask, _) in enumerate(train_loader):
            net.train()
            images = real_images
            gh_label = real_gh_label
            gah_label =  real_gah_label
            mask = real_mask

            images = Variable(images.type(torch.FloatTensor)).to(device)
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).to(device)
            gah_label = Variable(gah_label).to(device)
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).to(device)
            optimizer.zero_grad()
            out, _ = net(images)
            out1 = out[:, :, :, 0].to(device)
            out2 = out[:, :, :, 1].to(device)
            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()
            
            loss_value += loss.item()
            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch | training time for 2 batch {} | training loss {} '.format(epoch+1, index, len(train_loader), round(et-st,2), round((loss_value/2),15)))
                loss_time = 0
                loss_value = 0
                st = time.time()
            net.eval()
        if epoch == 0 or (epoch+1) % args.validation_intervel == 0 :
            print("Starting Validation...")
            validation_image_precisions = []
            validation_image_recall = []
            validation_image_hmean  = []
            skipped_gt = 0
            failed = 0
            path = './output/results/'
            for index, (images, gt_boxes) in enumerate(validation_loader):    #validation 
                images = Variable(images.type(torch.FloatTensor)).to(device)
                out, _ = net(images)
                score_text = out[0,:,:,0].cpu().data.numpy() 
                score_link = out[0,:,:,1].cpu().data.numpy()
                min_text = args.low_text
                difference = 0
                gt_len=len(gt_boxes)
                predicted_boxes, polys = craft_utils.getDetBoxes(score_text, score_link, args.text_threshold, args.link_threshold, min_text)   
                pred_len = len(predicted_boxes)
                if pred_len ==0:
                    val = 0.00
                    validation_image_precisions.append(val)
                    validation_image_recall.append(val)
                    validation_image_hmean.append(val)
                    skipped_gt += gt_len
                    failed+=1
                    print(f'failed to predict on index {index} image')
                    continue
                precision, recall, hmean, IOU_mean = craft_utils.calculate_validation_metrics(gt_boxes, predicted_boxes, iou_threshold = args.iou_threshold) #example of bounding boxes: 203,3,250,3,250,29,203,29
                validation_image_precisions.append(precision)
                validation_image_recall.append(recall)
                validation_image_hmean.append(hmean)
            validation_precision = np.mean(validation_image_precisions)
            validation_recall = np.mean(validation_image_recall)
            validation_hmean = np.mean(validation_image_hmean)
            print(f'Validation Metrics \t Precision:{validation_precision} \t Recall:{validation_recall} \t Hmean:{validation_hmean} \t IOU:{IOU_mean} with low_text {min_text}')
            if not best_val:
                best_val = validation_hmean
                print('Saving model')
                torch.save(net.state_dict(), (path + 'CRAFT_Detection_best_validation.pth'))
            if validation_hmean > best_val:
                print(f'Saving model with improved hmean value of {validation_hmean}')
                best_val = validation_hmean
                torch.save(net.state_dict(), (path + 'CRAFT_Detection_best_validation.pth'))
            if IOU_mean> best_IOU:
                print(f'Saving model with improved IOU value of {IOU_mean}')
                best_IOU = IOU_mean
                torch.save(net.state_dict(), (path + 'CRAFT_Detection_best_IOU.pth'))
        if (epoch+1) % args.save_interval == 0:
            print(f'Saving model checkpoint at epoch {epoch+1}')
            location = path + "epoch" + str(epoch+1) +'.pth'
            torch.save(net.state_dict(), location)
        if epoch % 10 == 0 and epoch != 0:
            print("*adjusting learning rate*")
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        print("-"*100)









