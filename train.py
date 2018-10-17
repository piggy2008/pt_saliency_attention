import torch
import torch.nn as nn
import torch.nn.functional as F
from image_data_loader import ImageAndPriorSeqData, ImageAndPriorSeqBboxData, ImageDataPretrain
import time
from models import VideoSaliency
from models_pspnet import PSPNet
from utils import load_weights_from_h5, load_part_of_model, freeze_some_layers
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from matplotlib import pyplot as plt
import numpy as np


def train(train_dir, label_dir, prior_dir, list_file_path):
    list_file = open(list_file_path)
    image_names = [line.strip() for line in list_file]

    # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
    dataset = ImageAndPriorSeqBboxData(train_dir, label_dir, prior_dir, None, None,
                                   None,
                                   image_names, None, '.jpg', '.png', 430, 400, 1,
                                   4,
                                   horizontal_flip=False, pulishment=False)

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_loss = []

    log_loss.append('image size:430 \n')
    log_loss.append('crop size:400 \n')
    # log_loss.append('freeze layers except after conv5 \n')
    log_loss.append('only smoothl1 loss \n')
    log_loss.append('only sigmoid cross entropy loss \n')
    log_loss.append('No bbox publishment \n')
    log_loss.append('gaussian sigma:0.45 \n')
    # log_loss.append(self.prior_type + '\n')
    device = torch.device('cuda')
    model = VideoSaliency().to(device)
    model.load_state_dict(torch.load('model/2018-08-29 09:56:15/6000/snap_model.pth'))
    # model.load_state_dict(torch.load('model/2018-08-31 09:32:41/8000/snap_model.pth'))

    # model = load_part_of_model(model, 'model/2018-08-29 09:56:15/10000/snap_model.pth')
    # model.load_state_dict(torch.load('model/2018-08-26 10:20:23/6000/snap_model.pth'))
    # model.load_state_dict(torch.load('model/2018-08-26 17:11:03/6000/snap_model.pth'))
    # model = load_part_of_model(model, 'model/2018-08-22 18:04:08/6000/snap_model.pth')
    # model = freeze_some_layers(model)

    # training smooth l1 loss
    criterion_regression = nn.SmoothL1Loss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


    # model.load_state_dict(torch.load('model/2018-08-23 13:02:15/6000/snap_model.pth'))
    # training sigmoid cross entropy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


    # load model from h5 file
    # h5_path = '/home/ty/code/tf_saliency_attention/mat_parameter/fusionST_parameter_ms.mat'
    # load_weights_from_h5(model, h5_path)

    # load model from a pre-trained pytorch pth


    model.train()
    for itr in range(4001):
        x, y, bbox = dataset.next_batch()
        # feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y}
        x_prior = torch.from_numpy(x)
        x = torch.from_numpy(x[:, :3, :, :])
        y = torch.from_numpy(y)
        bbox = torch.from_numpy(bbox)

        x = x.type(torch.cuda.FloatTensor)
        x_prior = x_prior.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        bbox = bbox.type(torch.cuda.FloatTensor)

        final_saliency, cap_feats, local_pos = model(x, x_prior)
        # final_saliency, cap_feats, local_pos, cap_feats2 = model(x, x_prior)
        train_loss1 = criterion(final_saliency, y)
        train_loss2 = criterion_regression(local_pos, bbox)

        # print(local_pos)
        # print(bbox)
        # final_saliency = F.sigmoid(final_saliency)
        # final_saliency = final_saliency.data.cpu().numpy()
        #
        # plt.subplot(2, 3, 1)
        # plt.imshow(final_saliency[3, 0, :, :])
        #
        # local_pos = local_pos.data.cpu().numpy() * 400
        # plt.subplot(2, 3, 2)
        # plt.imshow(final_saliency[3, 0, int(local_pos[3, 0]):int(local_pos[3, 2]), int(local_pos[3, 1]):int(local_pos[3, 3])])
        #
        # plt.subplot(2, 3, 3)
        # bbox = bbox.data.cpu().numpy() * 400
        # plt.imshow(final_saliency[3, 0, int(bbox[3, 0]):int(bbox[3, 2]), int(bbox[3, 1]):int(bbox[3, 3])])
        #
        # plt.subplot(2, 3, 4)
        # cap_feats = cap_feats.data.cpu().numpy()
        # plt.imshow(cap_feats[3, 0, :, :])
        #
        # plt.subplot(2, 3, 5)
        # cap_feats2 = cap_feats2.data.cpu().numpy()
        # plt.imshow(cap_feats2[3, 0, :, :])
        # plt.show()

        # loss = train_loss2
        loss = train_loss1 + train_loss2
        model.zero_grad()
        loss.backward()
        # train_loss2.backward()
        optimizer.step()


        if itr % 5 == 0:
            print('step: %d, train_loss1:%g' % (itr, train_loss1))
            log_loss.append('step: %d, train_loss1:%g' % (itr, train_loss1))
            # summary_writer.add_summary(summary_str, itr)
            print('step: %d, train_loss2:%g' % (itr, train_loss2))
            log_loss.append('step: %d, train_loss2:%g' % (itr, train_loss2))


        if itr % 2000 == 0:
            save_model(model, str(itr), time_str, log_loss)
            del log_loss[:]

def train_PSPNetBase(root_dir, list_file_path):
    # root_dir = '/home/ty/data/Pre-train'
    # list_file = open('/home/ty/data/Pre-train/pretrain_all_seq.txt')
    image_names = [line.strip() for line in list_file_path]
    # (self, root_dir, image_names, image_size, crop_size, batch_size, seq_size, horizontal_flip=False):
    dataset = ImageDataPretrain(root_dir, image_names, 550, 512, 5, horizontal_flip=True)


    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_loss = []

    log_loss.append('image size:550 \n')
    log_loss.append('crop size:512 \n')
    # log_loss.append('freeze layers except after conv5 \n')
    # log_loss.append('only smoothl1 loss \n')
    log_loss.append('only sigmoid cross entropy loss \n')
    log_loss.append('base network pspnet \n')
    log_loss.append('pre-trained model, resnet no 1024c block \n')
    # log_loss.append('pre-trained model, resnet no 1024c block \n')
    # log_loss.append('No bbox publishment \n')
    # log_loss.append('gaussian sigma:0.45 \n')
    # log_loss.append(self.prior_type + '\n')
    device = torch.device('cuda')
    model = PSPNet(n_classes=1, backend='resnet50').to(device)
    # training smooth l1 loss
    # criterion_regression = nn.SmoothL1Loss()

    # training sigmoid cross entropy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


    # load model from h5 file
    # h5_path = '/home/ty/code/tf_saliency_attention/mat_parameter/fusionST_parameter_ms.mat'
    # load_weights_from_h5(model, h5_path)

    # load model from a pre-trained pytorch pth


    model.train()
    for itr in range(50001):
        x, y = dataset.next_batch()

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)

        final_saliency = model(x)
        # saliency = final_saliency.data.cpu().numpy()
        # final_saliency, cap_feats, local_pos, cap_feats2 = model(x, x_prior)
        train_loss1 = criterion(final_saliency, y)

        loss = train_loss1
        model.zero_grad()
        loss.backward()
        # train_loss2.backward()
        optimizer.step()


        if itr % 5 == 0:
            print('step: %d, train_loss1:%g' % (itr, train_loss1))
            log_loss.append('step: %d, train_loss1:%g' % (itr, train_loss1))
            # summary_writer.add_summary(summary_str, itr)


        if itr % 5000 == 0:
            save_model(model, str(itr), time_str, log_loss)
            del log_loss[:]

    print('finish training time:', str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))


def save_model(model, itr, network_name, log_list=[]):
    model_dir = os.path.join('model', network_name, itr)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print('save model:', os.path.join(model_dir, 'snap_model.pth'))
    # self.saver.save(self.sess, os.path.join(model_dir, 'snap_model.ckpt'))
    torch.save(model.state_dict(), os.path.join(model_dir, 'snap_model.pth'))

    if log_list:
        log_file = open(os.path.join('model', network_name, 'log_loss.txt'), 'a')
        for log in log_list:
            log_file.writelines(log + '\n')

        log_file.flush()
        log_file.close()

if __name__ == '__main__':
    # image_dir = '/home/ty/data/video_saliency/train_all'
    # label_dir = '/home/ty/data/video_saliency/train_all_gt2_revised'
    # prior_dir = '/home/ty/data/video_saliency/train_all_prior'
    # # list_file_path = '/home/ty/data/video_saliency/train_all_seq.txt'
    #
    # list_file_path = '/home/ty/data/video_saliency/train_all_seq_bbx.txt'
    # train(image_dir, label_dir, prior_dir, list_file_path)

    root_dir = '/home/ty/data/Pre-train'
    list_file_path = open('/home/ty/data/Pre-train/pretrain_all_seq.txt')
    train_PSPNetBase(root_dir, list_file_path)