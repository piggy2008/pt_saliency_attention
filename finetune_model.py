import torch
import torch.nn as nn
import torch.nn.functional as F
from image_data_loader import ImageDataPretrain, ImageSeqData
import time
from models_pspnet import PSPNet
from models_base.model_dss import Model_dss
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models_base import ModelBuilder, SegmentationModule
from tools.training_tools import adjust_learning_rate, create_optimizers, adjust_learning_rate2
from tools.utils import load_part_of_model_dss
from gluoncvth.models import get_deeplab_resnet101_mine

def finetune_resnet101_deeplab(root_dir, list_file_path):
    param = {}
    param['lr'] = 0.00001
    param['momentum'] = 0.9
    param['beta1'] = 0.9
    param['weight_decay'] = 1e-4
    param['max_iters'] = 10000
    param['lr_pow'] = 0.9
    param['total_iters'] = 50001
    param['save_iters'] = 10000

    param['running_lr'] = param['lr']
    param['crop_size'] = 512

    param['pretrained_model'] = 'model/2018-12-21 10:20:40/30000/snap_model.pth'

    image_names = [line.strip() for line in list_file_path]
    # (self, root_dir, image_names, image_size, crop_size, batch_size, seq_size, horizontal_flip=False):
    # dataset = ImageDataPretrain(root_dir, image_names, 560, 512, 5, horizontal_flip=True)
    dataset = ImageSeqData(root_dir, root_dir + '_gt2_revised', image_names,
                           '.jpg', '.png', 580, param['crop_size'], 1, 5, horizontal_flip=False)

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_loss = []
    log_loss.append('feature extractor: resnet101 deeplab \n')
    log_loss.append('image size:530 \n')
    log_loss.append('crop size:512 \n')
    # log_loss.append('freeze layers except after conv5 \n')
    # log_loss.append('only smoothl1 loss \n')
    log_loss.append('only sigmoid cross entropy loss \n')
    log_loss.append('base network pspnet \n')
    log_loss.append('pre-trained model: ' + param['pretrained_model'] + '\n')
    log_loss.append('start lr: 0.0001\n')

    device = torch.device('cuda')

    model = get_deeplab_resnet101_mine(pretrained=True).to(device)
    model.load_state_dict(torch.load(param['pretrained_model']))
    crit = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])

    model.train()
    for itr in range(param['total_iters']):
        x, y = dataset.next_batch()

        # y = y.reshape([y.shape[0], y.shape[2], y.shape[3]])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        # y = y.type(torch.cuda.LongTensor)

        # one-hot mask transfer
        # y_one_hot = torch.cuda.LongTensor(y.size(0), 2, y.size(2), y.size(3))
        # target = y_one_hot.scatter_(1, y.data, 1)

        final_saliency = model(x)
        train_loss = crit(final_saliency[0], y)
        # saliency = final_saliency.data.cpu().numpy()
        # final_saliency, cap_feats, local_pos, cap_feats2 = model(x, x_prior)

        # loss = train_loss1
        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        # train_loss2.backward()

        if itr % 5 == 0:
            log = 'step: %d, lr: %g, train_loss1:%g' % (itr, param['running_lr'], train_loss)
            print(log)
            log_loss.append(log)
            # summary_writer.add_summary(summary_str, itr)

        if itr % param['save_iters'] == 0:
            save_model(model, str(itr), time_str, log_loss)
            del log_loss[:]

        adjust_learning_rate2(optimizer, itr, param)

    print('finish training time:', str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

def finetune_resnet50_dilated8_ppm_bilinear(root_dir, list_file_path):
    param = {}
    param['lr_encode'] = 0.0001
    param['lr_decode'] = 0.0001
    param['momentum'] = 0.9
    param['beta1'] = 0.9
    param['weight_decay'] = 1e-4
    param['max_iters'] = 10000
    param['lr_pow'] = 0.9
    param['total_iters'] = 50001
    param['save_iters'] = 10000

    param['running_lr_encoder'] = param['lr_encode']
    param['running_lr_decoder'] = param['lr_decode']

    image_names = [line.strip() for line in list_file_path]
    # (self, root_dir, image_names, image_size, crop_size, batch_size, seq_size, horizontal_flip=False):
    dataset = ImageDataPretrain(root_dir, image_names, 530, 512, 5, horizontal_flip=True)

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_loss = []
    log_loss.append('feature extractor: resnet50_dilated8_ppm_bilinear \n')
    log_loss.append('image size:530 \n')
    log_loss.append('crop size:512 \n')
    # log_loss.append('freeze layers except after conv5 \n')
    # log_loss.append('only smoothl1 loss \n')
    log_loss.append('only sigmoid cross entropy loss \n')
    log_loss.append('base network pspnet \n')
    log_loss.append('pre-trained model: model/2018-10-19 18:55:44/200000/snap_model.pth \n')
    log_loss.append('start lr: 0.0001\n')

    device = torch.device('cuda')

    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch='resnet50_dilated8',
        fc_dim=512,
        weights='weights_base/encoder_epoch_20.pth')
    net_decoder = builder.build_decoder(
        arch='ppm_bilinear_deepsup',
        fc_dim=2048,
        num_class=1,
        weights='weights_base/decoder_epoch_20.pth',
        use_softmax=False)

    crit = nn.BCEWithLogitsLoss()
    # crit = nn.NLLLoss()
    nets = (net_encoder, net_decoder, crit)

    model = SegmentationModule(net_encoder, net_decoder, crit, deep_sup_scale=0.4).to(device)
    model.load_state_dict(torch.load('model/2018-10-19 18:55:44/200000/snap_model.pth'))

    optimizers = create_optimizers(nets, param)

    model.train()
    for itr in range(param['total_iters']):
        x, y = dataset.next_batch()

        # y = y.reshape([y.shape[0], y.shape[2], y.shape[3]])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        # y = y.type(torch.cuda.LongTensor)

        # one-hot mask transfer
        # y_one_hot = torch.cuda.LongTensor(y.size(0), 2, y.size(2), y.size(3))
        # target = y_one_hot.scatter_(1, y.data, 1)

        loss = model(x, y)
        # saliency = final_saliency.data.cpu().numpy()
        # final_saliency, cap_feats, local_pos, cap_feats2 = model(x, x_prior)

        # loss = train_loss1
        model.zero_grad()
        loss.backward()
        # train_loss2.backward()
        for optimizer in optimizers:
            optimizer.step()

        if itr % 5 == 0:
            log = 'step: %d, lr_encode: %g, lr_decode: %g, train_loss1:%g' % (itr, param['running_lr_encoder'], param['running_lr_decoder'], loss)
            print(log)
            log_loss.append(log)
            # summary_writer.add_summary(summary_str, itr)

        if itr % param['save_iters'] == 0:
            save_model(model, str(itr), time_str, log_loss)
            del log_loss[:]

        adjust_learning_rate(optimizers, itr, param)

    print('finish training time:', str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

def finetune_PSPNetBase(root_dir, list_file_path, model_base='resnet50'):
    # root_dir = '/home/ty/data/Pre-train'
    # list_file = open('/home/ty/data/Pre-train/pretrain_all_seq.txt')
    image_names = [line.strip() for line in list_file_path]
    # (self, root_dir, image_names, image_size, crop_size, batch_size, seq_size, horizontal_flip=False):
    dataset = ImageDataPretrain(root_dir, image_names, 550, 512, 5, horizontal_flip=True)


    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_loss = []
    log_loss.append('feature extractor:'+ model_base + '\n')
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
    if model_base == 'dpn':
        model = PSPNet(n_classes=1, psp_size=832, backend='dpn68_warp', pretrained=False).to(device)
    else:
        model = PSPNet(n_classes=1, backend='resnet50', pretrained=False).to(device)
    # training smooth l1 loss
    # criterion_regression = nn.SmoothL1Loss()
    model.load_state_dict(torch.load('model/2018-10-17 19:57:48/50000/snap_model.pth'))

    # training sigmoid cross entropy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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
        if model_base == 'dpn':
            final_saliency = F.upsample(input=final_saliency, size=(512, 512), mode='bilinear')
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


        if itr % 10000 == 0:
            save_model(model, str(itr), time_str, log_loss)
            del log_loss[:]

    print('finish training time:', str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

def finetune_DSS(root_dir, list_file_path):
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

    model = Model_dss().to(device)
    model = load_part_of_model_dss(model, 'pretrained_models/dss_model_params.mat')

    # training smooth l1 loss
    # criterion_regression = nn.SmoothL1Loss()
    # model.load_state_dict(torch.load('model/2018-10-17 19:57:48/50000/snap_model.pth'))

    # training sigmoid cross entropy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

        score_list = model(x)
        train_loss1 = 0
        for score in score_list:
            train_loss1 += criterion(score, y)

        loss = train_loss1
        model.zero_grad()
        loss.backward()
        # train_loss2.backward()
        optimizer.step()

        if itr % 5 == 0:
            print('step: %d, train_loss1:%g' % (itr, train_loss1))
            log_loss.append('step: %d, train_loss1:%g' % (itr, train_loss1))
            # summary_writer.add_summary(summary_str, itr)

        if itr % 10000 == 0:
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

    # root_dir = '/home/ty/data/Pre-train'
    # list_file_path = open('/home/ty/data/Pre-train/pretrain_all_seq.txt')

    root_dir = '/home/ty/data/video_saliency/train_all'
    # prior_dir = '/home/ty/data/video_saliency/train_all_prior'
    # list_file_path = open('/home/ty/data/video_saliency/train_all_seq_step_1.txt')
    list_file_path = open('/home/ty/data/video_saliency/train_all_seq_5f.txt')
    # list_file_path = open('/home/ty/data/video_saliency/train_all_single_frame.txt')
    finetune_resnet101_deeplab(root_dir, list_file_path)