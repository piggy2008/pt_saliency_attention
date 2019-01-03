import torch
import torch.nn as nn
import torch.nn.functional as F
from image_data_loader import ImageDataPretrain, ImageSeqData, ImageAndPriorSeqData, ImageAndPriorSeqCenterPData
import time
from models_pspnet import PSPNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models_base.models import ModelBuilder, SegmentationModule, SegmentationBboxModule
from tools.training_tools import adjust_learning_rate, create_optimizers, adjust_learning_rate2
from tools.utils import *

def finetune_resnet50_dilated8_ppm_bilinear(root_dir, list_file_path):
    param = {}
    param['lr_encode'] = 0.0001
    param['lr_decode'] = 0.0001
    param['momentum'] = 0.95
    param['beta1'] = 0.9
    param['weight_decay'] = 1e-4
    param['max_iters'] = 20000
    param['lr_pow'] = 0.9
    param['total_iters'] = 40001
    param['save_iters'] = 10000
    param['crop_size'] = 400

    param['running_lr_encoder'] = param['lr_encode']
    param['running_lr_decoder'] = param['lr_decode']

    # param['pretrained_model'] = 'model/2018-11-11 12:51:58/30000/snap_model.pth'
    # param['pretrained_model'] = 'model/2018-11-15 21:39:15/30000/snap_model.pth' #best
    param['pretrained_model'] = 'model/2018-12-28 15:00:08/30000/snap_model.pth'

    image_names = [line.strip() for line in list_file_path]
    # (self, root_dir, image_names, image_size, crop_size, batch_size, seq_size, horizontal_flip=False):
    # dataset = ImageSeqData(root_dir, root_dir + '_gt2_revised', image_names,
    #                        '.jpg', '.png', 512, param['crop_size'], 1, 5, horizontal_flip=False)

    # dataset = ImageDataPretrain(root_dir, image_names, 420, param['crop_size'], 5, horizontal_flip=False)
    prior_dir = '/home/ty/data/video_saliency/train_all_prior'
    dataset = ImageAndPriorSeqCenterPData(root_dir, root_dir + '_gt2_revised', prior_dir, image_names,
                                          '.jpg', '.png', 450, param['crop_size'], 1, 5, horizontal_flip=False)

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_loss = []
    # log_loss.append('warming up with single image data \n')
    log_loss.append('training with video data \n')
    log_loss.append('feature extractor: resnet50_dilated8_ppm_bilinear \n')
    log_loss.append('image size:530 \n')
    log_loss.append('crop size:512 \n')
    # log_loss.append('freeze layers except after conv5 \n')
    # log_loss.append('only smoothl1 loss \n')
    log_loss.append('only sigmoid cross entropy loss \n')
    log_loss.append('base network pspnet \n')
    log_loss.append('pre-trained model: ' + param['pretrained_model'] + '\n')
    log_loss.append('start lr: ' + str(param['lr_encode']) + '\n')
    log_loss.append('momentum: ' + str(param['momentum']) + '\n')

    device = torch.device('cuda')

    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch='resnet50_dilated8',
        fc_dim=512,
        # weights='weights_base/encoder_epoch_20.pth'
    )
    # net_decoder = builder.build_decoder(
    #     arch='ppm_bilinear_deepsup',
    #     fc_dim=2048,
    #     num_class=1,
    #     # weights='weights_base/decoder_epoch_20.pth',
    #     use_softmax=False)
    net_decoder = builder.build_decoder(
        arch='ppm_bilinear_local',
        fc_dim=2048,
        num_class=1,
        # weights='weights_base/decoder_epoch_20.pth',
        use_softmax=False)

    crit = nn.BCEWithLogitsLoss()
    # crit = nn.NLLLoss()
    nets = (net_encoder, net_decoder, crit)

    model = SegmentationBboxModule(net_encoder, net_decoder, crit, bbox_weight=0.1).to(device)
    # model.load_state_dict(torch.load('model/2018-10-26 22:11:34/50000/snap_model.pth'))
    # model = load_part_of_model_PSP_LSTM(model, param['pretrained_model'])
    model = load_part_of_model(model, param['pretrained_model'])
    # model.load_state_dict(torch.load(param['pretrained_model']))
    # optimizers = create_optimizers(nets, param)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr_encode'])
    model.train()
    for itr in range(param['total_iters']):
        x, y, bbox = dataset.next_batch()

        # y = y.reshape([y.shape[0], y.shape[2], y.shape[3]])
        # x = torch.from_numpy(x)
        x = torch.from_numpy(x[:, :3, :, :])
        y = torch.from_numpy(y)
        bbox = torch.from_numpy(bbox)

        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        bbox = bbox.type(torch.cuda.FloatTensor)
        # y = y.type(torch.cuda.LongTensor)

        # one-hot mask transfer
        # y_one_hot = torch.cuda.LongTensor(y.size(0), 2, y.size(2), y.size(3))
        # target = y_one_hot.scatter_(1, y.data, 1)

        loss = model(x, y, bbox, input_size=(param['crop_size'], param['crop_size']))
        # saliency = final_saliency.data.cpu().numpy()
        # final_saliency, cap_feats, local_pos, cap_feats2 = model(x, x_prior)

        # loss = train_loss1
        model.zero_grad()
        loss.backward()
        # train_loss2.backward()
        optimizer.step()
        # for optimizer in optimizers:
        #     optimizer.step()

        if itr % 5 == 0:
            log = 'step: %d, lr_encode: %g, lr_decode: %g, train_loss1:%g' % (itr, param['running_lr_encoder'], param['running_lr_decoder'], loss)
            print(log)
            log_loss.append(log)
            # summary_writer.add_summary(summary_str, itr)

        if itr % param['save_iters'] == 0:
            save_model(model, str(itr), time_str, log_loss)
            del log_loss[:]

        # adjust_learning_rate(optimizers, itr, param)
        adjust_learning_rate2(optimizer, itr, param)

    print('finish training time:', str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

def finetune_resnet50_dilated8_ppm_bilinear_prior(root_dir, prior_dir, list_file_path):
    param = {}
    param['lr_encode'] = 0.0001
    param['lr_decode'] = 0.0001
    param['momentum'] = 0.95
    param['beta1'] = 0.9
    param['weight_decay'] = 1e-4
    param['max_iters'] = 10000
    param['lr_pow'] = 0.9
    param['total_iters'] = 30001
    param['save_iters'] = 10000
    param['crop_size'] = 512

    param['running_lr_encoder'] = param['lr_encode']
    param['running_lr_decoder'] = param['lr_decode']

    # param['pretrained_model'] = 'model/2018-11-11 12:51:58/30000/snap_model.pth'
    param['pretrained_model'] = 'model/2018-11-15 21:39:15/30000/snap_model.pth'

    image_names = [line.strip() for line in list_file_path]
    # (self, root_dir, image_names, image_size, crop_size, batch_size, seq_size, horizontal_flip=False):

    dataset = ImageAndPriorSeqData(root_dir, root_dir + '_gt2_revised', prior_dir, image_names,
                           '.jpg', '.png', 512, param['crop_size'], 1, 5, horizontal_flip=False)


    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_loss = []
    # log_loss.append('warming up with single image data \n')
    log_loss.append('training with video data \n')
    log_loss.append('feature extractor: resnet50_dilated8_ppm_bilinear \n')
    log_loss.append('image size:530 \n')
    log_loss.append('crop size:512 \n')
    # log_loss.append('freeze layers except after conv5 \n')
    # log_loss.append('only smoothl1 loss \n')
    log_loss.append('only sigmoid cross entropy loss \n')
    log_loss.append('base network pspnet \n')
    log_loss.append('pre-trained model: ' + param['pretrained_model'] + '\n')
    log_loss.append('start lr: ' + str(param['lr_encode']) + '\n')
    log_loss.append('momentum: ' + str(param['momentum']) + '\n')

    device = torch.device('cuda')

    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch='resnet50_dilated8_prior',
        fc_dim=512
        # weights='weights_base/encoder_epoch_20.pth'
    )
    net_decoder = builder.build_decoder(
        arch='ppm_bilinear_deepsup',
        fc_dim=2048,
        num_class=1,
        # weights='weights_base/decoder_epoch_20.pth',
        use_softmax=False)

    crit = nn.BCEWithLogitsLoss()
    # crit = nn.NLLLoss()
    nets = (net_encoder, net_decoder, crit)

    model = SegmentationModule(net_encoder, net_decoder, crit, deep_sup_scale=0.4).to(device)
    # model.load_state_dict(torch.load('model/2018-10-26 22:11:34/50000/snap_model.pth'))
    # model = load_part_of_model_PSP_LSTM(model, param['pretrained_model'])
    model = load_part_of_model_prior(model, param['pretrained_model'])
    # model.load_state_dict(torch.load(param['pretrained_model']))
    # optimizers = create_optimizers(nets, param)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr_encode'])
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

        loss = model(x, y, input_size=(param['crop_size'], param['crop_size']))
        # saliency = final_saliency.data.cpu().numpy()
        # final_saliency, cap_feats, local_pos, cap_feats2 = model(x, x_prior)

        # loss = train_loss1
        model.zero_grad()
        loss.backward()
        # train_loss2.backward()
        optimizer.step()
        # for optimizer in optimizers:
        #     optimizer.step()

        if itr % 5 == 0:
            log = 'step: %d, lr_encode: %g, lr_decode: %g, train_loss1:%g' % (itr, param['running_lr_encoder'], param['running_lr_decoder'], loss)
            print(log)
            log_loss.append(log)
            # summary_writer.add_summary(summary_str, itr)

        if itr % param['save_iters'] == 0:
            save_model(model, str(itr), time_str, log_loss)
            del log_loss[:]

        # adjust_learning_rate(optimizers, itr, param)
        adjust_learning_rate2(optimizer, itr, param)

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

    # single frame to training
    # root_dir = '/home/ty/data/video_saliency'
    # list_file_path = open('/home/ty/data/video_saliency/train_all_single_frame.txt')

    # video sequences to training
    root_dir = '/home/ty/data/video_saliency/train_all'
    prior_dir = '/home/ty/data/video_saliency/train_all_prior'
    # list_file_path = open('/home/ty/data/video_saliency/train_all_seq_step_1.txt')
    list_file_path = open('/home/ty/data/video_saliency/train_all_seq_5f.txt')
    list_file_path = open('/home/ty/data/video_saliency/train_all_seq_bbox_5f.txt')
    finetune_resnet50_dilated8_ppm_bilinear(root_dir, list_file_path)
    # finetune_resnet50_dilated8_ppm_bilinear_prior(root_dir, prior_dir, list_file_path)