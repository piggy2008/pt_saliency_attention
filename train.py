import torch
import torch.nn as nn
import torch.nn.functional as F
from image_data_loader import ImageAndPriorSeqData, ImageAndPriorSeqBboxData
import time
from models import VideoSaliency
from utils import load_weights_from_h5, load_part_of_model, freeze_some_layers
import os
from matplotlib import pyplot as plt

def train(train_dir, label_dir, prior_dir, list_file_path):
    list_file = open(list_file_path)
    image_names = [line.strip() for line in list_file]

    # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
    dataset = ImageAndPriorSeqBboxData(train_dir, label_dir, prior_dir, None, None,
                                   None,
                                   image_names, None, '.jpg', '.png', 512, 480, 1,
                                   4,
                                   horizontal_flip=False)

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_loss = []

    log_loss.append('image size:512 \n')
    log_loss.append('crop size:480 \n')
    log_loss.append('freeze layers except after conv5 \n')
    log_loss.append('only smoothl1 loss \n')
    # log_loss.append(self.prior_type + '\n')
    device = torch.device('cuda')
    model = VideoSaliency().to(device)
    model.load_state_dict(torch.load('model/2018-08-22 18:04:08/6000/snap_model.pth'))
    # model = load_part_of_model(model, 'model/2018-08-22 18:04:08/6000/snap_model.pth')
    # model = freeze_some_layers(model)

    # training smooth l1 loss
    criterion_regression = nn.SmoothL1Loss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


    # model.load_state_dict(torch.load('model/2018-08-23 13:02:15/6000/snap_model.pth'))
    # training sigmoid cross entropy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



    # load model from h5 file
    # h5_path = '/home/ty/code/tf_saliency_attention/mat_parameter/fusionST_parameter_ms.mat'
    # load_weights_from_h5(model, h5_path)

    # load model from a pre-trained pytorch pth


    model.train()
    for itr in range(16001):
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

        final_saliency, rnn_output, local_pos = model(x, x_prior)
        train_loss1 = criterion(final_saliency, y)
        train_loss2 = criterion_regression(local_pos, bbox)

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
    image_dir = '/home/ty/data/video_saliency/train_all'
    label_dir = '/home/ty/data/video_saliency/train_all_gt2_revised'
    prior_dir = '/home/ty/data/video_saliency/train_all_prior'
    # list_file_path = '/home/ty/data/video_saliency/train_all_seq.txt'

    list_file_path = '/home/ty/data/video_saliency/train_all_seq_bbx.txt'
    train(image_dir, label_dir, prior_dir, list_file_path)