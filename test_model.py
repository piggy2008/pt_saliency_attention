import torch
import torch.nn as nn
import torch.nn.functional as F
from image_data_loader import ImageDataPretrain
from models_pspnet import PSPNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from models_base import ModelBuilder, SegmentationModule




def train_model_test(root_dir, list_file_path):
    # root_dir = '/home/ty/data/Pre-train'
    # list_file = open('/home/ty/data/Pre-train/pretrain_all_seq.txt')
    image_names = [line.strip() for line in list_file_path]
    # (self, root_dir, image_names, image_size, crop_size, batch_size, seq_size, horizontal_flip=False):
    dataset = ImageDataPretrain(root_dir, image_names, 550, 512, 5, horizontal_flip=True)


    device = torch.device('cuda')
    model_name = 'PSP'
    if model_name == 'PSP':
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

        model = SegmentationModule(net_encoder, net_decoder, crit, deep_sup_scale=0.4).to(device)
    else:
        model = PSPNet(n_classes=1, psp_size=832, backend='dpn68_warp').to(device)
    # training smooth l1 loss
    # criterion_regression = nn.SmoothL1Loss()

    # training sigmoid cross entropy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    model.load_state_dict(torch.load('model/2018-10-19 18:55:44/200000/snap_model.pth'))
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
            # log_loss.append('step: %d, train_loss1:%g' % (itr, train_loss1))
            # summary_writer.add_summary(summary_str, itr)




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
    train_model_test(root_dir, list_file_path)