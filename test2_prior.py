import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from models_base import ModelBuilder, SegmentationModule
from tools.utils import preprocess3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tools.utils import load_part_of_model_PSP_whole



def test(test_dir, prior_dir, list_file_path, save_path):
    list_file = open(list_file_path)
    test_names = [line.strip() for line in list_file]

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    save_path = save_path + '_' + time_str
    device = torch.device('cuda')

    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch='resnet50_dilated8_prior',
        fc_dim=512,
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

    model = SegmentationModule(net_encoder, net_decoder, crit, deep_sup_scale=0.4).to(device)
    # model.load_state_dict(torch.load('model/2018-10-25 09:50:23/150000/snap_model.pth'))
    # model.load_state_dict(torch.load('model/2018-10-25 09:55:44/40000/snap_model.pth'))
    model.load_state_dict(torch.load('model/2018-12-24 22:38:10/30000/snap_model.pth'))
    # model = load_part_of_model_PSP_whole(model, 'model/2018-10-25 09:55:44/40000/snap_model.pth')
    # model.eval()
    # model = load_part_of_model(model, 'model/2018-08-22 18:04:08/6000/snap_model.pth')
    size = 380
    count = 0
    for name in test_names:
        images_path = name.split(',')
        batch_x = np.zeros([4, 4, size, size])

        for i, image_name in enumerate(images_path):
            if i == 4:
                continue
            image = Image.open(os.path.join(test_dir, image_name + '.jpg'))
            prior = Image.open(os.path.join(prior_dir, image_name + '.png'))

            src_w, src_h = image.size

            image = image.resize([size, size])
            prior = prior.resize([size, size])

            input = preprocess3(image)
            prior_arr = np.array(prior, dtype=np.float32)
            input = np.transpose(input, [0, 3, 1, 2])
            batch_x[i, :3, :, :] = input
            batch_x[i, 3, :, :] = prior_arr


        x = torch.from_numpy(batch_x)
        x = x.type(torch.cuda.FloatTensor)

        start = time.clock()
        saliency, branch, dymaic = model(x, None, segSize=(512, 512), input_size=(size, size))
        # saliency = saliency + branch
        saliency = F.sigmoid(saliency)
        branch = F.sigmoid(branch)
        end = time.clock()
        count += 1

        # final_saliency = saliency.data.cpu().numpy()
        # plt.subplot(1, 3, 1)
        # plt.imshow(final_saliency[3, 0, :, :])
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(final_saliency[2, 0, :, :])
        #
        # plt.subplot(1, 3, 3)
        # branch = branch.data.cpu().numpy()
        # plt.imshow(final_saliency[3, 0, :, :] - final_saliency[2, 0, :, :])
        #
        #
        # plt.show()

        saliency = saliency.data.cpu().numpy()
        saliency = saliency * 255
        save_sal = saliency.astype(np.uint8)
        save_img = Image.fromarray(save_sal[3, 0, :, :])
        save_img = save_img.resize([src_w, src_h])
        #
        image_path = os.path.join(save_path, images_path[-1] + '.png')
        print('process:', image_path)
        print('time:', end - start)
        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))

        save_img.save(image_path)

if __name__ == '__main__':
    # test dir
    # FBMS
    # test_dir = '/home/ty/data/FBMS/FBMS_Testset'
    # test_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
    # list_file_path = '/home/ty/data/FBMS/FBMS_seq_file.txt'

    # DAVIS
    test_dir = '/home/ty/data/davis/davis_test'
    test_prior_dir = '/home/ty/data/davis/davis_flow_prior'
    list_file_path = '/home/ty/data/davis/davis_test_seq.txt'

    save_path = 'total_result/result_rnn'

    test(test_dir, test_prior_dir, list_file_path, save_path)