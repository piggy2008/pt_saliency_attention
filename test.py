import torch
import torch.nn.functional as F
import time
from models import VideoSaliency
from tools.utils import preprocess, resize_image_prior
import os
import numpy as np
from PIL import Image

def test(test_dir, test_prior_dir, list_file_path, save_path):
    list_file = open(list_file_path)
    test_names = [line.strip() for line in list_file]

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    save_path = save_path + '_' + time_str
    device = torch.device('cuda')
    model = VideoSaliency().to(device)

    # sigmoid + bbox regression
    # model.load_state_dict(torch.load('model/2018-08-26 17:11:03/6000/snap_model.pth'))
    #  maybe it is the best one, bbox regression smoothl1 no pulishment
    # result at total_result/result_rnn_2018-08-26 21:33:19

    # sigmoid + bbox regression
    # attention channel is 4, local pool4 and fc8 + additional rnn output
    # model.load_state_dict(torch.load('model/2018-08-28 16:54:21/6000/snap_model.pth'))

    # sigmoid + bbox regression
    # attention channel is 6, local pool4 and fc8 + additional rnn output, c3d
    # model.load_state_dict(torch.load('model/2018-08-29 09:56:15/10000/snap_model.pth'))

    # test for every model
    # model.load_state_dict(torch.load('model/2018-08-29 09:56:15/10000/snap_model.pth'))
    model.eval()
    # model = load_part_of_model(model, 'model/2018-08-22 18:04:08/6000/snap_model.pth')
    src_w = 0
    src_h = 0
    size = 400
    count = 0
    for name in test_names:
        images_path = name.split(',')
        batch_x = np.zeros([4, 4, size, size])
        batch_x_no_prior = np.zeros([4, 3, size, size])
        for i, image_name in enumerate(images_path):
            image = Image.open(os.path.join(test_dir, image_name + '.jpg'))
            prior = Image.open(os.path.join(test_prior_dir, image_name + '.png'))

            src_w, src_h = image.size

            image, prior = resize_image_prior(image, prior, size)

            input_prior, input = preprocess(image, prior, size)

            batch_x_no_prior[i] = input
            batch_x[i] = input_prior

        x_prior = torch.from_numpy(batch_x)
        x = torch.from_numpy(batch_x_no_prior)
        x = x.type(torch.cuda.FloatTensor)
        x_prior = x_prior.type(torch.cuda.FloatTensor)
        # feed_dict = {self.X: batch_x_no_prior, self.X_prior: batch_x}
        start = time.clock()
        saliency, _, bbox = model(x, x_prior)
        saliency = F.sigmoid(saliency)
        end = time.clock()

        count += 1

        # if count == 15:
        #     final_saliency = saliency.data.cpu().numpy()
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(final_saliency[3, 0, :, :])
        #
        #
        #     plt.subplot(1, 2, 2)
        #     bbox = bbox.data.cpu().numpy() * 400
        #     plt.imshow(final_saliency[3, 0, int(bbox[3, 0]):int(bbox[3, 2]), int(bbox[3, 1]):int(bbox[3, 3])])
        #
        #     plt.show()

        saliency = saliency.data.cpu().numpy()
        saliency = saliency * 255
        save_sal = saliency.astype(np.uint8)
        save_img = Image.fromarray(save_sal[3, 0, :, :])
        save_img = save_img.resize([src_w, src_h])

        image_path = os.path.join(save_path, images_path[-1] + '.png')
        print('process:', image_path)
        print('time:', end - start)
        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))

        save_img.save(image_path)

        # del batch_x
        # del batch_x_no_prior
        # del x
        # del x_prior
        # torch.cuda.empty_cache()


        # batch_x_no_prior = batch_x_no_prior.transpose([0, 2, 3, 1])
        #
        # plt.subplot(2, 1, 1)
        # plt.imshow(batch_x_no_prior[3, :, :, 0])
        #
        # plt.subplot(2, 1, 2)
        # plt.imshow(saliency[3, 0, :, :])
        #
        # plt.show()

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