import numpy as np
import os
import cv2
import scipy.io as sio
import struct
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import torch

def resize_image_prior(image, prior, input_size=512):

    image = image.resize([input_size, input_size])
    prior = prior.resize([input_size, input_size])

    return image, prior

def preprocess(image, prior, input_shape=256):
    x = np.array(image, dtype=np.float32)
    x = x[:, :, ::-1]
    mean = (104.00699, 116.66877, 122.67892)
    mean = np.array(mean, dtype=np.float32)
    x = (x - mean)
    w, h, _ = x.shape
    prior_arr = np.array(prior, dtype=np.float32)
    input = np.zeros([input_shape, input_shape, 4], dtype=np.float32)
    input[:w, :h, :3] = x
    input[:w, :h, 3] = prior_arr

    input = input.transpose([2, 0, 1])

    return input[np.newaxis, ...], input[np.newaxis, :3, :, :]

def preprocess2(image, input_shape=512):
    x = np.array(image, dtype=np.float32)
    x = x[:, :, ::-1]
    mean = (104.00699, 116.66877, 122.67892)
    mean = np.array(mean, dtype=np.float32)
    x = (x - mean)
    w, h, _ = x.shape
    input = np.zeros([input_shape, input_shape, 3], dtype=np.float32)
    input[:w, :h, :] = x
    return input[np.newaxis, ...]

def preprocess3(image):
    x = np.array(image, dtype=np.float32)
    x = x[:, :, ::-1]
    mean = (104.00699, 116.66877, 122.67892)
    mean = np.array(mean, dtype=np.float32)
    x = (x - mean)
    # w, h, _ = x.shape
    # input = np.zeros([input_shape, input_shape, 3], dtype=np.float32)
    # input[:w, :h, :] = x
    return x[np.newaxis, ...]

def load_weights_from_h5(model, h5_path):
    m_dict = model.state_dict()
    parameter = sio.loadmat(h5_path)
    for name, param in m_dict.items():
        # print(name)
        layer_name, suffix = os.path.splitext(name)

        if layer_name == 'fc8':
            m_dict[layer_name + '.weight'].data = torch.from_numpy(parameter[layer_name + '_saliency_w'])
            print(name + '-------' + layer_name + '_saliency_w')
            m_dict[layer_name + '.bias'].data = torch.from_numpy(np.reshape(parameter[layer_name + '_saliency_b'], [-1]))
            print(name + '-------' + layer_name + '_saliency_b')
        elif layer_name == 'fc8_r2':
            m_dict[layer_name + '.weight'].data = torch.from_numpy(parameter['fc8_saliency_r2_w'])
            print(name + '-------' + 'fc8_saliency_r2_w')
            m_dict[layer_name + '.bias'].data = torch.from_numpy(np.reshape(parameter['fc8_saliency_r2_b'], [-1]))
            print(name + '-------' + 'fc8_saliency_r2_b')
        elif layer_name.find('convLSTM') >= 0:
            print(name)
            continue
        elif layer_name.find('loc_estimate') >= 0:
            print(name)
            continue
        elif layer_name.find('attention') >= 0:
            print(name)
            continue
        else:

            if suffix == '.weight':
                m_dict[layer_name + '.weight'].data = torch.from_numpy(parameter[layer_name + '_w'])
                print(name + '-------' + layer_name + '_w')
            elif suffix == '.bias':
                m_dict[layer_name + '.bias'].data = torch.from_numpy(np.reshape(parameter[layer_name + '_b'], [-1]))
                print(name + '-------' + layer_name + '_b')
            else:
                print (name)

    # print(m_dict['conv5_3_r2.bias'].data.shape)
    # torch.save(m_dict, 'model/base_model.pth')
    model.load_state_dict(m_dict)
    return model

def load_part_of_model(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)

        param = src_model.get(k)
        m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model_prior(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)
        if k.find('encoder.conv1.weight') >= 0:
            print('override')
            param = src_model.get(k)
            # tmp = m_dict[k].data
            # tmp[:, :3, :, :] = param
            # tmp[:, 3, :, :] = param[:, 2, :, :]
            # m_dict[k].data = tmp
            print(param.shape)
        elif k.find('LSTM') >= 0:
            param = src_model.get(k)

            # m_dict[k.replace('conv_last.4', 'classify_conv')] = param
            # print('new model param shape:', np.shape(m_dict[k].data))
            print('override and shape:', np.shape(param))
        elif k.find('classify_conv') >= 0:
            param = src_model.get(k)
            print('override and shape:', np.shape(param))
        else:
            param = src_model.get(k)
            m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model_LSTM(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)
        if k.find('LSTM') >= 0:
            param = src_model.get(k)

            # m_dict[k.replace('conv_last.4', 'classify_conv')] = param
            # print('new model param shape:', np.shape(m_dict[k].data))
            print('override and shape:', np.shape(param))
        # elif k.find('classify_conv') >= 0:
        #     param = src_model.get(k)
        #     print('override and shape:', np.shape(param))
        else:
            param = src_model.get(k)
            m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model_resnet(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)
        if k.find('fc') >= 0:
            print('override')

        # elif k.find('norm') >= 0:
        #     print('override convlstm norm')
        # elif k.find('loc_estimate') >= 0:
        #     print('override loc_estimate')
        else:
            param = src_model.get(k)
            m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model_decode(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)
        if k.find('last.4') >= 0 or k.find('last_deepsup') >= 0:
            param = src_model.get(k)
            print('new model param shape:', np.shape(m_dict[k].data))
            print('override and shape:', np.shape(param))
        # elif k.find('norm') >= 0:
        #     print('override convlstm norm')
        # elif k.find('loc_estimate') >= 0:
        #     print('override loc_estimate')
        else:
            param = src_model.get(k)
            m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model


def load_part_of_model_PSP_whole(new_model, src_model_path):

    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)

        if k.find('block.4') >= 0:
            param = src_model.get(k)

        if k.find('conv_last.4') >= 0:
            param = src_model.get(k)

            m_dict[k.replace('conv_last.4', 'classify_conv')] = param
            # print('new model param shape:', np.shape(m_dict[k].data))
            print('override and shape:', np.shape(param))
        # elif k.find('norm') >= 0:
        #     print('override convlstm norm')
        # elif k.find('loc_estimate') >= 0:
        #     print('override loc_estimate')
        elif k.find('auxlayer') >= 0:
            param = src_model.get(k)
            print('override and shape:', np.shape(param))
            m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model_dss(new_model, src_model_path):
    src_model = sio.loadmat(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        # print (k)
        if k == '__header__' or k == '__globals__' or k == '__version__':
            continue

        k_src = k

        if k.find('_w') >= 0:
            k = k.replace('_w', '.weight')

        if k.find('_b') >= 0:
            k = k.replace('_b', '.bias')

        if k.find('-'):
            k = k.replace('-', '_')

        print(k_src, '-------', k)



        param = src_model.get(k_src)

        if k.find('bias') >= 0:
            param = np.reshape(param, [-1])

        m_dict[k].data = torch.from_numpy(param)

def load_part_of_model_PSP_LSTMNorm(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)
        if k.find('g_norm') >= 0 or k.find('i_norm') >= 0 or k.find('f_norm') >= 0 \
                or k.find('o_norm') >= 0 or k.find('c_norm') >= 0:
            param = src_model.get(k)

            # m_dict[k.replace('conv_last.4', 'classify_conv')] = param
            # print('new model param shape:', np.shape(m_dict[k].data))
            print('override and shape:', np.shape(param))
        # elif k.find('norm') >= 0:
        #     print('override convlstm norm')
        # elif k.find('loc_estimate') >= 0:
        #     print('override loc_estimate')

        else:
            param = src_model.get(k)
            m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model_PSP_LSTM(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)
        if k.find('LSTM') >= 0:
            param = src_model.get(k)

            # m_dict[k.replace('conv_last.4', 'classify_conv')] = param
            # print('new model param shape:', np.shape(m_dict[k].data))
            print('override and shape:', np.shape(param))
        # elif k.find('norm') >= 0:
        #     print('override convlstm norm')
        # elif k.find('loc_estimate') >= 0:
        #     print('override loc_estimate')
        else:
            param = src_model.get(k)
            m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model

def freeze_some_layers(model):
    for child in model.named_children():
        if child[0].find('fc') >= 0 or child[0].find('attention') >= 0 \
                or child[0].find('loc_estimate') >= 0:
            print(child[0] + ' not froze')
            continue
        else:
            print(child[0] + ' froze')
            for param in child[1].parameters():
                param.requires_grad = False

    return model

def gaussian_mask(center_x, center_y, sigma=0.25):

    x = np.arange(0, 1, 0.0025)
    y = np.arange(0, 1, 0.0025)
    x, y = np.meshgrid(x, y)
    z = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (sigma ** 2))

    # plt.figure()
    #
    # plt.imshow(z, plt.cm.gray)
    # plt.show()
    return z

def generate_gaussian_mask(center, sigma, size):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    # Need an (N, 2) array of (x, y) pairs.
    x, y = np.meshgrid(x, y)

    mu = np.array(center)

    # sigma = np.array([[0.075, 0], [0, 0.025]])
    sigma = np.array([[sigma[0], 0], [0, sigma[1]]])
    pos = np.empty(x.shape + (2,))

    pos[:, :, 0] = x
    pos[:, :, 1] = y

    # print(pos.shape)

    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)

    N = np.sqrt((2 * np.pi) ** n * sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu)
    mask = np.exp(-fac / 2) / N

    return mask / np.max(mask)

if __name__ == '__main__':
    # img = cv2.imread('Comp_195.bmp')
    # anno = cv2.imread('Comp_195.png', 0)
    # crf_refine('test', img, anno)
    # loadBinFile('.dcl_crf/test.bin')
    # weight = np.zeros([4, 3, 3, 3], dtype=np.float16)
    # weight[:, :, :, 0] = np.ones([4, 3, 3], dtype=np.float16)
    # import random
    # for i in range(0, 20):
    #     print(random.randint(0,2))
    # print (weight[3, :, :, 1])

    # from models_base import VideoSaliency
    #
    # model = VideoSaliency()
    #
    # h5_path = '/home/ty/code/tf_saliency_attention/mat_parameter/fusionST_parameter_ms.mat'
    #
    # load_weights_from_h5(model, h5_path)

    # load_part_of_model(model, 'model/2018-08-20 21:35:07/26000/snap_model.pth')
    a = torch.ones([1, 64, 50, 50])
    a = a.type(torch.FloatTensor)
    center = [0.5, 0.5]
    # sigma:0.005 ~ 0.025, 0.005 ~ 0.025
    sigma = [0.025, 0.005]

    z = generate_gaussian_mask(center, sigma, 50)
    z = z.reshape([1, 1, 50, 50])
    z = torch.from_numpy(z)
    z = z.type(torch.FloatTensor)
    print(z.size())
    plt.imshow(z[0, 0, :, :])
    plt.show()




