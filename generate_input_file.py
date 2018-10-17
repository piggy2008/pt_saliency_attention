import os
from PIL import Image
import cv2
import numpy as np
import glob

path = '/home/ty/data/video_saliency/train_all_gt2_revised'
save_path = '/home/ty/data/video_saliency/train_all_seq_bbox.txt'
# path = '/home/ty/data/davis/davis_test'
# save_path = '/home/ty/data/davis/davis_test_seq.txt'
# save_path = '/home/ty/data/video_saliency/train_all_seq.txt'
folders = os.listdir(path)
file = open(save_path, 'w')

batch = 4

def generate_one():

    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            name, suffix = os.path.splitext(image)
            print (os.path.join(folder, name))
            file.writelines(os.path.join(folder, name) + '\n')

    file.close()

def generate_seq():
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for i in range(1, len(images) - batch + 1):
            image_batch = ''
            for j in range(batch):

                image = images[i + j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 3:
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_gt_box():
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for i in range(1, len(images) - batch + 1):
            image_batch = ''
            for j in range(batch):

                image = images[i + j]
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 3:
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def change_suffix():
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            img = Image.open(os.path.join(path, folder, image))
            name, suffix = os.path.splitext(image)
            if not os.path.exists(os.path.join(save_path, folder)):
                os.makedirs(os.path.join(save_path, folder))
            img.save(os.path.join(save_path, folder, name + '.jpg'))

def gt_generate():
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            img = cv2.imread(os.path.join(path, folder, image), 0)
            img[np.where(img > 100)] = 255
            img[np.where(img <= 100)] = 0
            # img[np.where(img == 255)] = 1
            if not os.path.exists(os.path.join(save_path, folder)):
                os.makedirs(os.path.join(save_path, folder))

            cv2.imwrite(os.path.join(save_path, folder, image), img)

def generate_MSRA10K(root):
    names = os.listdir(os.path.join(root, 'MSRA10K_Imgs_GT', 'Imgs'))
    names.sort()

    save_path = '/home/ty/data/Pre-train/pretrain_all_seq.txt'
    file = open(save_path, 'w')
    for name in names:
        # print (os.path.join(root, 'Imgs', name[:-4] + '.png'))
        # cv2.imread(os.path.join(root, 'GT', name[:-4] + '.png'))
        line = os.path.join('MSRA10K_Imgs_GT', 'Imgs', name) + ' ' + os.path.join('MSRA10K_Imgs_GT', 'GT', name[:-4] + '.png')
        print (line)
        file.writelines(line + '\n')

    file.close()
    print(len(names))

def generate_THUR15K(root):
    folders = os.listdir(os.path.join(root, 'THUR15000'))
    # names.sort()

    save_path = '/home/ty/data/Pre-train/pretrain_all_seq2.txt'
    file = open(save_path, 'w')
    for folder in folders:
        names = os.listdir(os.path.join(root, 'THUR15000', folder, 'GT'))
        # cv2.imread(os.path.join(root, 'GT', name[:-4] + '.png'))
        for name in names:

            line = os.path.join('THUR15000', folder, 'Src', name[:-4] + '.jpg') + ' ' + os.path.join('THUR15000', folder, 'GT', name)
            print (line)
            print(os.path.exists(os.path.join(root, 'THUR15000', folder, 'Src', name[:-4] + '.jpg')))
            print(os.path.exists(os.path.join(root, 'THUR15000', folder, 'GT', name)))

            file.writelines(line + '\n')

    file.close()
    # print(len(names))


# generate_one()

# generate_MSRA10K('/home/ty/data/Pre-train')
generate_THUR15K('/home/ty/data/Pre-train')
# change_suffix()
# generate_seq()