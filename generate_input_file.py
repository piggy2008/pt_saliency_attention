import os
from PIL import Image
import cv2
import numpy as np
import glob



def generate_one(folders, path, file):

    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            name, suffix = os.path.splitext(image)
            print (os.path.join(folder, name))
            file.writelines(os.path.join(folder, name) + '\n')

    file.close()

def generate_seq(folders, path, file, batch):
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
                if j == (batch - 1):
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_with_step(folders, path, file, batch, step):
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for i in range(1, len(images) - batch * step + 1):
            image_batch = ''
            for j in range(i, i + batch * step, step):

                image = images[j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == (i + batch * step - step):
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_gt_box(folders, path, batch, file):
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

def change_suffix(folders, path, save_path):
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            img = Image.open(os.path.join(path, folder, image))
            name, suffix = os.path.splitext(image)
            if not os.path.exists(os.path.join(save_path, folder)):
                os.makedirs(os.path.join(save_path, folder))
            img.save(os.path.join(save_path, folder, name + '.jpg'))

def gt_generate(folders, path, save_path):
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
def generate_video_data_single_frame(root):
    folders = os.listdir(os.path.join(root, 'train_all'))
    # names.sort()

    save_path = '/home/ty/data/video_saliency/train_all_single_frame.txt'
    file = open(save_path, 'w')
    for folder in folders:
        names = os.listdir(os.path.join(root, 'train_all', folder))
        # cv2.imread(os.path.join(root, 'GT', name[:-4] + '.png'))
        for name in names:

            line = os.path.join('train_all', folder,  name[:-4] + '.jpg') + ' ' + os.path.join('train_all_gt2_revised', folder, name[:-4] + '.png')
            print (line)
            # print(os.path.exists(os.path.join(root, 'THUR15000', folder, 'Src', name[:-4] + '.jpg')))
            # print(os.path.exists(os.path.join(root, 'THUR15000', folder, 'GT', name)))

            file.writelines(line + '\n')

    file.close()

def split_VOS(all_file_root, test_root, save_root):
    test_folders = os.listdir(test_root)
    all_folders = os.listdir(all_file_root)
    test_folders.sort()
    all_folders.sort()
    import shutil
    for folder in all_folders:
        if folder not in test_folders:
            src_path = os.path.join(all_file_root, folder)
            dst_path = os.path.join(save_root, folder)
            print (src_path + '------>' + dst_path)
            shutil.copytree(src_path, dst_path)

def generate_VOS_seq(root, batch, save_path):
    gt_folders = os.listdir(root + '/gt')
    img_folders = root + '/imgs'
    file = open(save_path, 'w')
    for folder in gt_folders:
        gt_imgs = os.listdir(root + '/gt/' + folder)
        imgs = os.listdir(root + '/imgs/' + folder)
        gt_imgs.sort()
        imgs.sort()
        for img in gt_imgs:
            name = img[:-4]
            index = imgs.index(name + '.jpg')
            img_name = imgs[index]
            # print (img_name + '-------' + img)
            line = ''
            for i in range(batch):
                if i == batch - 1:
                    line += folder + '/' + imgs[index - batch + 1 + i][:-4]
                else:
                    line += folder + '/' + imgs[index - batch + 1 + i][:-4] + ','

            print (line)
            file.writelines(line + '\n')

    file.close()

if __name__ == '__main__':

    # generate_one()
    # generate_seq()
    # generate_MSRA10K('/home/ty/data/Pre-train')
    # generate_THUR15K('/home/ty/data/Pre-train')
    # generate_video_data_single_frame('/home/ty/data/video_saliency')
    # change_suffix()
    # generate_seq()

    # generate_seq_with_step(3)
    # path = '/home/ty/data/video_saliency/train_all_gt2_revised'
    # save_path = '/home/ty/data/video_saliency/train_all_seq_step_1.txt'


    # path = '/home/ty/data/davis/davis_test'
    # save_path = '/home/ty/data/davis/davis_test_seq_5f.txt'
    # # save_path = '/home/ty/data/video_saliency/train_all_seq.txt'
    # folders = os.listdir(path)
    # file = open(save_path, 'w')
    #
    # batch = 5
    #
    # generate_seq(folders, path, file, batch)

    # all_file_root = '/home/ty/data/VOS/Mask'
    # test_root = '/home/ty/data/VOS_test/images'
    # save_root = '/home/ty/data/VOS_train/gt'
    # split_VOS(all_file_root, test_root, save_root)

    root = '/home/ty/data/VOS_train'
    save_path = '/home/ty/data/VOS_train/train_VOS_seq_5f.txt'
    generate_VOS_seq(root, 5, save_path)
