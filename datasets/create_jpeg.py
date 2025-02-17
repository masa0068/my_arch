import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .imgfolder import random_split, ImageFolderTrainVal
from augmentations.simsiam_aug import SimSiamTransform

import diffusion.utils as utils

def create_jpeg(args, n_task, img_path, data_loaders):
    
    order_name = os.path.join(img_path,"order_seed={}.pkl".format("None"))#.format(order_random_seed))
    print("Order name:{}".format(order_name))

    if os.path.exists(order_name):
        print("Loading orders")
        order = utils.unpickle(order_name)
    else:
        print("Generating orders")
        #if order_random_seed is None:
        if None is None:
            order = np.arange(100)
        else:
            order = np.arange(100)
            np.random.shuffle(order)
        utils.savepickle(order, order_name)

    num_img = 0
    dsets_list = []
    for task in range(1, n_task+1):#n_task+1
        print("="*29,"CREATE JPEG FOR[ {} ]TASK".format(task),"="*29)
        img_task_path = os.path.join(img_path, str(task))#各タスクで画像を保存するためのフォルダ

        dsets, count_img = create_train_test_val_imagefolders(args, task, data_loaders, img_task_path, num_img)
        num_img += count_img
        #torch.save(dsets, os.path.join(img_path, str(task), outfile))
        print("SIZES: train={}, val={}, test={}, num_img={}".format(len(dsets['train']), len(dsets['val']),
                                                        len(dsets['test']),num_img))
        dsets_list.append(dsets)
    return dsets_list


def create_train_test_val_imagefolders(args, task, data_loaders, img_task_path, num_img):
    dsets = {}

    # train, val, test フォルダのパスを作成
    train_path = os.path.join(img_task_path, 'train')
    val_path = os.path.join(img_task_path, 'val')
    test_path = os.path.join(img_task_path, 'test')

    #jpeg作成
    print("creating jpeg...")
    train_list, train_count = save_to_JPEG(data_loaders["train"][task-1], train_path, num_img)#data_loaders["train"]だけだとリストなので，[task-1]でデータローダ
    val_list, val_count = save_to_JPEG(data_loaders["val"][task-1], val_path, num_img)
    test_list, _ = save_to_JPEG(data_loaders["test"][task-1], test_path, num_img)
    print("creation finish")

    normalize =  transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
                                        transforms.Resize(64),])
    
    # 各データセットのクラスを取得
    train_classes = get_classes_in_dataloader(data_loaders["train"][task-1])
    val_classes = get_classes_in_dataloader(data_loaders["val"][task-1])
    test_classes = get_classes_in_dataloader(data_loaders["test"][task-1])
    print("classes:", train_classes)

    augmentation = SimSiamTransform(args.dataset.image_size)#SimSiam用

    dsets['train'] = ImageFolderTrainVal(train_path, None,
                                        transform=augmentation, classes=train_classes,
                                        class_to_idx=train_classes,
                                        imgs=train_list)
    dsets['val'] = ImageFolderTrainVal(val_path, None,
                                        transform=normalize, classes=val_classes,
                                        class_to_idx=val_classes,
                                        imgs=val_list)
    dsets['test'] = ImageFolderTrainVal(test_path, None,
                                        transform=normalize, classes=test_classes,
                                        class_to_idx=test_classes,
                                        imgs=test_list)

    return dsets, train_count+val_count


def save_to_JPEG(dataloader, output_path, num_img):
    return_list = []
    counter = 0
    # print("type dataloader : ", type(dataloader))

    if not os.path.exists(output_path):  # 指定したファイルが存在しない
        os.makedirs(output_path, exist_ok=True)
        
        for batch in dataloader:  # DataLoaderからバッチで取得
            imgs, labels = batch  # imgとlabelをバッチとして取得
            
            # imgsがリストで各要素がバッチとなっている場合の処理
            for img, label in zip(imgs, labels):  # imgsの各バッチ内をループ

                label_val = label.item()  # tensorをintに変換
                filename = os.path.join(output_path, f"{label_val}_{num_img + counter}.JPEG")

                # TensorをPIL Imageに変換してリサイズ
                img = transforms.ToPILImage()(img)  # バッチ内の1つの画像を変換
                #resized_img = img.resize((64, 64), Image.BILINEAR)
                resized_img = img
                resized_img.save(filename)

                return_list.append((filename, label_val))
                counter += 1

    else:  # 指定したファイルが存在する リストを返すための処理
        print("FOLDER EXISTS:", output_path)
        img_list = os.listdir(output_path)
        for img in img_list:
            p = img
            l = int(p.split('_')[0])
            return_list.append((os.path.join(output_path, p), l))
            counter += 1

    return return_list, counter

def get_classes_in_dataloader(dataloader):
    classes = set()  # 重複を避けるためにセットを使用

    for batch in dataloader:
        _, labels = batch  # ラベルを取得
        classes.update(labels.tolist())  # ラベルをリストに変換してセットに追加

    classes = sorted(list(classes))  # ソートしてリストに変換
    return classes



