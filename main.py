import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple

import cv2
import time
import wandb
from wandb import AlertLevel
import torch.distributed as dist
from datetime import datetime
from copy import deepcopy
from itertools import cycle
import copy
from itertools import zip_longest
from utils.buffer import Buffer
from stable_diffusion_sampling.stable_diffusion_util import ImageGenerator,create_gen_dataset
from augmentations.simsiam_aug import SimSiamTransform

def main(device, args):  

    dataset = get_dataset(args)

    dataset_copy = get_dataset(args)
    
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
    dataset_copy = get_dataset(args)

    """保存するフォルダ"""
    today = datetime.today().strftime('%Y%m%d')
    parent_exp_path = "SAMPLE/{}/{}task_{}class/{}ep".format(args.name, dataset.N_TASKS, dataset.N_CLASSES_PER_TASK, args.train.stop_at_epoch)
    if not os.path.exists(parent_exp_path):
        os.makedirs(parent_exp_path, exist_ok=True)
        
    """wandb"""
    if args.wandb:
        wand_project_name = "SDLUMP_{}_{}task_{}class_{}e_{}bs".format(args.dataset.name,dataset.N_TASKS, dataset.N_CLASSES_PER_TASK, args.train.num_epochs, args.train.batch_size)

    """結果辞書"""
    results = {'knn-cls-acc':[],
                'knn-cls-each-acc':[],
                'knn-cls-max-acc':[],
                'knn-cls-fgt':[],}
    
    """モデル定義"""
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))
    logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)

    """画像生成クラス"""
    gen = ImageGenerator(args, dataset=dataset, device="cuda")
    
    """データローダのセットアップ"""
    train_loaders, memory_loaders, test_loaders = [], [], []
    for t in range(dataset.N_TASKS):
        print("=====[",t+1,"]TASK DATALODAR SETTING=====")
        ntra_tr, ntra_me, ntra_te = dataset_copy.get_data_loaders(args)
        train_loaders.append(ntra_tr)
        memory_loaders.append(ntra_me)
        test_loaders.append(ntra_te)
    print("=============ALL",dataset.N_TASKS,"TASKS=============\n")


    """各タスク処理"""
    for t in range(dataset.N_TASKS):
        print("\n", "#"*10, "\n  ",t+1,"TASK\n","#"*10,"\n")

        if args.wandb:  
            wandb.init(project=wand_project_name, group=today, name=str(t+1))

        if args.eval.type == 'all':
            eval_tids = [j for j in range(dataset.N_TASKS)]
        elif args.eval.type == 'curr':
            eval_tids = [t]
        elif args.eval.type == 'accum':
            eval_tids = [j for j in range(t + 1)]
        else:
            sys.exit('Stopped!! Wrong eval-type.')


        """画像生成"""
        if t > 0: #2タスク目以降は画像生成
            exp_dir = os.path.join(parent_exp_path, today+"/"+str(t+1)+"task")
            os.makedirs(exp_dir, exist_ok=True)

            """潜在変数をバッファに格納"""
            print("="*3,"LATENT -> BUFFER","="*3)
            gen.latent_to_buffer(dset_loaders=copy_train_loaders)

            """画像生成"""
            print("="*34,"SAMPLING","="*34)
            samples = gen.sampling(dir_path=exp_dir)
            
            """生成画像のデータローダ"""
            transform = SimSiamTransform(args.dataset.image_size)
            gen_dset_loaders = {x: 
                                create_gen_dataset(samples, 
                                                   transform, 
                                                   batch_size= t * dataset.N_CLASSES_PER_TASK * args.diffusion.gen_class_num * args.model.choose_latents // len(copy_train_loaders), #生成画像数 / イテレーション数 
                                                   shuffle=True)#データローダ
                            for x in ["train"]}

        """画像生成に使うデータローダ"""
        copy_train_loaders = copy.deepcopy(train_loaders[t])


        """学習"""
        print("="*32,"MODEL TRANING START","="*32)
        global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
        """各エポック処理"""
        for epoch in global_progress:
            """wandb"""
            if args.wandb:  
                wandb.log({"epoch":epoch})

            model.train()
            
            if t == 0:
                ziploaders = enumerate(train_loaders[t])
            else:
                # gen_dset_loadersとdset_loadersをzip_longestで結合
                combined_loader = zip_longest(train_loaders[t], gen_dset_loaders["train"], fillvalue=None)
                ziploaders = enumerate(combined_loader)

            local_progress = tqdm(ziploaders, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
            for idx, data in local_progress:
                if t == 0:
                    (images1, images2, notaug_images), _ = data  
                else:  
                    if data[0] is not None:
                        (images1, images2, notaug_images), _ = data[0]

                        if data[1] is not None: #生成データが存在する
                            (gen_images1, gen_images2, gen_notaug_images) = data[1]

                            if args.diffusion.mixup: #mixupする　（ここら辺の処理はもっとスマートにできると思います）
                                lam = np.random.beta(args.train.alpha, args.train.alpha)
                                
                                if images1.size(0) >= gen_images1.size(0) :#１バッチで実画像の方が多く，実画像が生成画像に合わせる場合
                                    images1_mixup, images1_remaining = images1[:gen_images1.size(0)], images1[gen_images1.size(0):]
                                    images2_mixup, images2_remaining = images2[:gen_images2.size(0)], images2[gen_images2.size(0):]
                                    notaug_images_mixup, notaug_images_remaining = notaug_images[:gen_notaug_images.size(0)], notaug_images[gen_notaug_images.size(0):]
                                    mixup_images1 = gen_images1
                                    mixup_images2 = gen_images2
                                    mixup_notaug_images = gen_notaug_images

                                else : #１バッチで生成画像の方が多く，生成画像が実画像に合わせる場合
                                    images1_mixup, images1_remaining = gen_images1[:images1.size(0)], gen_images1[images1.size(0):]
                                    images2_mixup, images2_remaining = gen_images2[:images2.size(0)], gen_images2[images2.size(0):]
                                    notaug_images_mixup, notaug_images_remaining = gen_notaug_images[:notaug_images.size(0)], gen_notaug_images[notaug_images.size(0):]
                                    mixup_images1 = images1
                                    mixup_images2 = images2
                                    mixup_notaug_images = notaug_images

                                #mixup処理 + 残りの結合処理
                                images1_mixup = lam * images1_mixup + (1 - lam) * mixup_images1
                                images2_mixup = lam * images2_mixup + (1 - lam) * mixup_images2
                                notaug_images_mixup = lam * notaug_images_mixup + (1 - lam) * mixup_notaug_images

                                images1 = torch.cat((images1_mixup, images1_remaining), dim=0)
                                images2 = torch.cat((images2_mixup, images2_remaining), dim=0)
                                notaug_images = torch.cat((notaug_images_mixup, notaug_images_remaining), dim=0)

                            else:#mixupしない
                                images1 = torch.cat((images1, gen_images1))
                                images2 = torch.cat((images2, gen_images2))
                                notaug_images = torch.cat((notaug_images, gen_notaug_images))

                data_dict = model.observe(images1, images2)# モデル入力

                logger.update_scalers(data_dict)
            global_progress.set_postfix(data_dict)


            """全エポック終了後  １タスクごとの評価"""
            if (epoch + 1) == args.train.stop_at_epoch:
                if args.train.knn_monitor:#学習中の評価を監視
                    knn_acc_list = []
                    for i in eval_tids:
                        acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i],
                                                    device, args.cl_default, task_id=i, k=min(args.train.knn_k, len(eval_tids)))
                        knn_acc_list.append(acc)
                    
                    kfgt = []
                    results['knn-cls-each-acc'].append(knn_acc_list[-1])
                    results['knn-cls-max-acc'].append(knn_acc_list[-1])

                    for j in range(t):#学習済みのタスク分
                        if knn_acc_list[j] > results['knn-cls-max-acc'][j]:
                            results['knn-cls-max-acc'][j] = knn_acc_list[j]
                        kfgt.append(results['knn-cls-each-acc'][j] - knn_acc_list[j])
                    results['knn-cls-acc'].append(np.mean(knn_acc_list))
                    results['knn-cls-fgt'].append(np.mean(kfgt))
            

        """モデル保存"""
        model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.net.state_dict()
        }, model_path)
        print(f"Task Model saved to {model_path}")

        with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
            f.write(f'{model_path}')
        with open(os.path.join(f'{args.log_dir}', f"%s_accuracy_logs.txt"%args.name), 'w+') as f:
            f.write(str(results))
        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if args.wandb:
            if (t+1) == dataset.N_TASKS:
                wandb.alert(
                    title="["+today+"]_"+wand_project_name,
                    text='{}の学習が終わりました'.format(wand_project_name),
                    level=AlertLevel.INFO
                )
            wandb.finish() 


    if args.eval is not False and args.cl_default is False:
        args.eval_from = model_path



def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if classifier is not None:
                outputs = classifier(outputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
        
        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')

    