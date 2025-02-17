import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter, knn_monitor
from datasets import get_dataset
from models.optimizers import get_optimizer, LR_Scheduler
from utils.loggers import *

import time
from datetime import datetime


def evaluate_single(model, dataset, test_loader, memory_loader, device, k, last=False) -> Tuple[list, list, list, list]:
    accs, accs_mask_classes = [], []
    knn_accs, knn_accs_mask_classes = [], []
    correct = correct_mask_classes = total = 0
    knn_acc, knn_acc_mask = knn_monitor(model.net.module.backbone, dataset, memory_loader, test_loader, device, args.cl_default, task_id=k, k=min(args.train.knn_k, len(dataset.memory_loaders[k].dataset))) 

    return knn_acc


def main(device, args):

    for evl in ["all", "accum"]:
        args.eval.type = evl

        wand_project_name = "SDLUMP_EVAL"
        today = datetime.today().strftime('%Y%m%d')

        csv_path = f"[{args.eval.type}]_{args.name}_each_task_each_model.csv"
        print("type : " + args.eval.type)

        dataset = get_dataset(args)
        dataset_copy = get_dataset(args)
        train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
        model = get_model(args, device, len(train_loader), get_aug(train=False, train_classifier=False, **args.aug_kwargs))
        results = {'knn-cls-acc':[],
                    'knn-cls-each-acc':[],
                    'knn-cls-max-acc':[],
                    'knn-cls-fgt':[],}
        each_task_each_model = []

        train_loaders, memory_loaders, test_loaders = [], [], []
        for t in range(dataset.N_TASKS):
            tr, me, te = dataset.get_data_loaders(args)
            train_loaders.append(tr)
            memory_loaders.append(me)
            test_loaders.append(te)

        for t in tqdm(range(0, dataset_copy.N_TASKS), desc='Evaluatinng'):
            if args.eval.type == 'all':
                eval_tids = [j for j in range(dataset.N_TASKS)]
            elif args.eval.type == 'curr':
                eval_tids = [t]
            elif args.eval.type == 'accum':
                eval_tids = [j for j in range(t + 1)]
            else:
                sys.exit('Stopped!! Wrong eval-type.')

            model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
            save_dict = torch.load(model_path, map_location='cpu')

            msg = model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k}, strict=True)
            model = model.to(args.device)
            knn_acc_list = []


            for i in eval_tids:
                acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i],
                        device, args.cl_default, task_id=i, k=min(args.train.knn_k, t+1))
                knn_acc_list.append(acc) 

            each_task_each_model.append(knn_acc_list)

            kfgt = []
            results['knn-cls-each-acc'].append(knn_acc_list[-1])
            results['knn-cls-max-acc'].append(knn_acc_list[-1])

            for j in range(t):
                if knn_acc_list[j] > results['knn-cls-max-acc'][j]:
                    results['knn-cls-max-acc'][j] = knn_acc_list[j]
                kfgt.append(results['knn-cls-max-acc'][j] - knn_acc_list[j])
            results['knn-cls-acc'].append(np.mean(knn_acc_list))
            results['knn-cls-fgt'].append(np.mean(kfgt))

        print("results : ",results)
        print("each task each model results : ", each_task_each_model)


        """精度＆忘却率の平均"""
        #精度の平均を計算
        acc_avg = sum(results['knn-cls-acc']) / len(results['knn-cls-acc'])
        print("accracy average : ",acc_avg)

        # nanを削除
        filtered_fgt = [x for x in results['knn-cls-fgt'] if not np.isnan(x)]

        # 忘却率の平均を計算
        fgt_avg = sum(filtered_fgt) / len(filtered_fgt)
        print("forgetting average : ",fgt_avg)


        """txt保存"""
        with open(os.path.join(f'{args.log_dir}', f"%s_accuracy_logs.txt"%args.name), 'w+') as f:
            f.write(str(results))


        """CSV保存"""
        with open(csv_path, mode="w", encoding="utf-8", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            for task_results in each_task_each_model:
                csv_writer.writerow(task_results)



if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
