# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple


def reservoir(num_seen_latents: int, buffer_size: int) -> int:
    if num_seen_latents < buffer_size:
        return num_seen_latents

    rand = np.random.randint(0, num_seen_latents + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_latents: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_latents % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """初期化"""
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size #引数指定
        self.device = device
        self.num_seen_latents = 0
        self.functional_index = eval(mode)

        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['latents'] #保存する属性


    """必要なバッファを初期化"""
    def init_tensors(self, latents: torch.Tensor) -> None:
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))


    """バッファに追加"""
    def add_data(self, latents):
        if not hasattr(self, 'latents'):#属性が存在しないなら初期化
            self.init_tensors(latents)

        for i in range(latents.shape[0]):
            index = reservoir(self.num_seen_latents, self.buffer_size)
            self.num_seen_latents += 1
            if index >= 0:
                latents_tensor = latents[i].clone().detach().to('cuda')
                self.latents[index] = latents_tensor

    # """指定タスクの中でバッファから取り出し"""
    # def get_task_data(self, task: int, classes: int) -> Tuple:
    #     start_idx = (task-1) * classes  # タスクに対応する開始インデックス
    #     end_idx = start_idx + classes     # タスクに対応する終了インデックス

    #     # 範囲外のインデックスが指定されないように制約を適用
    #     start_idx = max(0, start_idx)
    #     end_idx = min(self.latents.shape[0], end_idx)

    #     # バッファから指定されたタスク範囲の潜在変数を取得
    #     task_latents = self.latents[start_idx:end_idx]

    #     # 選択された潜在変数をTensor化して返す
    #     ret_tuple = (torch.stack([ee.cpu() for ee in task_latents]).to(self.device),)

    #     return ret_tuple


    """指定タスクの中でバッファから取り出し"""
    def get_task_data(self, task: int, classes: int, choose_latents: int) -> Tuple:
        start_idx = (task-1) * classes * choose_latents  # タスクに対応する開始インデックス
        end_idx = start_idx + classes * choose_latents    # タスクに対応する終了インデックス
        # 範囲外のインデックスが指定されないように制約を適用
        start_idx = max(0, start_idx)
        end_idx = min(self.latents.shape[0], end_idx)

        # バッファから指定されたタスク範囲の潜在変数を取得
        task_latents = self.latents[start_idx:end_idx]

        # 選択された潜在変数をTensor化して返す
        ret_tuple = (torch.stack([ee.cpu() for ee in task_latents]).to(self.device),)

        return ret_tuple

    """バッファから取り出し"""
    def get_data(self, size: int) -> Tuple:
        if size > min(self.num_seen_latents, self.latents.shape[0]):
            size = min(self.num_seen_latents, self.latents.shape[0])

        choice = np.random.choice(min(self.num_seen_latents, self.latents.shape[0]),
                                  size=size, replace=False)
        ret_tuple = (torch.stack([ee.cpu() for ee in self.latents[choice]]).to(self.device),)

        return ret_tuple


    """バッファの有無"""
    def is_empty(self) -> bool:
        if self.num_seen_latents == 0:
            return True
        else:
            return False


    """全てのデータを取得"""
    def get_all_data(self) -> Tuple:
        ret_tuple = (torch.stack([ee.cpu() for ee in self.latents]).to(self.device),)
        return ret_tuple


    """バッファを削除"""
    def empty(self) -> None:
        if hasattr(self, 'latents'):
            delattr(self, 'latents')
        self.num_seen_latents = 0
