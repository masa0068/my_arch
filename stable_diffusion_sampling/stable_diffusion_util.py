import os
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
from diffusers import StableDiffusionImg2ImgPipeline
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from typing import List, Tuple
from utils.buffer import Buffer
from torch.cuda.amp import autocast
import numpy as np

from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torchvision import transforms

class ImageGenerator:
    def __init__(self, args, dataset, device="cuda"):
        self.args = args
        self.device = device

        self.cluster_fig_save = False

        self.split_num = self.args.diffusion.split_img #GPU負担軽減
        self.gen_class_num = self.args.diffusion.gen_class_num

        self.classes = dataset.N_CLASSES_PER_TASK
        self.tasks = dataset.N_TASKS
        self.choose_latents = self.args.model.choose_latents

        self.guidance_scale = self.args.diffusion.guidance_scale
        self.strength_min = self.args.diffusion.strength_min
        self.strength_max = self.args.diffusion.strength_max

        if self.args.diffusion.strength_min > self.args.diffusion.strength_max:
            self.strength_min = self.args.diffusion.strength_max
            self.strength_max = self.args.diffusion.strength_min

        """バッファの定義"""
        self.buffer = Buffer(self.tasks * (self.classes*self.choose_latents) - (self.classes*self.choose_latents), "cuda")

        """パイプラインの定義"""
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.args.diffusion.model_id, torch_dtype=torch.float16).to(self.device)
        self.pipe.safety_checker = dummy_safety_checker

        self.now_task = 1

        import shutil
        self.choice_data_dir = "./choice_data"
        if os.path.exists(self.choice_data_dir):  # ディレクトリが存在する場合
            shutil.rmtree(self.choice_data_dir)  # ディレクトリとその中身を削除



    """潜在変数をバッファに保存"""
    def latent_to_buffer(self, dset_loaders):
        
        latent_list = []

        """1バッチ取り出し"""
        for idx, data in enumerate(dset_loaders):
            (_, _, notaug_images), _ = data
            break

        """バッチごと"""
        for i in range(0, len(notaug_images), self.split_num):
            batch_images = notaug_images[i:i+self.split_num].to(self.device, dtype=torch.float16)

            """潜在変数"""
            latents = self.pipe(
                prompt=[""]*len(batch_images), 
                image=batch_images,
                strength=0,
                output_type="latent"
            ).images

            for latent in latents:
                latent_list.append(latent)
            
        """潜在変数クラスタリング"""
        latents = self.latent_clastaring(latent_list)
        
        """何が選択されたか確認"""
        if self.args.diffusion.save_img:
            img = self.pipe(
                prompt=[""]*len(latents), 
                image=latents,
                strength=0
            ).images

            """ディレクトリを再作成"""
            os.makedirs(self.choice_data_dir, exist_ok=True)
            for idx, output_image in enumerate(img):
                resized_image = output_image.resize((64, 64))  # サイズ変更（必要なら）
                resized_image.save(f"{self.choice_data_dir}/task{self.now_task}_img{idx + 1}.JPEG")

        self.buffer.add_data(latents)



    """潜在変数をクラスタリング"""
    def latent_clastaring(self, latent_list):
        latent_array = torch.stack(latent_list).cpu().numpy()
        latent_flat = latent_array.reshape(latent_array.shape[0],-1)

        """K-means"""
        kmeans = KMeans(
            n_clusters=self.classes,
            init='k-means++',
            n_init=10,          #初期化
            max_iter=500,       #イテレーション
            tol=1e-4,           #収束
            algorithm='elkan',  # 効率的なアルゴリズム
        )
        clusters = kmeans.fit_predict(latent_flat)
        cluster_centers = kmeans.cluster_centers_  # KMeansクラスタの中心を取得

        """データ選別"""
        latents_to_save_list = []
        selected_points = []  # 選択された潜在変数の座標を保存

        """潜在変数をクラスタごとに分割"""
        clusterd_latents = [[] for _ in range(self.classes)]#空リストをクラスタ分用意
        for latent, cluster_idx in zip(latent_list, clusters):
            clusterd_latents[cluster_idx].append(latent)#各クラスタごとの潜在変数を保持したリストのリスト
        
        """各クラスタから中心に最も近いデータを選択"""
        for cluster_idx, latent_group in enumerate(clusterd_latents):
            latent_group_tensor = torch.stack(latent_group)  # クラスタ内のテンソルをまとめる
            latent_group_array = latent_group_tensor.cpu().numpy().reshape(len(latent_group), -1)  # 平坦化
            distances = np.linalg.norm(latent_group_array - cluster_centers[cluster_idx], axis=1)  # 距離を計算
            
            closest_idx = np.argsort(distances)[:self.choose_latents] #中心に近い点を複数選択
            for idx in closest_idx:
                latents_to_save_list.append(latent_group_tensor[idx])  # テンソル形式でリストに追加
                selected_points.append(latent_group_array[idx])  # 座標を記録

        latents_to_save_tensor = torch.stack(latents_to_save_list)  #選別し，まとめて返す値

        """選択されたデータの可視化"""
        if self.cluster_fig_save:
            """PCA"""
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_flat)#2次元[バッチ，2]
            cluster_centers_2d = pca.transform(cluster_centers) # PCAで2次元に変換
            selected_points_2d = pca.transform(np.array(selected_points))  # 選択されたデータを2次元に変換

            """可視化"""
            plt.figure(figsize=(10, 8))
            for cluster_idx in range(self.classes):
                cluster_points = latent_2d[clusters == cluster_idx]#クラスタに属するデータ点のみを抽出
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx}")
            
            for center in cluster_centers_2d:  # クラスタの中心をプロット
                plt.scatter(center[0], center[1], s=200, c='red', marker='x')

            """選択されたデータをプロット"""
            plt.scatter(
                selected_points_2d[:, 0],
                selected_points_2d[:, 1],
                s=100,
                c='none',
                marker='o',
                edgecolors='black',  # 外枠を黒に
                label="Selected Points",
            )
            save_dir = "cluster_fig" #保存先のディレクトリ
            os.makedirs(save_dir, exist_ok=True)  

            plt.legend()
            plt.title("Latent Clustering")
            plt.grid(True)
            save_path = os.path.join(save_dir, str(self.now_task)+"-latent_clustering.png")
            plt.savefig(save_path)  
            plt.close()

        return latents_to_save_tensor



    """画像生成"""
    def sampling(self, dir_path=None):
        
        total_task_images = self.gen_class_num * self.classes * self.choose_latents  # 1タスクあたり生成する画像数
        
        output_images = []  # 全タスクで生成された画像を格納

        aug3d_promts = [
            "Oil painting style", 
            "Watercolor style", 
            "Pixel art style", 
            "Metallic texture",
            "Wood texture",
            "Stained glass style", 
            "Mosaic style"
        ]

        for current_task in range(self.now_task):  # 各タスクを順に処理
            generated_images_count = 0  # 現在のタスクで生成した画像数
            task_output_images = []  # 現在のタスクで生成された画像を格納

            """バッファから潜在変数を取得（各タスクのクラス数分"""
            latents = self.buffer.get_task_data(current_task+1, self.classes, self.choose_latents)[0].to(self.device, dtype=torch.float16)

            latents_split = torch.split(latents, self.split_num)#分割　GPU軽減 (おそらくまともに機能しない)

            for latents_batch in latents_split:
                latents_size = latents_batch.size(0)
                
                for _ in range(self.gen_class_num): #潜在変数1つにつき生成する枚数

                    """分割分のプロンプト作成"""
                    positive_prompts = random.choices(aug3d_promts, k=latents_size)
                    negative_prompts = [self.args.diffusion.negative_prompt] * latents_size

                    """潜在変数から生成"""
                    gen_img = self.pipe(
                        prompt=positive_prompts,
                        negative_prompt=negative_prompts,
                        image=latents_batch,
                        strength=random.uniform(self.strength_min, self.strength_max),
                        guidance_scale=self.guidance_scale
                    ).images

                    """タスクの出力に追加"""
                    task_output_images.extend(gen_img)
                    generated_images_count += len(gen_img)

                    print(f"[Task {current_task + 1}] Generated {generated_images_count}/{total_task_images} images.")

            """保存"""
            if self.args.diffusion.save_img:
                if self.args.diffusion.save_all:#nタスク以外の過去のタスクも保存
                    for idx, output_image in enumerate(task_output_images):
                        output_image.resize((64, 64))
                        output_image.save(f"{dir_path}/task{current_task + 1}_img{idx + 1}.JPEG")
                    print(f"*.*. {len(task_output_images)} samples saved for Task {current_task + 1} .*.*")

                elif current_task+1 == self.now_task:#nタスクのみ
                    for idx, output_image in enumerate(task_output_images):
                        output_image.resize((64, 64))
                        output_image.save(f"{dir_path}/task{current_task+1}_img{idx + 1}.JPEG")
                    print(f"*.*. {len(task_output_images)} samples saved for Task {current_task + 1} .*.*")

            output_images.extend(task_output_images)# 全体の出力に追加
        self.now_task += 1

        return output_images

    

"""ダミーの安全チェッカー関数"""
def dummy_safety_checker(images, clip_input: torch.Tensor) -> Tuple[List, List[bool]]:
    return images, [False] * len(images) #NSFWにより黒で返される画像を無効化



class GeneratedImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform if transform else ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image
    

def create_gen_dataset(samples, transform=None, batch_size=256, shuffle=True):

    dataset = GeneratedImageDataset(samples, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



