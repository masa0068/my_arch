import os

cmd = "python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c100.yaml --ckpt_dir ./checkpoints/cifar100_results --hide_progress"
os.system(cmd)
cmd = "python linear_eval_alltasks.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c100.yaml --ckpt_dir ./checkpoints/cifar100_results --hide_progress"
os.system(cmd)