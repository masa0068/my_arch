import os

cmd = "python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results --hide_progress"
os.system(cmd)
cmd = "python linear_eval_alltasks.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results --hide_progress"
os.system(cmd)