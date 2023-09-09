#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=GAN
#SBATCH --cpus-per-task 1
#SBATCH -o test_out.txt
#SBATCH -e test_err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate condaTorch

unzip /Dev/COMP3710-Demo-2/Part_3/data/img_align_celeba.zip /Dev/COMP3710-Demo-2/Part_3/data/celeba
python ~/Dev/COMP3710-Demo-2/Part_3/training.py
python ~/Dev/COMP3710-Demo-2/Part_3/eval.py
