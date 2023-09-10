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

python /home/Student/s4501559/Dev/COMP3710-Demo-2/Part_3/training.py