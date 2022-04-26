#!/bin/bash
#SBATCH --time=0-30:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000M
#SBATCH --account=def-bpirenne
#SBATCH --job-name=ketos-5s-window

module load python/3.9
module load cuda cudnn
source /home/xmouy/ketos-env/bin/activate

echo '5-second window'
python3 ./train_ketos.py --batch_size=100 --n_epochs=30 --db_file=./ketos_databases/spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800/database.h5 --recipe_file=./ketos_databases/spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800/recipe.json --spec_config_file=./ketos_databases/spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800/spec_config.json --out_dir=./results/5s --checkpoints_dir=./results/5s --logs_dir=./results/5s
