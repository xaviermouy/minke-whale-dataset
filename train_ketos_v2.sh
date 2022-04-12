#!/bin/bash
#SBATCH --time=0-15:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6000M
#SBATCH --account=def-bpirenne
#SBATCH --job-name=ketos_train_test2

module load python/3.9
module load cuda cudnn
source /home/xmouy/ketos-env/bin/activate

echo 'First test...'
python3 /home/xmouy/documents/minke/scripts/train_ketos.py --batch_size 100 --n_epochs 30 --db_file ./database.h5 --recipe_file ./recipe.json --spec_config_file ./spec_config.json --out_dir ./ --checkpoints_dir ./ --logs_dir ./
