#!/bin/bash
#SBATCH --time=0-15:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6000M
#SBATCH --account=def-bpirenne
#SBATCH --job-name=ketos_train_test

module load python/3.9
module load cuda cudnn
source /home/xmouy/ketos-env/bin/activate

cp /home/xmouy/documents/minke/ketos_db/spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800/database.h5 $SLURM_TMPDIR
cp /home/xmouy/documents/minke/ketos_db/spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800/recipe.json $SLURM_TMPDIR
cp /home/xmouy/documents/minke/ketos_db/spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800/spec_config.json $SLURM_TMPDIR


echo 'First test...'
python3 /home/xmouy/documents/minke/scripts/train_ketos.py --batch_size 100 --n_epochs 30 --db_file $SLURM_TMPDIR/database.h5 --recipe_file $SLURM_TMPDIR/recipe.json --spec_config_file $SLURM_TMPDIR/spec_config.json --out_dir /home/xmouy/documents/minke/runs --checkpoints_dir /home/xmouy/documents/minke/runs --logs_dir /home/xmouy/documents/minke/runs

