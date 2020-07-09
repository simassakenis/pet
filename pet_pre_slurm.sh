#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_dgx1        # Partition to submit to
#SBATCH --mem=16000          # Memory pool for all cores (see also --mem-pe    r-cpu)
#SBATCH --gres=gpu:1        # Number of gpus
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inse    rts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inse    rts jobid
module load Anaconda3/2019.10
source activate pet
python run_training.py \
--wrapper_type mlm \
--train_examples 10 \
--data_dir data/mnli/ \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name mnli \
--output_dir /n/shieber_lab/Lab/users/ssakenis/pet/output_pet_pre_mnli_10 \
--do_train \
--do_eval \
--pattern_ids 0 1 2 3 \
--lm_training \
--lm_train_examples_per_label 10000 \
--save_train_logits \
--gradient_accumulation_steps 4 \
--max_steps 1000 \
--per_gpu_train_batch_size 1 \
--per_gpu_helper_batch_size 3
