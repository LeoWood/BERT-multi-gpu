#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -J pre_R_cscd

/public/software/deeplearning/anaconda3/bin/python3 run_pretraining.py\
 --input_file=/public/home/zzx6320/lh/Projects/bert/data/cscibert_pre_training/pre_training_R_cscd_128.tfrecord\
 --bert_config_file=/public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/bert_config.json\
 --init_checkpoint=/public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/bert_model.ckpt\
 --output_dir=/public/home/zzx6320/lh/Projects/bert/outputs/Pre_cscd_R_128_from_bert\
 --max_seq_length=128\
 --do_train=True\
 --do_eval=True\
 --train_batch_size=32\
 --learning_rate=2e-5\
 --num_train_steps=800000\
 --save_checkpoints_steps=10000\
 --n_gpus=4^

