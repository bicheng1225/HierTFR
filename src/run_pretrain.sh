#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=24000
#SBATCH --job-name="gopt"
#SBATCH --output=../exp/log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source ../venv-gopt/bin/activate

lr=1e-3
depth=3
head=3
batch_size=25
embed_dim=24
model=gopt
am=librispeech

exp_dir=../exp_hierTFR/v6_aspFix_preMdl/v61_v1

# repeat times
repeat_list=(0)

for repeat in "${repeat_list[@]}"
do
  mkdir -p $exp_dir/${repeat}
  python traintest_hierTFR_v6_aspfix1_pretrain.py --lr ${lr} --exp-dir ${exp_dir}/${repeat} --goptdepth ${depth} --goptheads ${head} \
  --batch_size ${batch_size} --embed_dim ${embed_dim} \
  --model ${model} --am ${am} --loss_w_phn 1 --loss_w_word 1 --loss_w_utt 1
done

