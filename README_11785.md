## Download ESC50
mkdir -p ~/datasets
cd ~/datasets
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git
### check dataset
ls ~/datasets/ESC-50/meta/esc50.csv
ls ~/datasets/ESC-50/audio | head
## Finetune with ESC50
cd ~/11-785_EfficientAT
export EFFICIENTAT_ESC50_DIR=~/datasets/ESC-50
python ex_esc50.py --cuda --pretrained --model_name=mn10_as --fold=1
## Evaluation
python ex_esc50.py   --cuda   --eval_only   --model_name=mn10_as   --fold=1   --checkpoint_path=wandb/ESC50_20260311_204237/mn10_esc50_epoch_79_acc_952.pt
