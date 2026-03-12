## Finetune with ESC50
cd ~/11-785_EfficientAT
export EFFICIENTAT_ESC50_DIR=~/datasets/ESC-50
python ex_esc50.py --cuda --pretrained --model_name=mn10_as --fold=1
