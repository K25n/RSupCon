MODEL="--model resnet18" 
DATASET="--dataset cifar100"

EPOCH="200"
LOSS_TYPE="SupCon"
LOSS_OPT="--loss_type ${LOSS_TYPE}"

ATK_OPT="--atk_eps 8 --atk_alpha 2 --attack_steps 10"
ATK_OPT_SUP="--trans_order t0,s0,b0 --atk_anchor t0,s0 --atk_contrast t0,s0,b0 --atk_randstart t0,s0 --cln_anchor b0"
ENCODER_OPT="--batch_size 512 --weight_decay 5e-4 --learning_rate 0.1 --temp 0.07"
LOG_NAME="from_scratch"

# # Encoder training from scratch
python main_rsupcon.py $MODEL $DATASET $ENCODER_OPT --epochs $EPOCH $ATK_OPT $ATK_OPT_SUP  --name $LOG_NAME
MODEL_CKPT="./ckpt/cifar100/"$LOG_NAME"/last.pth"

# # # Linear classifier training
LINEAR_AUG="trivial"
LINEAR_LOSS="CE"
LINEAR_BATCH="512"
LINEAR_EPOCH="10"
LINEAR_LR="1.0"

python main_linear.py $MODEL $DATASET --batch_size $LINEAR_BATCH --epoch $LINEAR_EPOCH --learning_rate $LINEAR_LR --ckpt $MODEL_CKPT --linear_loss $LINEAR_LOSS --name $LOG_NAME
LINEAR_CKPT="./ckpt/cifar100/"$LOG_NAME"/linear_"$LINEAR_LOSS"_"$LINEAR_AUG"_lr_"$LINEAR_LR"_epoch_"$LINEAR_EPOCH"_bsz_"$LINEAR_BATCH".pth"

# # Robustness evaluation across all metrics
EVAL_OPT="--test_option all"
python main_eval.py --test_option all --model_ckpt $MODEL_CKPT --linear_ckpt $LINEAR_CKPT --log_name $LOG_NAME

