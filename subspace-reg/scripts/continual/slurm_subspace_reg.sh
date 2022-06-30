

CURRENT="$PWD"
echo "$PWD"
DUMPED_PATH="$CURRENT/dumped"
EXP_FOLDER=$DUMPED_PATH/"cub/continual/pretrained/finetune_subspace_memory_base+novel_converge"
DATA_PATH="$CURRENT/data"
BACKBONE_FOLDER="${DUMPED_PATH}/final/subspace/"
mkdir -p $EXP_FOLDER
mkdir -p $BACKBONE_FOLDER
today_da=$(date "+%Y%m%d%H%M%S")


# For a single run comment out above nested for loop, leaving variable definitions,
# and use below.

EXP_NAME="seed_${today_da}"
LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
BACKBONE_PATH="${DUMPED_PATH}/cubtest/backbones/continual/efficientb0/pretrained0.01_freq50/ckpt_epoch_150.pth"
PRETRAINED_PATH="${BACKBONE_FOLDER}/ckpt_iter_9.pth"
python eval_incremental_copy_pretrained.py --model_path $BACKBONE_PATH \
                            --model efficientnet0 \
                            --no_dropblock \
                            --save_model_path $PRETRAINED_PATH \
                            --data_root $DATA_PATH \
                            --n_ways 10 \
                            --n_shots 5 \
                            --classifier linear \
                            --eval_mode few-shot-incremental-fine-tune \
                            --min_novel_epochs 20 \
                            --learning_rate 0.002 \
                            --freeze_backbone_at 1 \
                            --test_base_batch_size 64 \
                            --continual \
                            --n_queries 30 \
                            --lmbd_reg_transform_w 0.3 \
                            --target_train_loss 0.0 \
                            --label_pull 0.03 \
                            --set_seed 2 \
                            --stable_epochs 2 \
                            --num_workers 0 \
                            --word_embed_size 300 \
                            --attraction_override "distance2subspace" \
                            --memory_replay 1 > $LOG_STDOUT 2> $LOG_STDERR
                            --n_base_support_samples 1

