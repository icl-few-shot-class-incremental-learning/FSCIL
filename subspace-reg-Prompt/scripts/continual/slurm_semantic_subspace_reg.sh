


CURRENT="$PWD"
DUMPED_PATH="$CURRENT/dumped"
EXP_FOLDER=$DUMPED_PATH/"cub/continual/pretrained/finetune_semantic_subspace_memory_base+novel_converge"
DATA_PATH="$CURRENT/cub"
mkdir -p $EXP_FOLDER

today_da=$(date "+%Y%m%d""%H%M%S")
EXP_NAME="pretrained_seed_${today_da}"
LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"





# For a single run comment out above nested for loop, leaving variable definitions,
# and use below.

BACKBONE_PATH="${DUMPED_PATH}/cubtest/backbones/continual/efficientb0/-0.1/ckpt_epoch_150.pth"

python eval_incremental_copy_pretrained.py --model_path $BACKBONE_PATH \
                        --model efficientnet0 \
                        --no_dropblock \
                        --data_root $DATA_PATH \
                        --n_shots 5 \
                        --n_ways 10 \
                        --classifier linear \
                        --eval_mode few-shot-incremental-fine-tune \
                        --min_novel_epochs 20 \
                        --learning_rate 0.002 \
                        --freeze_backbone_at 1 \
                        --test_base_batch_size 64 \
                        --continual \
                        --num_workers 0 \
                        --n_queries 30 \
                        --lmbd_reg_transform_w 0.2 \
                        --word_embed_size 300 \
                        --target_train_loss 0.0 \
                        --label_pull 0.2 \
                        --set_seed 2 \
                        --lmbd_reg_novel 0.3 \
                        --glove \
                        --temperature 3.0 \
                        --weight_decay 5e-4 \
                        --n_base_support_samples 1 \
                        --memory_replay 1 > $LOG_STDOUT 2> $LOG_STDERR
                        


