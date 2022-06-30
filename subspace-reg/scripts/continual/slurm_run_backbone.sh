
CURRENT="$PWD"
DUMPED_PATH="$CURRENT/dumped/cubtest"
DATA_PATH="$CURRENT/cub"
BACKBONE_FOLDER="${DUMPED_PATH}/backbones/continual/efficientb0/pretrained0.05/"
mkdir -p BACKBONE_FOLDER

today_da=$(date "+%Y%m%d""%H%M%S")
EXP_NAME="pretrained_seed_${today_da}"

LOG_STDOUT="${DUMPED_PATH}/${EXP_NAME}.out"
LOG_STDERR="${DUMPED_PATH}/${EXP_NAME}.err"

# If running for a single seed use below (comment out above
# keeping the variable definitions such as BACKBONE_FOLDER):
#--trial pretrain \#--tb_path tb \
python train_supervised_copy.py --data_root $DATA_PATH \
                        --classifier linear \
                        --model_path $BACKBONE_FOLDER \
                        --continual \
                        --model efficientnet0 \
                        --save_freq 50 \
                        --no_dropblock  \
                        --no_linear_bias \
                        --epochs 150