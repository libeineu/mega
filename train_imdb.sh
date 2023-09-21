seed=1

# DATA=/home/v-lbei/speech_commands/processed_data/raw/
DATA=../lra/imdb-4000
# SAVE=checkpoints/speech_command/ode_mega_sc_raw_base_fp32
SAVE=checkpoints/lra/listops/ode_mega_lra_imdb_lr004
CHUNK=128
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh



model=ode_mega_lra_imdb
cmd="python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-text --input-type text \
    --encoder-layers 4 --n-dim 16 --chunk-size ${CHUNK} \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --norm-type 'scalenorm' --sen-rep-type 'mp' \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.004 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 25 --sentence-avg --update-freq 2 --max-update 25000 --required-batch-size-multiple 1 \
    --lr-scheduler linear_decay --total-num-update 25000 --end-learning-rate 0.0 \
    --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 100 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 --enc-calculate-num 2"


export HIP_VISIBLE_DEVICES=1
cmd="nohup "${cmd}" > $SAVE/train.log 2>&1 &"
eval $cmd
tail -f $SAVE/train.log
