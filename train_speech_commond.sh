seed=1

DATA=/home/v-lbei/speech_commands/processed_data/raw/
SAVE=checkpoints/speech_command/ode_mega_sc_raw_base_fp32
CHUNK=1000
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh



model=ode_mega_sc_raw_base
cmd="python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters
    -a ${model} --task speech_commands --encoder-normalize-before
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01
    --batch-size 10 --sentence-avg --update-freq 2 --max-update 250000
    --lr-scheduler linear_decay --total-num-update 250000 --end-learning-rate 0.0
    --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0
    --sentence-class-num 10 --max-positions 16000 --sc-dropped-rate 0 --enc-calculate-num 2"


export CUDA_VISIBLE_DEVICES=0
cmd="nohup "${cmd}" > $SAVE/train.log 2>&1 &"
eval $cmd
tail -f $SAVE/train.log
