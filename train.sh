D:/apps/miniconda3/envs/GNN/python.exe -u main_local_global.py \
    --train 1 \
    --use_all True \
    --interp_grad False \
    --focal_loss False \
    --config_filename 'setting/Treatment_local.yaml' \
    --mixup_rate 0.4 \
    --shift_robust False \
    --shift_loss_weight 1.0 \
    --use_qn True \
    --use_duration True \
    --lr 0.005 \
    --wd 0.01 \
    --edropout 0.3 \
    --dropout 0.5 \
    --snowball_layer_num 8 \
    --ckpt_path './save_models/pre_only' \
    --stepsize 200 \
    --gamma 0.5 \
    --norm_mode PN-SI \
    --norm_scale 10.0 \
> train.log 2>&1 