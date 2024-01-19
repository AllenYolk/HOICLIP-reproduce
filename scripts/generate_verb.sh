ulimit -n 4096
set -x
EXP_DIR=exps/generate_hico_verb/hoiclip

swapon --show
free -h
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=8
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port $[29403 + i] \
        --use_env \
        main.py \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 1 \
        --use_nms_filter \
        --fix_clip \
        --batch_size 16 \
        --pretrained params/detr-r50-pre-2branch-hico.pth \
        --with_clip_label \
        --with_obj_clip_label \
        --gradient_accumulation_steps 1 \
        --num_workers 4 \
        --opt_sched "multiStep" \
        --dataset_root GENERATE_VERB \
        --model_name GENERATE_VERB \
        --no_training
sleep 120
