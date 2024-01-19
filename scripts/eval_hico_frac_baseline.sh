python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained ./exps/hico_5%/hoiclip_baseline/checkpoint_best.pth \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --frac 0.05 \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --eval \
        --num_workers 1 \
        --model_name HOICLIP \
        --zero_shot_type default \
        --with_clip_label \
        --with_obj_clip_label \
        --use_nms_filter \
        --verb_pth ./tmp/verb.pth \
        --output_dir exps/hico_5%/hoiclip_baseline/eval \
        --training_free_enhancement_path \
            ./training_free_ehnahcement
