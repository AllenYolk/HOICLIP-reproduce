python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained ./exps/hico/hoiclip/checkpoint_best.pth \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --frac 0.05 \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --eval \
        --model_name HOICLIP \
        --zero_shot_type default \
        --with_clip_label \
        --with_obj_clip_label \
        --use_nms_filter \
        --verb_pth ./tmp/verb.pth \
        --verb_weight 0.1 \
        --output_dir exps/hico/hoiclip/eval \
        --training_free_enhancement_path \
            ./training_free_ehnahcement/