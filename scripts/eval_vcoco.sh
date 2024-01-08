python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained ./exps/vcoco/hoiclip/checkpoint_best.pth \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --eval \
        --model_name HOICLIP \
        --zero_shot_type default \
        --with_clip_label \
        --with_obj_clip_label \
        --use_nms_filter \
        --verb_pth ./tmp/vcoco_verb.pth \
        --verb_weight 0.1 \
        --output_dir ./exps/vcoco/hoiclip/eval \
        --training_free_enhancement_path \
            ./training_free_ehnahcement/