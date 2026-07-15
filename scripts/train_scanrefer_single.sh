TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
    --nproc_per_node 2 --master_port 12221 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /data1/bone/mjgong/datasets/TSP3D_DATA/ \
    --val_freq 3 --batch_size 14 --grad_accum_steps 1 --save_freq 6 --print_freq 500 \
    --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=0.01 --num_workers=16 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --log_dir /data1/bone/mjgong/datasets/TSP3D_DATA/output/logs \
    --lr_decay_epochs 50 75 \
    --augment_det \
    --rng_seed 0 \
    --use_spota \
    --spota_greedy_topk \
    --spota_k 32 \
    --wandb --wandb_project TSP3D_BASE --wandb_run_name baseline+spota_greedy_topk_seed0_4 \