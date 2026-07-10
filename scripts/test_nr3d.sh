TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 2222 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /data1/bone/mjgong/TSP3D_DATA/ \
    --val_freq 3 --batch_size 1 --save_freq 6 --print_freq 500 \
    --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=0.01 --num_workers=8 \
    --dataset nr3d --test_dataset nr3d \
    --detect_intermediate --joint_det \
    --log_dir /data1/bone/mjgong/datasets/TSP3D_DATA/output/logs \
    --lr_decay_epochs 150 \
    --checkpoint_path /data1/bone/mjgong/research/backups/TSP3D_vanilla/TSP3D_2/pretrained_models/ckpt_nr3d.pth \
    --eval