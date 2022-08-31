cd src
# train
python main.py ctdet --exp_id kitti_certainnet_a6 --dataset kitti --arch certainnet_34 --lr_step 45,60 --num_epochs 70 --wh_weight 0.2 --gpus 0,1 --batch_size 15 --master_batch_size 7 --ablation 6
# test
python test.py ctdet --exp_id kitti_certainnet_a6 --dataset kitti --arch certainnet_34 --resume
cd ..