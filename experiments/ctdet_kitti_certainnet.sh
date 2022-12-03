cd src
# train
python main.py ctdet --exp_id kitti_certainnet_a6-22 --dataset kitti --arch certainnet_34 --lr_step 45,60 --num_epochs 80 --wh_weight 0.2 --gpus 0,1 --batch_size 16 --ablation 6

python main.py ctdet --exp_id bdd_certainnet_a6 --dataset bdd --arch certainnet_34 --lr_step 45,60 --num_epochs 80 --wh_weight 0.2 --gpus 0,1 --batch_size 16 --ablation 6
# test
python test.py ctdet --exp_id kitti_certainnet_a6-10 --dataset kitti --arch certainnet_34 --resume --ablation 6
python test.py ctdet --exp_id kitti_certainnet_a6-22 --dataset kitti --arch certainnet_34 --resume --ablation 6 --flip_test
# demo
python demo.py ctdet --exp_id kitti_certainnet_a6-10 --dataset kitti --arch certainnet_34 --resume --ablation 6 --demo /work/shuhei-ky/exp/CenterNet/data/kitti/training/image_2/000019.png
cd ..