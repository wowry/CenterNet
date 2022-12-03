# train
python main.py ctdet --exp_id kitti_centernet-5 --dataset kitti --arch dla_34 --gpus 0,1
# test
python test.py ctdet --exp_id kitti_centernet-2 --dataset kitti --arch dladcnddu_34 --resume --flip_test
