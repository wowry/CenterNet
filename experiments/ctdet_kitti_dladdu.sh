# train
python main.py ctdet --exp_id kitti_resddu --dataset kitti --arch resddu_101 --gpus 0,1
python main.py ctdet --exp_id kitti_dladdu-100 --dataset kitti --arch dladdu_34 --gpus 0,1
python main.py ctdet --exp_id kitti_resdcnddu --dataset kitti --arch resdcnddu_101 --gpus 0,1

# test
python test.py ctdet --exp_id kitti_dladdu-100 --dataset kitti --arch dladdu_34 --resume
python test.py ctdet --exp_id kitti_resddu --dataset kitti --arch resddu_101 --resume
python test.py ctdet --exp_id kitti_resdcnddu --dataset kitti --arch resdcnddu_101 --resume
