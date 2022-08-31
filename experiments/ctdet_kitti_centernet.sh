cd src
# train
python main.py ctdet --exp_id kitti_centernet --dataset kitti --arch dla_34 --gpus 0,1
# test
python test.py ctdet --exp_id kitti_centernet --dataset kitti --arch dla_34 --resume
cd ..
