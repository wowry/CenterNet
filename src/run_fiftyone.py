import fiftyone as fo

data_path = "/work/shuhei-ky/exp/CenterNet/data/kitti/training/image_2"
labels_path = "/work/shuhei-ky/exp/CenterNet/data/kitti/annotations/kitti_3dop_val.json"

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
)

session = fo.launch_app(dataset, remote=True, address="0.0.0.0")
