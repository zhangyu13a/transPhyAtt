
# Transferable Physical Attack against Object Detection with Separable Attention


## Requirements

 - To install requirements:
    - pytorch3d: 0.6.0
    - torch: 1.8.0
    - torchvision: 0.9.0

>📋  you need to download dataset and model weight before running the code:

- dataset 
    - both trainset and testset can be accessed in https://pan.baidu.com/s/17Ct17jdDPOripL79peGIcw (tran)

    - the trainset is made up of two .rar files, which could be unpaced into dataset\trainset with WinRAR software.

- model weight

    - All the models are trained on Visdrone2019 and our dataset collected from CARLA. The checkpoint of yolo-v3 can be downloaded here: https://pan.baidu.com/s/1E3-n3S2hSkb5rINI7sHgYw (tran)

>📋  Our 3D model mesh and texture are provided in 3d_model folder and indexs of the sampled faces could be accessed in top_faces.txt.   

## Training

To train the model in the paper, run this command:

```train
python train.py --train_dir <path_to_data> --weightfile <path_to_weight>   --batch_size 2 --epochs 5
```

## Testing
After training, the code yields two files (patch_save.pt and idx_save.pt), which include information of adversarial texture. The final adversarial samples could be obtained by running this command:

```test
python test.py --test_dir <path_to_data>    --patch_dir <path_to_patch>
```

<!-- ## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.  -->
