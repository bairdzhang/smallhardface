# Robust Face Detection via Learning Small Faces on Hard Images
## Performance on WIDER FACE val, FDDB, Pascal Faces and AFW
[Link to the trained model](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EV4rn_lxo45Lj5VYEwljqncBIhwbJns4zTJ_BHwjwPa05g?e=hR6QnT)  

| WIDER FACE val easy | WIDER FACE val medium | WIDER FACE val hard | FDDB | Pascal Faces | AFW |
|:-------|:-------|:-------|:-------|:-------|:-------|
| 95.7 | 94.9 | 89.7 | 98.7 | 99.0 | 99.6 |  

## Build source code
1. Clone this repository to `$ROOT`
1. Install python library `cd $ROOT; pip install -r requirements.txt`
1. Install graphviz `apt-get install -y graphviz`
1. Edit the caffe configure file `$ROOT/caffe/Makefile.config`
1. Compile caffe `cd $ROOT/caffe; make -j; make -j pycaffe`
1. Compile extra library `cd $ROOT/lib; make -j`

## Download the dataset
* WIDER FACE  
Download dataset from [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/), and also download the `ground_truth` from [official evaluation toolkit](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip).
```
.
|-- WIDER_train
|   `-- images [61 entries exceeds filelimit, not opening dir]
|-- WIDER_val
|   `-- images [61 entries exceeds filelimit, not opening dir]
|-- ground_truth
|   |-- wider_easy_val.mat
|   |-- wider_face_val.mat
|   |-- wider_hard_val.mat
|   `-- wider_medium_val.mat
`-- wider_face_split
    |-- readme.txt
    |-- wider_face_test.mat
    |-- wider_face_test_filelist.txt
    |-- wider_face_train.mat
    |-- wider_face_train_bbx_gt.txt
    |-- wider_face_val.mat
    `-- wider_face_val_bbx_gt.txt
```
* FDDB  
Download dataset from [FDDB](http://vis-www.cs.umass.edu/fddb/), and also download and compile the [official evaluation code](http://vis-www.cs.umass.edu/fddb/evaluation.tgz).  
Then merge files in `FDDB-folds` by running:
```
cd FDDB-folds
for i in `seq -f "%02g" 01 10`; do cat FDDB-fold-${i}.txt >> val.txt; done
for i in `seq -f "%02g" 01 10`; do cat FDDB-fold-${i}-ellipseList.txt >> val_gt.txt; done
```
```
.
├── 2002
│   ├── 07 [13 entries exceeds filelimit, not opening dir]
│   ├── 08 [31 entries exceeds filelimit, not opening dir]
│   ├── 09 [30 entries exceeds filelimit, not opening dir]
│   ├── 10 [31 entries exceeds filelimit, not opening dir]
│   ├── 11 [30 entries exceeds filelimit, not opening dir]
│   └── 12 [30 entries exceeds filelimit, not opening dir]
├── 2003
│   ├── 01 [21 entries exceeds filelimit, not opening dir]
│   ├── 02 [28 entries exceeds filelimit, not opening dir]
│   ├── 03 [31 entries exceeds filelimit, not opening dir]
│   ├── 04 [27 entries exceeds filelimit, not opening dir]
│   ├── 05 [30 entries exceeds filelimit, not opening dir]
│   ├── 06 [30 entries exceeds filelimit, not opening dir]
│   ├── 07 [31 entries exceeds filelimit, not opening dir]
│   ├── 08 [30 entries exceeds filelimit, not opening dir]
│   └── 09
│       ├── 01
│       │   └── big [39 entries exceeds filelimit, not opening dir]
│       └── 02
│           └── big
│               ├── img_38.jpg
│               └── img_44.jpg
├── evaluation [27 entries exceeds filelimit, not opening dir]
└── FDDB-folds [22 entries exceeds filelimit, not opening dir]
```
* Pascal Faces  
Image list is at `$ROOT/data/pascal_img_list.txt`.
```
.
├── images [851 entries exceeds filelimit, not opening dir]
└── pascal_img_list.txt
```
* AFW
Image list is at `$ROOT/data/afw_img_list.txt`.
```
.
├── 1004109301.jpg
├── 1051618982.jpg
├── ...
└── afw_img_list.txt
```

## Evaluate on WIDER FACE val with [trained model](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EV4rn_lxo45Lj5VYEwljqncBIhwbJns4zTJ_BHwjwPa05g?e=hR6QnT)
```
export WIDERFACEPATH=/path/to/your/wider/face/dataset
export MODELPATH=/path/to/your/caffemodel/final.caffemodel
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_test.py --train false --conf configs/smallhardface.toml --amend TEST.MODEL $MODELPATH DATA_DIR $WIDERFACEPATH
```
The result will be stored in `$ROOT/output/face/wider_val/face_$TIME` where `$TIME` is the timestamp when the code begins to run. The APs computed by our unofficial python code `lib/wider_eval_tools/wider_eval.py` can be found at the end of `$ROOT/output/face/wider_val/face_$TIME/stderr.log`. For APs computed by official MATLAB code, please evaluate `$ROOT/output/face/wider_val/face_$TIME/result.tar.gz` with [official evaluation toolkit](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip).

## Evaluate on FDDB with [trained model](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EV4rn_lxo45Lj5VYEwljqncBIhwbJns4zTJ_BHwjwPa05g?e=hR6QnT)
```
export FDDBPATH=/path/to/your/fddb/dataset
export MODELPATH=/path/to/your/caffemodel/final.caffemodel
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_test.py --train false --conf configs/smallhardface-fddb.toml --amend TEST.MODEL $MODELPATH DATA_DIR $FDDBPATH
```
The result will be stored in `$ROOT/output/face/fddb_val/face_$TIME` where `$TIME` is the timestamp when the code begins to run. The TPR@1000 can be found at the end of `$ROOT/output/face/fddb_val/face_$TIME/stderr.log`. To plot TPR curve, copy `rect_DiscROC.txt` into `$ROOT/external/marcopede-face-eval-f2870fd85d48/detections/fddb` and run `$ROOT/external/marcopede-face-eval-f2870fd85d48/plot_AP_fddb.py`.

## Evaluate on Pascal Faces with [trained model](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EV4rn_lxo45Lj5VYEwljqncBIhwbJns4zTJ_BHwjwPa05g?e=hR6QnT)
```
export PASCALPATH=/path/to/your/pascal/dataset
export MODELPATH=/path/to/your/caffemodel/final.caffemodel
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_test.py --train false --conf configs/smallhardface-pascal.toml --amend TEST.MODEL $MODELPATH DATA_DIR $PASCALPATH
```
The result will be stored in `$ROOT/output/face/pascalface_val/face_$TIME` where `$TIME` is the timestamp when the code begins to run. To plot PR curve, copy `pascal_res.txt` into `$ROOT/external/marcopede-face-eval-f2870fd85d48/detections/PASCAL` and run `$ROOT/external/marcopede-face-eval-f2870fd85d48/plot_AP.py`.

## Evaluate on AFW with [trained model](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EV4rn_lxo45Lj5VYEwljqncBIhwbJns4zTJ_BHwjwPa05g?e=hR6QnT)
```
export AFWPATH=/path/to/your/afw/dataset
export MODELPATH=/path/to/your/caffemodel/final.caffemodel
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_test.py --train false --conf configs/smallhardface-afw.toml --amend TEST.MODEL $MODELPATH DATA_DIR $AFWPATH
```
The result will be stored in `$ROOT/output/face/afw_val/face_$TIME` where `$TIME` is the timestamp when the code begins to run. To plot PR curve, copy `afw_res.txt` into `$ROOT/external/marcopede-face-eval-f2870fd85d48/detections/AFW` and run `$ROOT/external/marcopede-face-eval-f2870fd85d48/plot_AP.py`.


## Train face detector on WIDER FACE train
[Link to the ImageNet pre-trained model](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EQYSHRgDH-BEoAkrl6rlDgwBPd08W5CJTeKc6BVbjZ3D9g?e=RE9xXK)
```
export WIDERFACEPATH=/path/to/your/wider/face/dataset
export MODELPATH=/path/to/your/pretrained/model.caffemodel
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_test.py --conf configs/smallhardface.toml --amend DATA_DIR $WIDERFACEPATH TRAIN.PRETRAINED $MODELPATH
```
The trained model will be stored in `$ROOT/output/face/wider_train/face_$TIME` where `$TIME` is the timestamp when the code begins to run.
