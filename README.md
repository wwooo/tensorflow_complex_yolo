## Complex-YOLO  implementation in tensorflow
---
### Contents

[Overview](#overview)
[Examples](#examples)
[Dependencies](#dependencies)
[How to use it](#how-to-use-it)
[Others](#others)
[ToDo](#todo)

### Overview

The warehouse is an unofficial implementation of complex-yolo, and the model structure is slightly inconsistent with what the paper describes. [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199).Point cloud data preprocessing reference[AI-liu/Complex-YOLO](https://github.com/AI-liu/Complex-YOLO), Model structure reference[WojciechMormul/yolo2](https://github.com/WojciechMormul/yolo2).On this basis, a complete complex-yolo algorithm is implemented.

Complex-yolo takes point cloud data as input and encodes point cloud into RGB-map of bird 's-eye view to predict the position and yaw angle of objiects in 3d space.  In order to improve the efficiency of the training model, the point cloud data set is firstly made into RGB dataset.  The experiment is based on the kitti dataset. The kitti dataset has a total of 7481 tagged data. The dataset is divided into two parts, the first 1000 samples are used as test sets,  and the remaining samples are used as training sets.

### Examples

Below are some prediction examples of the Complex-Yolo， the predictions were made on  the splited test set.

| | |
|---|---|
|<div align="center"><img src="./examples/1.png" width="700" height="450" /></div>|<div align="center"><img src="./examples/2.png" width="700" height="450" /></div> |
| <div align="center"><img src="./examples/3.png" width="700" height="450" /></div> |  <div align="center"><img src="./examples/4.png" width="700" height="450" /></div>  |

### Dependencies

* Python 3.x
* Numpy
* TensorFlow 1.x
* OpenCV

### How to use it

How to prepare data:

1 . Download the data from the official website of kitti.

* [data_object_velodyne.zip](http://www.cvlibs.net/download.php?file=data_object_velodyne.zip)
* [data_object_label_2.zip](http://www.cvlibs.net/download.php?file=data_object_label_2.zip)
* [data_object_calib.zip](http://www.cvlibs.net/download.php?file=data_object_calib.zip)

2 . Create the following folder structure in the current working directory

./kitti/training/

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -velodyne/
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -label_２/
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -calib/
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-test.txt
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- train.txt
                         
 3 . Unzip the downloaded kitti dataset and get the following data. Place the data in the corresponding folder created above.
         
data_object_velodyne/trainin/\*.bin&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\*.bin ->  velodyne
data_object_label_2/training/label_2/\*.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\*.txt -> label_２
data_object_calib/training/calib/\*.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\*.txt -> calib

The test.txt and train.txt store the test and train sample index
test 	000000-000999
train 	001000-007480
 
Then create  RGB-image data set：
 
4 . Create the following folder structure in the current working directory
./test/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-images/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-labels/
           
 ./train/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-images/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-labels/
           
5 . run python make_image_dataset.py

 Note：This model only predicts the area of 60*80 in front of the car, and encodes the point cloud in this area into a 768 *1024 RGB-map. In the kitti data set, not all samples have objects in this area. Therefore, in the process of making an image dataset, the script will automatically filter out  samples of that doesn't  have objects  in the aera.
      
6 . run python make_train_test.py  will generate test_image_list.txt  and train_image_list.txt in the config folder.  This step is optional, the two  files already exist in config folder.
                        
How to train a model:
1 .  Adjust the training parameters in train.py according to the actual  situation.

 2 .   run python train.py

How to predict:

1 . run python predict.py

When running predict.py , directly use point cloud data as input to the model, and the script saves the predicted result in the predict_result folder.

How to eval:

run python kitti_eval.py

This script will save the prediction results consistent with the kitti label format. Then use kitti's official evaluation script to evaluate.

###Others
You can run  visualize_augumented_data.py to visualize the transformed  data and labels.
### ToDo

