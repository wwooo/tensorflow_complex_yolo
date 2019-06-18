## Complex-YOLO  implementation in tensorflow
---
### Contents

[Overview](#overview)<br>[Examples](#examples)<br>[Dependencies](#dependencies)<br>[How to use it](#how-to-use-it)<br>[Others](#others)<br>[ToDo](#todo)

### Overview

The project is an unofficial implementation of complex-yolo, and the model structure is slightly inconsistent with what the paper describes. [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199).Point cloud data preprocessing reference[AI-liu/Complex-YOLO](https://github.com/AI-liu/Complex-YOLO), model structure reference[WojciechMormul/yolo2](https://github.com/WojciechMormul/yolo2).On this basis, a complete complex-yolo algorithm is implemented.

Complex-yolo takes point cloud data as input and encodes point cloud into RGB-map of bird 's-eye view to predict the position and yaw angle of objiects in 3d space.  In order to improve the efficiency of  training model, the point cloud data set is firstly made into RGB dataset.  The experiment is based on the kitti dataset. The kitti dataset has a total of 7481 labeled data. The dataset is divided into two parts, the first 1000 samples are used as test sets,  and the remaining samples are used as training sets.

### Examples

Below are some prediction examples of the Complex-Yolo， the predictions were made on  the splited test set. The iou of car and cyclist are set to 0.5, 0.3 respectively.

| |  |
|---|---|
|<div align="center"><img src="https://github.com/wwooo/tensorflow_complex_yolo/blob/master/examples/1.png" width="500" height="350" /></div>|<div align="center"><img src="https://github.com/wwooo/tensorflow_complex_yolo/blob/master/examples/2.png" width="500" height="350" /></div> |
| <div align="center"><img src="https://github.com/wwooo/tensorflow_complex_yolo/blob/master/examples/3.png" width="500" height="350" /></div> |  <div align="center"><img src="https://github.com/wwooo/tensorflow_complex_yolo/blob/master/examples/4.png" width="500" height="350" /></div>  |
|<div align="center"><img src="https://github.com/wwooo/tensorflow_complex_yolo/blob/master/examples/car_detection_ground.png" width="500" height="350" /></div>|<div align="center"><img src="https://github.com/wwooo/tensorflow_complex_yolo/blob/master/examples/cyclist_detection_ground.png" width="500" height="350" /></div> |

### Dependencies

* Python 3.x
* Numpy
* TensorFlow 1.x
* OpenCV

### How to use it

Clone this repo
1. 
```bash
git clone https://github.com/wwooo/tensorflow_complex_yolo
```
2.

```bash
cd tensorflow_complex_yolo
```
How to prepare data:

1 . Download the data from the official website of kitti.

* [data_object_velodyne.zip](http://www.cvlibs.net/download.php?file=data_object_velodyne.zip)
* [data_object_label_2.zip](http://www.cvlibs.net/download.php?file=data_object_label_2.zip)
* [data_object_calib.zip](http://www.cvlibs.net/download.php?file=data_object_calib.zip)

2 . Create the following folder structure in the current working directory

```
tensorflow_complex_yolo
            training
                     calib
                     label_2
                     velodyne
```

                         
 3 . Unzip the downloaded kitti dataset and get the following data. Place the data in the corresponding folder created above.
  
data_object_velodyne/training/\*.bin&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\*.bin ->  velodyne

data_object_label_2/training/label_2/\*.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\*.txt -> label_２

data_object_calib/training/calib/\*.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\*.txt -> calib
 
Then create  RGB-image data set：

4 . run python utils/make\_image_dataset.py

 This script will convert the point cloud data into image data, which will be automatically saved in the  ./kitti/image_dataset/, and will generate test_image_list.txt  and train_image_list.txt in the ./config folder. 

 Note：This model only predicts the area of 60x80 in front of the car, and encodes the point cloud in this area into a 768 x1024 RGB-map. In the kitti data set, not all samples have objects in this area. Therefore, in the process of making  image dataset, the script will automatically filter out  samples of that doesn't  have objects  in the area.
                        
How to train a model:

1 .   run python train.py <br>--load\_weights <br> --batch\_size<br> --weights\_path<br> --gpu\_id<br>--num\_iter<br>--save\_interval<br>--save\_dir<br> <br> If you want to load model weights, set --load\_weights=True ,  default is False, you must provide the weights\_path. --num_iter, set the number of iterations. --save_interval, how many epochs to save the model,  default is 2 . --save\_dir,  where the model is saved, default is ./weights/ . Set  --gpu_id to specify which card to use for training, default is 0.

How to predict:

1 . run python predict.py <br>--weights\_path<br>--draw_gt_box

When running predict.py , directly use point cloud data as input to the model, and the script saves  predicted result in the predict\_result folder. You can set draw\_gt_box = True or False to decide whether to draw the ground truth box on  predicted result.

How to eval:

run python utils/kitti_eval.py

This script will save the prediction results consistent with the kitti label format. Then use kitti's official evaluation script to evaluate. You should study the official evaluation script of kitti.

### Others

You can run  utils/visualize_augumented_data.py to visualize the transformed  data and labels.

### ToDo

