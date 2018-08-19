# Semantic Segmentation
### Introduction
In this project, pixels of a road images are labelled using a Fully Convolutional Network (FCN) based on VGG16.
It's a partial re-implementation of the network architecture proposed in [Shelhamer et al](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

The network consists of an encoder and decoder part. The encoder extracts features and encodes semantic information in a very compact representation within the "bottleneck" of the network by using only convolution layers. From the bottleneck the encoded semantic information is projected back on an image in the decoder part, which is done by upsampling that information to the original image size, also referred to as "transposed convolutions" in the deep-learning community.
In this example the final layer of the decoder network contains two filters, one for the road and one for otherwise. 

For evaluation of segmentation tasks a common accuracy metric is the mean intersection over union metric, it computes a ratio between the intersection and the union of two sets of pixels, where one set would be ground-truth and the other the prediction.


### Setup

#### Using a GPU
`main.py` will check to make sure you are using GPU, training the provided network without a GPU might take days and is therefore not encouraged. If you don't have a GPU on your system, you can use AWS or another cloud computing platform.

#### Frameworks and Packages
Make sure you have the following Python modules installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 
#### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip). Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.
For this purpose there is also a helper function (`maybe_download_pretrained_vgg()`) within the file `helper.py`.

#### Run the training
Run the following command to start the training:
```
python main.py 
```
or explicitly start in training mode:
```
python main.py --mode=train
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.
 
#### Run inference on test images
Call `main.py` with `--mode=test` in order to run the previously trained network on the test images located in `data/data_road/testing`, which also creates a new folder in `runs/` containing the test images with segementation applied.
```
python main.py --mode=test
```

### Implementation details



### Training results
Training only the added layers for the decoder network for 50 epochs:
* Mean IoU: 0.766
* Loss: 0.504

Training all layers of the encoder and decoder for 50 epochs:
* Mean IoU: 0.844
* Loss: 0.157

### Test results





### Project notes
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
