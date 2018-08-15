#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess:      TensorFlow Session
    :param vgg_path:  Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return:          Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes,
           reg_scale=1e-3, stddev_init=0.01):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out:  TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out:  TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out:  TF Tensor for VGG Layer 7 output
    :param num_classes:     Number of classes to classify
    :return:                The Tensor for the last layer of output
    """

    conv1x1_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                   padding="same", name="conv1x1_out",
                                   kernel_initializer= tf.random_normal_initializer(stddev=stddev_init), 
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))

    upscale1_out = tf.layers.conv2d_transpose(conv1x1_out, num_classes, 4,
                                              strides= (2, 2), 
                                              padding="same", name="upscale1_out",
                                              kernel_initializer= tf.random_normal_initializer(stddev=stddev_init),                                              
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))

    # respective VGG layer with an amount of `num_classes` filters
    vgg_layer4_out_match = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                        padding="same", name="vgg_layer4_out_match",
                                        kernel_initializer= tf.random_normal_initializer(stddev=stddev_init), 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))

    upscale1_skip_out = tf.add(vgg_layer4_out_match, upscale1_out, name="upscale1_skip_out")

    
    
    upscale2_out = tf.layers.conv2d_transpose(upscale1_skip_out, num_classes, 4,
                                              strides= (2, 2), 
                                              padding="same", name="upscale2_out",
                                              kernel_initializer= tf.random_normal_initializer(stddev=stddev_init),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    
    # respective VGG layer with an amount of `num_classes` filters
    vgg_layer3_out_match = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                            padding="same", name="vgg_layer3_out_match",
                                            kernel_initializer= tf.random_normal_initializer(stddev=stddev_init), 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    
    upscale2_skip_out = tf.add(upscale2_out, vgg_layer3_out_match, name="decoder_output")
    

    upscale3_out = tf.layers.conv2d_transpose(upscale2_skip_out, num_classes, 16,
                                              strides= (8, 8),
                                              padding="same", name="upscale3_out",
                                              kernel_initializer= tf.random_normal_initializer(stddev=stddev_init),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))

    return upscale3_out

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes,
             trainable_vars=[]):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer:    TF Tensor of the last layer in the neural network
    :param correct_label:    TF Placeholder for the correct label image
    :param learning_rate:    TF Placeholder for the learning rate
    :param num_classes:      Number of classes to classify
    :param trainable_vars:   List of trainable variables
    :return:                 Tuple of (logits, train_op, cross_entropy_loss)
    """

    # default to all trainable vars, if none are given
    if trainable_vars == []:
        trainable_vars = tf.trainable_variables()

    l2_loss = tf.losses.get_regularization_loss()
    logits  = tf.reshape(nn_last_layer, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_last_layer, labels=correct_label))
    loss = tf.add(cross_entropy_loss, l2_loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=trainable_vars)

    return logits, train_op, loss

#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss_op, input_image,
             correct_label, keep_prob, learning_rate, learning_rate_value, keep_prob_value, last_layer):
    """
    Train neural network and print out the loss during training.
    :param sess:                 TF Session
    :param epochs:               Number of epochs
    :param batch_size:           Batch size
    :param get_batches_fn:       Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op:             TF Operation to train the neural network
    :param loss_op:              TF Tensor for the amount of loss
    :param input_image:          TF Placeholder for input images
    :param correct_label:        TF Placeholder for label images
    :param keep_prob:            TF Placeholder for dropout keep probability
    :param learning_rate:        TF Placeholder for learning rate
    :param learning_rate_value:  Actual learning rate value
    :param keep_prob_value:      Actual keep probability value
    :param last_layer            Last layer in the network
    """

    num_classes = tf.shape(last_layer)[-1]
    #correct_label_reshape = tf.reshape(correct_label, (-1, num_classes))
    #last_layer_reshape = tf.reshape(last_layer, (-1, num_classes))

    #predicted_label = tf.argmax(tf.reshape(last_layer, (-1, 2)), axis=-1)
    #sparse_correct_label = tf.argmax(tf.reshape(correct_label, (-1, 2)), axis=-1)
    #mean_iou, update_op = tf.metrics.mean_iou(correct_label_reshape, last_layer_reshape, 2)
    #mean_iou, update_op = tf.metrics.mean_iou(sparse_correct_label, predicted_label, 2)               
    prediction   = tf.argmax(last_layer, axis=-1)
    ground_truth = tf.argmax(correct_label, axis=-1)
    mean_iou, update_op = tf.metrics.mean_iou(ground_truth, prediction, 2)

    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(1,epochs+1):
        print("==== Epoch {} ===".format(epoch))
        for img, label in get_batches_fn(batch_size):
            feed_dict = {keep_prob: keep_prob_value,
                         learning_rate: learning_rate_value,
                         input_image: img,
                         correct_label: label}
            #print("shape(last_layer): ", sess.run(tf.shape(correct_label), feed_dict=feed_dict))
            _, loss_result, _ = sess.run([train_op, loss_op, update_op], feed_dict=feed_dict)
            iou_result = sess.run(mean_iou)
            print("  Mean IoU = {:.5f}".format(iou_result))
            print("  Loss: {:.3f}".format(loss_result))

        saver.save(sess, './runs/semantic_segmentation_model.ckpt')    

#tests.test_train_nn(train_nn)



def run():
    num_classes = 2
    batch_size = 50
    epochs = 50
    learning_rate_value = 5e-4
    keep_prob_value = 0.8
    
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # TF placeholders
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name="correct_label")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    with tf.Session() as sess:
        
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        tvars = tf.trainable_variables()

        trainable_vars = [var for var in tvars if "conv1x1"   in var.name or 
                                                  "vgg_layer" in var.name or
                                                  "upscale"   in var.name or
                                                  "correct"   in var.name]
        
        #trainable_vars = tvars

        print("trainable vars:")
        for var in trainable_vars:
            print("\t{}".format(var))

        # Set-up optimizer
        logits, train_op, loss = optimize(last_layer, correct_label, learning_rate, num_classes, trainable_vars)
        
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn,
                 train_op, loss, image_input, correct_label,
                 keep_prob, learning_rate, learning_rate_value, keep_prob_value, last_layer)


        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
