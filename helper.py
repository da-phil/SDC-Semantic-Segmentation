import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        return sess.graph, ops


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def segment_image(sess, logits, keep_prob, image_input_op, image, image_shape):
    """
    Generate test output using the test images
    TODO: Get ground-truth data for test image, then loss and IoU can be computed

    :param sess:            TF session
    :param logits:          TF Tensor for the logits
    :param keep_prob:       TF Placeholder for the dropout keep robability
    :param image_input_op:  TF Placeholder for the image placeholder
    :param image:           Image as ND array
    :param image_shape:     Tuple - Shape of image
    :return:                Output for for each test image
    """

    image = scipy.misc.imresize(image, image_shape)
    """
    # inference with loss and mean IoU computation:
    mean_iou_value, mean_iou_update_op = tf.metrics.mean_iou(ground_truth, prediction, 2)
    feed_dict = {keep_prob: 1.0,
                 image_input_op: [image],
                 correct_label: [label]}
    im_softmax, loss_result, _  = sess.run([tf.nn.softmax(logits), loss, mean_iou_update_op],
                                           feed_dict=feed_dict)
    iou_result = sess.run(mean_iou_value)
    """

    # inference only with segmentation visualization
    feed_dict = {keep_prob: 1.0,
                 image_input_op: [image]}
    im_softmax = sess.run([tf.nn.softmax(logits)],
                          feed_dict=feed_dict)
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.asarray(street_im)


def save_inference_samples(runs_dir, model_checkpoint, data_dir, sess, image_shape):
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    try:
        saver.restore(sess, model_checkpoint)
    except:
        print("Couldn't load model last checkpoint ({}).".format(model_checkpoint))
        print("You need to either provide the required checkpoint files or train the network from scratch!")
        return

    input_image_op      = graph.get_tensor_by_name("image_input:0")
    logits_op           = graph.get_tensor_by_name("decoder_logits:0")
    keep_prob           = graph.get_tensor_by_name("keep_prob:0")
    loss                = graph.get_tensor_by_name("decoder_loss:0")
    correct_label       = graph.get_tensor_by_name("decoder_loss:0")
    prediction          = graph.get_tensor_by_name("decoder_prediction:0")
    ground_truth        = graph.get_tensor_by_name("decoder_ground_truth:0")

    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print("Saving test images to: {}".format(output_dir))
    data_folder = os.path.join(data_dir, "data_road/testing")
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imread(image_file)
        segmented_image = segment_image(sess, logits_op, keep_prob, input_image_op, image, image_shape)

        scipy.misc.imsave(os.path.join(output_dir, os.path.basename(image_file)), segmented_image)


def process_video_image(sess, logits, keep_prob, image_input_op, image_src, image_shape):
    # first crop away top of image to correct aspect of `image_shape` to match `image_shape`
    image_src_shape  = image_src.shape
    new_y = (image_shape[0] * image_src_shape[1]) // image_shape[1]
    image_crop = image_src[new_y:,:]

    return segment_image(sess, logits, keep_prob, image_input_op, image_crop, image_shape)


def save_inference_video_samples(sess, videos, model_checkpoint, video_fps, video_output_folder, image_shape):
    from moviepy.editor import VideoFileClip, vfx

    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    try:
        saver.restore(sess, model_checkpoint)
    except:
        print("Couldn't load model last checkpoint ({}).".format(model_checkpoint))
        print("You need to either provide the required checkpoint files or train the network from scratch!")
        return

    input_image_op  = graph.get_tensor_by_name("image_input:0")
    logits_op       = graph.get_tensor_by_name("decoder_logits:0")
    keep_prob       = graph.get_tensor_by_name("keep_prob:0")
    
    for video in videos:
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)
        result_path = video_output_folder + os.path.basename(video)
        if not os.path.isfile(video):
            print("Video {} doesn't exist!".format(video))
        else:
            clip1 = VideoFileClip(video) #.subclip(*clip_part)
            video_slowdown_factor = video_fps / clip1.fps
            clip1 = clip1.fx(vfx.speedx, video_slowdown_factor)
            white_clip = clip1.fl_image(lambda img: process_video_image(sess, logits_op, keep_prob, input_image_op, img, image_shape))
            white_clip.write_videofile(result_path, audio=False, fps=video_fps)