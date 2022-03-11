import json
import os

from keras_segmentation.data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
import six
from keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import sys

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

history = LossHistory()

####Adjust the Parameter####
model = "mobilenet_segnet"
train_images = "D:\Point2IMG\Original_Img\Train_IMG"
train_annotations = "D:\Point2IMG\Original_Img\Train_Label"
Model_Path = "D:\image-segmentation-keras\Model\Three_Class_model.h5"

input_height = 128
input_width = 128
n_classes = 6
channel_count = 3

weightArr = ([6, 5, 10, 14, 33, 50])
#weightArr = ([1, 1, 1, 1, 1, 1])

verify_dataset = False

checkpoints_path = "D:\image-segmentation-keras\Output\Three_Class_"

epochs = 4500
batch_size = 8

validate = False
val_images = "D:\Point2IMG\Original_Img\Test_IMG"
val_annotations = "D:\Point2IMG\Original_Img\Test_Label"
val_batch_size = 8

auto_resume_checkpoint=None 
load_weights=None

default_callback = ModelCheckpoint( 
            filepath=checkpoints_path + ".{epoch:05d}",
            save_weights_only=True, period = 2, verbose=True)
callbacks = [default_callback, tf.keras.callbacks.TensorBoard(log_dir='.\\logs'), history ]

####### Never Change########
steps_per_epoch=512
val_steps_per_epoch=512
gen_use_multiprocessing=False
ignore_zero_class=False
optimizer_name='adam'
do_augment=False
augmentation_name="aug_all"
custom_augmentation=None
other_inputs_paths=None
preprocessing=None
read_image_type=1  # cv2.IMREAD_COLOR = 1 (rgb),
                   # cv2.IMREAD_GRAYSCALE = 0,
                   # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)

############################




def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    # This is legacy code, there should always be a "checkpoint" file in your directory

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    if len(all_checkpoint_files) == 0:
        all_checkpoint_files = glob.glob(checkpoints_path + "*.*")
    all_checkpoint_files = [ff.replace(".index", "") for ff in
                            all_checkpoint_files]  # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))

    return latest_epoch_checkpoint

def weighted_categorical_crossentropy(weights):
    from keras import backend as K
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    #print(categorical_crossentropy(gt, pr) * mask)
    return categorical_crossentropy(gt, pr) * mask

class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


from keras_segmentation.models.all_models import model_from_name
# check if user gives model name instead of the model object
if isinstance(model, six.string_types):
    # create the model from the name
    print(input_height, input_width)
    assert (n_classes is not None), "Please provide the n_classes"
    if (input_height is not None) and (input_width is not None):
        model = model_from_name[model](
            n_classes, input_height=input_height, input_width=input_width, channels = channel_count)
    else:
        model = model_from_name[model](n_classes)

n_classes = model.n_classes
input_height = model.input_height
input_width = model.input_width
output_height = model.output_height
output_width = model.output_width

if validate:
    assert val_images is not None
    assert val_annotations is not None

if optimizer_name is not None:

    if ignore_zero_class:
        loss_k = masked_categorical_crossentropy
    else:
        loss_k = weighted_categorical_crossentropy(weightArr)#'categorical_crossentropy'

    model.compile(loss=loss_k,
                  optimizer=optimizer_name,
                  metrics=['accuracy'])

if checkpoints_path is not None:
    config_file = checkpoints_path + "_config.json"
    dir_name = os.path.dirname(config_file)

    if ( not os.path.exists(dir_name) )  and len( dir_name ) > 0 :
        os.makedirs(dir_name)

    with open(config_file, "w") as f:
        json.dump({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }, f)

if load_weights is not None and len(load_weights) > 0:
    print("Loading weights from ", load_weights)
    model.load_weights(load_weights)

initial_epoch = 0

if auto_resume_checkpoint and (checkpoints_path is not None):
    latest_checkpoint = find_latest_checkpoint(checkpoints_path)
    if latest_checkpoint is not None:
        print("Loading the weights from latest checkpoint ",
              latest_checkpoint)
        model.load_weights(latest_checkpoint)

        initial_epoch = int(latest_checkpoint.split('.')[-1])

if verify_dataset:
    print("Verifying training dataset")
    verified = verify_segmentation_dataset(train_images,
                                           train_annotations,
                                           n_classes)
    assert verified
    if validate:
        print("Verifying validation dataset")
        verified = verify_segmentation_dataset(val_images,
                                               val_annotations,
                                               n_classes)
        assert verified

train_gen = image_segmentation_generator(
    train_images, train_annotations,  batch_size,  n_classes,
    input_height, input_width, output_height, output_width,
    do_augment=do_augment, augmentation_name=augmentation_name,
    custom_augmentation=custom_augmentation, other_inputs_paths=other_inputs_paths,
    preprocessing=preprocessing, read_image_type=read_image_type)

if validate:
    val_gen = image_segmentation_generator(
        val_images, val_annotations,  val_batch_size,
        n_classes, input_height, input_width, output_height, output_width,
        other_inputs_paths=other_inputs_paths,
        preprocessing=preprocessing, read_image_type=read_image_type)
'''
if callbacks is None and (not checkpoints_path is  None) :
    default_callback = ModelCheckpoint(
            filepath=checkpoints_path + ".{epoch:05d}",
            save_weights_only=True,
            period = 5,
            verbose=True
        )

    if sys.version_info[0] < 3: # for pyhton 2 
        default_callback = CheckpointsCallback(checkpoints_path)

    callbacks = [
        default_callback
    ]'''

if callbacks is None:
    callbacks = []

if not validate:
    model.fit(train_gen, steps_per_epoch=steps_per_epoch,
              epochs=epochs, callbacks=callbacks, initial_epoch=initial_epoch)
    model.save(Model_Path)
else:
    model.fit(train_gen,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_gen,
              validation_steps=val_steps_per_epoch,
              epochs=epochs, callbacks=callbacks,
              use_multiprocessing=gen_use_multiprocessing, initial_epoch=initial_epoch)
    model.save(Model_Path)

history.loss_plot('epoch')
