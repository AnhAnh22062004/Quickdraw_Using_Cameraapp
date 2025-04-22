import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import shutil
import argparse
from src.model import create_model, CLASS_IDS
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def get_args():
    parser = argparse.ArgumentParser(description="Argument")
    
    parser.add_argument("-o","--optimier", type=str, default="adam", choices= ["adam", "sgd"])
    parser.add_argument("-e", "--epochs", type=int, default= 10, help="number of total epochs to run model")
    parser.add_argument("-b", "--batch_size", type=int, default= 1024)
    parser.add_argument("-l","--learning_rate", type= float, default= 0.001, help="initial learning rate")
    parser.add_argument("--data_path", default="data", type= str)
    parser.add_argument("--log_path", default="data/tensorboard", type= str)
    parser.add_argument("--saved_path", default="data/trained_models", type=str)

    args = parser.parse_args()
    return args

def scale(image, label):
    image= tf.cast(image, tf.float32)
    image = image/255
    label = tf.squeeze(tf.where(tf.math.equal(label, tf.constant(list(CLASS_IDS.keys()), dtype=tf.int64))), axis= 0)

def main(opt):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    num_gpus= len(tf.config.list_physical_devices("GPU"))
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if os.path.isdir(opt.saved_path):
        shutil.rmtree(opt.saved_path)
    os.makedirs(opt.saved_path)
    
    train_dataset = tfds.load(name= "quickdraw_bitmap", as_supervised= True, data_dir= opt.data_path, split='train[:80%]')
    train_dataset = train_dataset.filter(
        lambda img, label: tf.reduce_any(
            tf.math.equal(label, tf.constant(list(CLASS_IDS.keys()), dtype=tf.int64)))).map(
        scale).shuffle(opt.batch_size * 100).batch(opt.batch_size)
    
    test_dataset = tfds.load(name="quickdraw_bitmap", as_supervised= True, data_dir=opt.data_path, split="train[80%:]")
    test_dataset= test_dataset.filter(
        lambda image, label:tf.reduce_any(
            tf.math.equal(label, tf.constant(list(CLASS_IDS.keys()), dtype= tf.int64)))).map(
        scale).batch(opt.batch_size)
    

    if num_gpus <2:
        model = create_model()
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy() # Distribute the model across multiple GPUs
        with mirrored_strategy.scope():
            model = create_model()
    
    tensorboard_callback = TensorBoard(log_dir= opt.log_path,update_freg="batch", histogram_freq= 1, write_graph= True, write_images= True) # Tensorboard callback to visualize the training process

    def schedule(epoch):
        if epoch < opt.epocks / 2:
            return opt.learning_rate
        elif opt.epochs/2 <= epoch < opt.epochs / 5:
            return opt.learning_rate / 10
        else:
            return opt.learning_rate / 100
        
    lr_schedule_callback = LearningRateScheduler(schedule)
    model_checkpoint_callbacks = ModelCheckpoint(filepath = opt.saved_path)
    class PrintLR(tf.keras.callbacks.Callback): # Custom callback to print the learning rate at the end of each epoch
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                              model.optimizer.lr.numpy()))
            
    callbacks = [tensorboard_callback, lr_schedule_callback, model_checkpoint_callbacks, PrintLR()]
    model.fit(train_dataset, epochs = opt.epochs, validation_data= test_dataset, callbacks= callbacks)

if __name__ == "__main__":
    opt = get_args()
    main(opt)
