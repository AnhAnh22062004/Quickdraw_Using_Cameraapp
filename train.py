import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
import os
import shutil
import argparse
from src.model import create_model, CLASS_IDS

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Optional, để đồng nhất số học

def get_args():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("-o", "--optimier", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=1024)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("--data_path", default="data", type=str)
    parser.add_argument("--log_path", default="data/tensorboard", type=str)
    parser.add_argument("--saved_path", default="data/trained_models", type=str)
    return parser.parse_args()

# Scale images & map label
def scale(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.squeeze(tf.where(tf.math.equal(label, tf.constant(list(CLASS_IDS.keys()), dtype=tf.int64))), axis=0)
    return image, label

def main(opt):
    # GPU memory growth (TF2-style)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Clean and prepare paths
    for path in [opt.log_path, opt.saved_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # Load and prepare dataset
    filter_fn = lambda image, label: tf.reduce_any(tf.math.equal(label, tf.constant(list(CLASS_IDS.keys()), dtype=tf.int64)))
    
    train_dataset = tfds.load("quickdraw_bitmap", split="train[:80%]", as_supervised=True, data_dir=opt.data_path)
    train_dataset = train_dataset.filter(filter_fn).map(scale).shuffle(opt.batch_size * 100).batch(opt.batch_size)

    test_dataset = tfds.load("quickdraw_bitmap", split="train[80%:]", as_supervised=True, data_dir=opt.data_path)
    test_dataset = test_dataset.filter(filter_fn).map(scale).batch(opt.batch_size)

    # Multi-GPU support
    if len(gpus) < 2:
        model = create_model()
    else:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = create_model()

    # Callbacks
    tensorboard_callback = TensorBoard(log_dir=opt.log_path, histogram_freq=1, write_graph=True, write_images=True)

    def schedule(epoch):
        if epoch < opt.epochs / 2:
            return opt.learning_rate
        elif epoch < opt.epochs * 0.8:
            return opt.learning_rate / 10
        else:
            return opt.learning_rate / 100

    lr_schedule_callback = LearningRateScheduler(schedule)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(opt.saved_path, "model.h5"), save_best_only=True)

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.learning_rate
            print(f"\nLearning rate for epoch {epoch + 1} is {tf.keras.backend.get_value(lr)}")

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate) if opt.optimier == "adam" else tf.keras.optimizers.SGD(learning_rate=opt.learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(train_dataset,
              validation_data=test_dataset,
              epochs=opt.epochs,
              callbacks=[tensorboard_callback, lr_schedule_callback, checkpoint_callback, PrintLR()])

if __name__ == "__main__":
    opt = get_args()
    main(opt)
