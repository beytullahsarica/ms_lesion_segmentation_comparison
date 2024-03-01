# -*- coding: utf-8 -*-
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras
from tensorflow.keras.callbacks import Callback
from tensorflow.python.client import device_lib


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nClearMemory on_epoch_end....")
        gc.collect()
        tensorflow.keras.backend.clear_session()


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def device_info():
    devices = device_lib.list_local_devices()
    for d in devices:
        t = d.device_type
        name = d.physical_device_desc
        l = [item.split(':', 1) for item in name.split(", ")]
        name_attr = dict([x for x in l if len(x) == 2])
        dev = name_attr.get('name', 'Unnamed device')
        print(f" {d.name} || {dev} || {t} || {sizeof_fmt(d.memory_limit)}")


def format_duration(duration):
    mapping = [
        ('s', 60),
        ('m', 60),
        ('h', 24),
    ]
    duration = int(duration)
    result = []
    for symbol, max_amount in mapping:
        amount = duration % max_amount
        result.append(f'{amount}{symbol}')
        duration //= max_amount
        if duration == 0:
            break

    if duration:
        result.append(f'{duration}d')

    return ' '.join(reversed(result))


def plot_history(history, save_name=None):
    # Plot training & validation iou_score values
    plt.figure(figsize=(24, 5))
    plt.subplot(131)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    # Plot training & validation loss values
    plt.subplot(132)
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.title('Learning curve')
    plt.ylabel('(Dice + Binary Focal) Loss')
    plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper right')
    plt.legend()
    # Plot training & validation f1-score values
    plt.subplot(133)
    plt.plot(history.history['f1-score'])
    plt.plot(history.history['val_f1-score'])
    plt.title('Model Dice/F1 score')
    plt.ylabel('Dice/F1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    # save and show
    plt.savefig(os.path.join("training_output", save_name + "_loss_iou_f1_score.png"))
    plt.show()


def check_version():
    print("tensorflow version: ", tensorflow.__version__)
    print("keras version: ", tensorflow.keras.__version__)


def check_device():
    device_name = tensorflow.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    pass


if __name__ == "__main__":
    main()
