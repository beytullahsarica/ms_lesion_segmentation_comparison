# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import time
from random import seed
from time import strftime

import numpy as np

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import tensorflow.keras
from augmentation import *
from dataloader import DataLoader
from dataset import MSDataset
from keras_unet_collection import models
from keras_unet_collection import losses
from sklearn.model_selection import train_test_split
from metrics import recall, precision
from model_types import ModelType
from dataset_type import DatasetType
from utils import get_available_devices, device_info, plot_history, ClearMemory, format_duration, check_device, check_version, make_directory

sm.set_framework('tf.keras')
sm.framework()

tensorflow.keras.backend.set_image_data_format('channels_last')

seed(2022)
img_channel = 3
mask_channel = 1

img_w = 224
img_h = 224
dataset_root_path = "dataset"
BACKBONE = 'resnet34'
n_classes = 1
preprocess_input = sm.get_preprocessing(BACKBONE)


def hybrid_loss(y_true, y_pred):
    loss_focal_tversky = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4 / 3)
    loss_dice = losses.dice(y_true, y_pred)
    return loss_focal_tversky + loss_dice


def compile_model(model=None):
    optim = tensorflow.keras.optimizers.Adam(0.0001)
    total_loss = hybrid_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), precision, recall]
    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
    return model


def get_unet_2d():
    model = models.unet_2d((img_w, img_h, img_channel),
                           [64, 128, 256, 512, 1024], n_labels=1,
                           stack_num_down=2, stack_num_up=1,
                           activation='ReLU', output_activation='Sigmoid',
                           batch_norm=True, pool='max', unpool='nearest', name='unet')
    return compile_model(model)


def get_vnet_2d():
    model = models.vnet_2d((img_w, img_h, img_channel),
                           filter_num=[16, 32, 64, 128, 256],
                           n_labels=1,
                           res_num_ini=1, res_num_max=3,
                           activation='PReLU', output_activation='Sigmoid',
                           batch_norm=True, pool=False, unpool=False, name='vnet')
    return compile_model(model)


def get_r2_unet_2d():
    model = models.r2_unet_2d((224, 224, 3), [64, 128, 256, 512], n_labels=1,
                              stack_num_down=2, stack_num_up=1, recur_num=2,
                              activation='ReLU', output_activation='Sigmoid',
                              batch_norm=True, pool='max', unpool='nearest', name='r2unet')
    return compile_model(model)


def get_att_unet_2d():
    model = models.att_unet_2d((img_w, img_h, img_channel),
                               filter_num=[64, 128, 256, 512, 1024],
                               n_labels=1,
                               stack_num_down=2, stack_num_up=2, activation='ReLU',
                               atten_activation='ReLU', attention='add', output_activation='Sigmoid',
                               batch_norm=True, pool=False, unpool=False,
                               name='attunet')

    return compile_model(model)


def get_trans_unet_2d():
    model = models.transunet_2d((img_w, img_h, img_channel),
                                filter_num=[32, 64, 128, 256],
                                n_labels=1, stack_num_down=2, stack_num_up=2,
                                embed_dim=768, num_mlp=512, num_heads=6, num_transformer=6,
                                activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid',
                                batch_norm=True, pool=True, unpool='bilinear', name='transunet')

    return compile_model(model)


def get_swin_unet_2d():
    model = models.swin_unet_2d((img_w, img_h, img_channel),
                                filter_num_begin=64, n_labels=1,
                                depth=4, stack_num_down=2, stack_num_up=2,
                                patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512,
                                output_activation='Sigmoid', shift_window=True, name='swin_unet')

    return compile_model(model)


def get_and_compile_model(model_type=ModelType.UNET_2D):
    if model_type == ModelType.UNET_2D:
        return get_unet_2d()
    elif model_type == ModelType.VNET_2D:
        return get_vnet_2d()
    elif model_type == ModelType.TRANSUNET_2D:
        return get_trans_unet_2d()
    elif model_type == ModelType.SWIN_UNET_2D:
        return get_swin_unet_2d()
    elif model_type == ModelType.ATT_UNET_2D:
        return get_att_unet_2d()
    elif model_type == ModelType.R2_UNET_2D:
        return get_r2_unet_2d()
    else:
        return get_unet_2d()


def get_isbi2015_dataset_path():
    rater_1_train_image_path = os.path.join(dataset_root_path, "isbi2015", "rater1_images.npy")
    rater_1_train_label_path = os.path.join(dataset_root_path, "isbi2015", "rater1_masks.npy")

    rater_2_train_image_path = os.path.join(dataset_root_path, "isbi2015", "rater2_images.npy")
    rater_2_train_label_path = os.path.join(dataset_root_path, "isbi2015", "rater2_masks.npy")

    return rater_1_train_image_path, rater_1_train_label_path, rater_2_train_image_path, rater_2_train_label_path


def get_msseg2016_dataset_path():
    train_image_path_msseg = os.path.join("dataset", "msseg2016", "train_images.npy")
    train_label_path_msseg = os.path.join("dataset", "msseg2016", "train_masks.npy")
    return train_image_path_msseg, train_label_path_msseg


def get_data_all(validation_split_size_percentage=0.10):
    train_image_path_msseg, train_label_path_msseg = get_msseg2016_dataset_path()

    train_images_msseg = np.load(train_image_path_msseg).astype(np.float32)
    train_labels_msseg = np.load(train_label_path_msseg).astype(np.float32)

    rater_1_train_image_path, rater_1_train_label_path, rater_2_train_image_path, rater_2_train_label_path = get_isbi2015_dataset_path()

    rater_1_train_images = np.load(rater_1_train_image_path).astype(np.float32)
    rater_1_train_labels = np.load(rater_1_train_label_path).astype(np.float32)

    rater_2_train_images_1 = np.load(rater_2_train_image_path).astype(np.float32)
    rater_2_train_labels_1 = np.load(rater_2_train_label_path).astype(np.float32)

    train_img = np.concatenate((rater_1_train_images, rater_2_train_images_1, train_images_msseg))
    label_img = np.concatenate((rater_1_train_labels, rater_2_train_labels_1, train_labels_msseg))

    train_img = train_img.reshape(len(train_img), img_w, img_h, img_channel)
    label_img = label_img.reshape(len(label_img), img_w, img_h, mask_channel)

    train_images, valid_images, train_labels, valid_labels = train_test_split(train_img, label_img,
                                                                              test_size=validation_split_size_percentage,
                                                                              random_state=2022)
    train_images = preprocess_input(train_images)
    train_labels = preprocess_input(train_labels)
    valid_images = preprocess_input(valid_images)
    valid_labels = preprocess_input(valid_labels)

    print(f"Size of train images: {train_images.shape}")
    print(f"Size of train labels: {train_labels.shape}")
    print(f"Size of valid images: {valid_images.shape}")
    print(f"Size of valid labels: {valid_labels.shape}")

    return train_images, train_labels, valid_images, valid_labels


def get_data_msseg(validation_split_size_percentage=0.10):
    train_image_path, train_label_path = get_msseg2016_dataset_path()

    print(f"Train image path: {train_image_path}")
    print(f"Train label path: {train_label_path}")

    train_img = np.load(train_image_path).astype(np.float32)
    label_img = np.load(train_label_path).astype(np.float32)

    train_img = train_img.reshape(len(train_img), img_w, img_h, img_channel)
    label_img = label_img.reshape(len(label_img), img_w, img_h, mask_channel)

    print(f"Size of train_img: {train_img.shape}")
    print(f"Size of label_img: {label_img.shape}")

    train_images, valid_images, train_labels, valid_labels = train_test_split(train_img, label_img,
                                                                              test_size=validation_split_size_percentage,
                                                                              random_state=2022)

    train_images = preprocess_input(train_images)
    train_labels = preprocess_input(train_labels)
    valid_images = preprocess_input(valid_images)
    valid_labels = preprocess_input(valid_labels)

    print(f"Size of train images: {train_images.shape}")
    print(f"Size of train labels: {train_labels.shape}")
    print(f"Size of valid images: {valid_images.shape}")
    print(f"Size of valid labels: {valid_labels.shape}")

    return train_images, train_labels, valid_images, valid_labels


def get_data_isbi(validation_split_size_percentage=0.10):
    rater_1_train_image_path, rater_1_train_label_path, rater_2_train_image_path, rater_2_train_label_path = get_isbi2015_dataset_path()

    print(f"rater_1_train_image_path: {rater_1_train_image_path}")
    print(f"rater_1_train_label_path: {rater_1_train_label_path}")
    print(f"rater_2_train_image_path: {rater_2_train_image_path}")
    print(f"rater_2_train_label_path: {rater_2_train_label_path}")

    rater_1_train_images = np.load(rater_1_train_image_path).astype(np.float32)
    rater_1_train_labels = np.load(rater_1_train_label_path).astype(np.float32)

    rater_2_train_images = np.load(rater_2_train_image_path).astype(np.float32)
    rater_2_train_labels = np.load(rater_2_train_label_path).astype(np.float32)

    print(f"Size of rater 1 train images: {rater_1_train_images.shape}")
    print(f"Size of rater 1 train labels: {rater_1_train_labels.shape}")
    print(f"Size of rater 2 train images: {rater_2_train_images.shape}")
    print(f"Size of rater 2 train labels: {rater_2_train_labels.shape}")

    train_img = np.concatenate((rater_1_train_images, rater_2_train_images))
    label_img = np.concatenate((rater_1_train_labels, rater_2_train_labels))

    train_img = train_img.reshape(len(train_img), img_w, img_h, img_channel)
    label_img = label_img.reshape(len(label_img), img_w, img_h, mask_channel)

    print(f"Size of train_img: {train_img.shape}")
    print(f"Size of label_img: {label_img.shape}")

    train_images, valid_images, train_labels, valid_labels = train_test_split(train_img, label_img,
                                                                              test_size=validation_split_size_percentage,
                                                                              random_state=2022)
    train_images = preprocess_input(train_images)
    train_labels = preprocess_input(train_labels)
    valid_images = preprocess_input(valid_images)
    valid_labels = preprocess_input(valid_labels)

    print(f"Size of train images: {train_images.shape}")
    print(f"Size of train labels: {train_labels.shape}")
    print(f"Size of valid images: {valid_images.shape}")
    print(f"Size of valid labels: {valid_labels.shape}")

    return train_images, train_labels, valid_images, valid_labels


def get_callbacks(model_name, best_modal_name):
    tensorboard_log_path = os.path.join("training_output/logs/", model_name)
    make_directory(tensorboard_log_path)
    log_dir = os.path.join(tensorboard_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return [
        ClearMemory(),
        tensorboard_callback,
        tensorflow.keras.callbacks.EarlyStopping(patience=60, verbose=1, monitor='val_loss', mode="min"),
        tensorflow.keras.callbacks.ModelCheckpoint('training_output/' + best_modal_name, monitor='val_loss', verbose=1,
                                                   save_best_only=True, mode='min'),
        tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
                                                     cooldown=10, min_lr=1e-5),
    ]


def start_train(dataset_type: DatasetType, model_type: ModelType = ModelType.UNET_2D, batch_size: int = 8, epochs: int = 10,
                validation_split_size_percentage: float = 0.10):
    if dataset_type == DatasetType.MSSEG2016:
        train_images, train_labels, valid_images, valid_labels = get_data_msseg(validation_split_size_percentage=validation_split_size_percentage)
    elif dataset_type == DatasetType.ISBI2015:
        train_images, train_labels, valid_images, valid_labels = get_data_isbi(validation_split_size_percentage=validation_split_size_percentage)
    else:
        train_images, train_labels, valid_images, valid_labels = get_data_all(validation_split_size_percentage=validation_split_size_percentage)

    train_dataset = MSDataset(train_images=train_images,
                              train_labels=train_labels,
                              augmentation=get_training_augmentation(),
                              preprocessing=get_preprocessing(preprocess_input))

    valid_dataset = MSDataset(train_images=valid_images,
                              train_labels=valid_labels,
                              augmentation=get_validation_augmentation(),
                              preprocessing=get_preprocessing(preprocess_input))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    model_name = model_type.value
    model = get_and_compile_model(model_type=model_type)
    model.summary()

    if dataset_type == DatasetType.MSSEG2016:
        # transfer learning from isbi2015
        isbi_weight_path = os.path.join("training_output", f"{DatasetType.ISBI2015.value}_{model_name}_best_model_all.h5")
        if os.path.exists(isbi_weight_path):
            print(f"isbi_weight_path: {isbi_weight_path}")
            model.load_weights(isbi_weight_path)
        else:
            print(f"Error: File '{isbi_weight_path}' not found. ISBI2015 pretained weights could not be loaded.")

    save_name = f"{dataset_type.value}_{model_name}"
    best_modal_name = f"{dataset_type.value}_{model_name}_best_model_all.h5"
    model_all_json = f"{dataset_type.value}_{model_name}_model_all.json"
    model_history = f"{dataset_type.value}_{model_name}_model_history.npy"

    model_json = model.to_json()
    with open(os.path.join('training_output', model_all_json), 'w') as json_file:
        json_file.write(model_json)

    time_start = time.time()
    print(f"start time at {strftime('%m/%d/%Y, %H:%M:%S', time.gmtime(time_start))}")
    print(f"model type: {model_type}")
    print("using dataloader...")
    history = model.fit(train_dataloader,
                        steps_per_epoch=len(train_dataloader),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=get_callbacks(model_name, best_modal_name),
                        validation_data=valid_dataloader,
                        validation_steps=len(valid_dataloader),
                        verbose=1)

    end_time = time.time()
    print(f"end time at {strftime('%m/%d/%Y, %H:%M:%S', time.gmtime(end_time))}")
    print(f"elapsed training duration: {format_duration(end_time - time_start)}")

    np.save(os.path.join('training_output', model_history), history.history)
    plot_history(history, save_name)


def main():
    # python train.py --dataset_type isbi2015 --model_type unet_2d --validation_split_percentage 0.2 --batch_size 8 --epochs 300

    parser = argparse.ArgumentParser(description='Training models for MS lesion segmentation: a comparison assessment')

    parser.add_argument('--dataset_type', type=str, choices=[dataset.value for dataset in DatasetType],
                        help='Dataset type to use: "isbi2015" or "msseg2016" or "all". Default is "isbi2015".',
                        default="isbi2015")

    parser.add_argument('--model_type', type=str, choices=[model.value for model in ModelType],
                        help='Name of the model to use: "unet_2d", "vnet_2d", "r2_unet_2d", "att_unet_2d", "transunet_2d", "swin_unet_2d". Default is "unet_2d".',
                        default="unet_2d")

    parser.add_argument('--validation_split_percentage', type=float,
                        help='Percentage of data to use for validation. Default is 10%.',
                        default=0.10)

    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training. Default size is 8.',
                        default=8)

    parser.add_argument('--epochs', type=int,
                        help='Number of epochs for training. Default is 10 epochs.',
                        default=10)

    args = parser.parse_args()
    try:
        dataset_type = DatasetType.from_str(args.dataset_type)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    try:
        model_type = ModelType.from_str(args.model_type)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    print("Parsed arguments:")
    print(f"Dataset Type: {dataset_type}")
    print(f"Train Model Type: {model_type}")
    print(f"Validation Split Percentage: {args.validation_split_percentage}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    check_version()
    check_device()
    device_info()

    print(get_available_devices())

    start_train(dataset_type=dataset_type, model_type=model_type, batch_size=args.batch_size, epochs=args.epochs,
                validation_split_size_percentage=args.validation_split_percentage)


if __name__ == "__main__":
    main()
