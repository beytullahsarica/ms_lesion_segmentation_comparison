# -*- coding: utf-8 -*-
import tensorflow.keras.backend as K


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_out = true_positives / (possible_positives + K.epsilon())
    return recall_out


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_out = true_positives / (predicted_positives + K.epsilon())
    return precision_out


def main():
    pass


if __name__ == "__main__":
    main()
