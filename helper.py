import random
random.seed(0)

import torch
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gen_idx_byclass(labels):
    from collections import Counter
    classes = Counter(labels).keys()  # obtain a list of classes
    idx_byclass = {}

    for class_label in classes:
        # Find samples of this class:
        class_idx = []  # indices for samples that belong to this class
        for idx in range(len(labels)):
            if labels[idx] == class_label:
                class_idx.append(idx)
        idx_byclass[class_label] = class_idx

    return idx_byclass


def split_train_val_byclass(data, labels, train_ratio):
    idx_byclass = gen_idx_byclass(labels)
    train_idx = []
    train_labels = []
    val_idx = []
    val_labels = []

    for class_label in idx_byclass:
        # Process for this class:
        idx_thisclass = idx_byclass[class_label]
        random.shuffle(idx_thisclass)
        train_idx_thisclass = idx_thisclass[0 : int(train_ratio * len(idx_thisclass))]
        val_idx_thisclass = idx_thisclass[int(train_ratio * len(idx_thisclass)) : ]

        # Append:
        train_idx += train_idx_thisclass
        val_idx += val_idx_thisclass
        train_labels += [class_label] * len(train_idx_thisclass)
        val_labels += [class_label] * len(val_idx_thisclass)

    train_data = data[train_idx, :]
    val_data = data[val_idx, :]

    # Note that the labels are passed in as a Python list.
    # Here we turn it into a tensor and reshape as a column vector.
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    return train_data, train_labels, val_data, val_labels

