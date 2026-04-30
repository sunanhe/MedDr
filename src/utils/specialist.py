"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Helper functions.
"""
import torch
import numpy
import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import os
from PIL import Image
from torch.utils.data import Dataset
from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT
from medmnist import MedMNIST2D
from medmnist.evaluator import getACC, getAUC


class HAMMNIST(MedMNIST2D):
    flag = "ham10000"


class WCE(MedMNIST2D):
    flag = "wce"


class VinDrSpineXR(MedMNIST2D):
    flag = "vindr_spinexr"


class VinDrPCXR(MedMNIST2D):
    flag = "vindr_pcxr"


class PCam200(MedMNIST2D):
    flag = "PCam200"


class RetOCT(MedMNIST2D):
    flag = "RetOCT"


class BRSET(MedMNIST2D):
    flag = "BRSET"


class Derm7PT(MedMNIST2D):
    flag = "Derm7PT"


class Dermnet(MedMNIST2D):
    flag = "Dermnet"


class CBIS_DDSM_CALC(MedMNIST2D):
    flag = "CBIS_DDSM_CALC"


class CBIS_DDSM_MASS(MedMNIST2D):
    flag = "CBIS_DDSM_MASS"


class Mammo(MedMNIST2D):
    flag = "Mammo"


class MedFMCEndo(MedMNIST2D):
    flag = "MedFMCEndo"


class MedFMColon(MedMNIST2D):
    flag = "MedFMColon"


class MedFMChest(MedMNIST2D):
    flag = "MedFMChest"


class PCXR(MedMNIST2D):
    flag = "PCXR"

def seed_worker(worker_id):
    """Set the seed for the current worker."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_passed_time(start_time, end_time):
    """
    Calculate the time needed for running the code

    :param: start_time: Start time.
    :param: end_time: End time.
    :return: Duration in hh:mm:ss.ss
    """

    # Calculate the duration
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Return the duration in hours, minutes and seconds
    return int(hours), int(minutes), seconds


def extract_embeddings(model, device, dataloader):
    """
    Extracts the embeddings from the model.

    :param model: Model.
    :param device: Device.
    :param dataloader: Dataloader.
    """
    embeddings_db, labels_db = [], []

    with torch.no_grad():
        for extracted in tqdm(dataloader):
            images, labels = extracted
            images = images.to(device)

            output = model.forward_features(images)
            output = model.forward_head(output, pre_logits=True)
            output = output.reshape(output.shape[0], -1)

            labels_db.append(deepcopy(labels.cpu().numpy()))
            embeddings_db.append(deepcopy(output.cpu().numpy()))

    data = {
        'embeddings': np.concatenate(embeddings_db, axis=0),
        'labels': np.concatenate(labels_db, axis=0),
    }

    return data


def extract_features(model, device, dataloader):
    """
    Extracts the embeddings from the model.

    :param model: Model.
    :param device: Device.
    :param dataloader: Dataloader.
    """
    embeddings_db, labels_db = [], []

    with torch.no_grad():
        for extracted in tqdm(dataloader):
            images, labels = extracted
            images = images.to(device)

            output = model.forward_features(images)
            # output = model.forward_head(output, pre_logits=True)
            output = output.reshape(output.shape[0], -1)

            labels_db.append(deepcopy(labels.cpu().numpy()))
            embeddings_db.append(deepcopy(output.cpu().numpy()))

    data = {
        'embeddings': np.concatenate(embeddings_db, axis=0),
        'labels': np.concatenate(labels_db, axis=0),
    }

    return data


def extract_embeddings_densenet(model, device, dataloader):
    """
    Extracts the embeddings from the DenseNet.

    :param model: Model.
    :param device: Device.
    :param dataloader: Dataloader.
    """
    embeddings_db, labels_db = [], []

    with torch.no_grad():
        for extracted in tqdm(dataloader):
            images, labels = extracted
            images = images.to(device)

            output = model.forward_features(images)
            output = model.global_pool(output)
            output = output.reshape(output.shape[0], -1)

            labels_db.append(deepcopy(labels.cpu().numpy()))
            embeddings_db.append(deepcopy(output.cpu().numpy()))

    data = {
        'embeddings': np.concatenate(embeddings_db, axis=0),
        'labels': np.concatenate(labels_db, axis=0),
    }

    return data


def extract_features_densenet(model, device, dataloader):
    """
    Extracts the embeddings from the DenseNet.

    :param model: Model.
    :param device: Device.
    :param dataloader: Dataloader.
    """
    embeddings_db, labels_db = [], []

    with torch.no_grad():
        for extracted in tqdm(dataloader):
            images, labels = extracted
            images = images.to(device)

            output = model.forward_features(images)
            # output = model.global_pool(output)
            output = output.reshape(output.shape[0], -1)

            labels_db.append(deepcopy(labels.cpu().numpy()))
            embeddings_db.append(deepcopy(output.cpu().numpy()))

    data = {
        'embeddings': np.concatenate(embeddings_db, axis=0),
        'labels': np.concatenate(labels_db, axis=0),
    }

    return data


def extract_embeddings_alexnet(model, device, dataloader):
    """
    Extracts the embeddings from the AlexNet.

    :param model: Model.
    :param device: Device.
    :param dataloader: Dataloader.
    """
    embeddings_db, labels_db = [], []
    with torch.no_grad():
        for extracted in tqdm(dataloader):
            images, labels = extracted
            images = images.to(device)

            output = model(images)
            output = output.reshape(output.shape[0], -1)

            labels_db.append(deepcopy(labels.cpu().numpy()))
            embeddings_db.append(deepcopy(output.cpu().numpy()))

    data = {
        'embeddings': np.concatenate(embeddings_db, axis=0),
        'labels': np.concatenate(labels_db, axis=0),
    }
    return data


def extract_features_alexnet(model, device, dataloader):
    """
    Extracts the embeddings from the AlexNet.

    :param model: Model.
    :param device: Device.
    :param dataloader: Dataloader.
    """
    embeddings_db, labels_db = [], []
    old_classifier = model.classifier
    model.classifier = torch.nn.Sequential(*[model.classifier[i] for i in range(4)])
    with torch.no_grad():
        for extracted in tqdm(dataloader):
            images, labels = extracted
            images = images.to(device)

            output = model(images)
            output = output.reshape(output.shape[0], -1)

            labels_db.append(deepcopy(labels.cpu().numpy()))
            embeddings_db.append(deepcopy(output.cpu().numpy()))

    data = {
        'embeddings': np.concatenate(embeddings_db, axis=0),
        'labels': np.concatenate(labels_db, axis=0),
    }
    model.classifier = old_classifier
    return data


def knn_majority_vote(nbrs, embeddings, support_set_labels, task):
    """
    Finds the k nearest neighbors for each embedding in "embeddings" and performs a majority vote on their labels.
    """

    # Find the k nearest neighbors for each embedding in "embeddings"
    distances, indices = nbrs.kneighbors(embeddings)

    # For each set of k nearest neighbors, find the majority class
    outputs = []
    for neighbors in indices:
        if task == "multi-label, binary-class":
            classes = np.array([support_set_labels[i] for i in neighbors])
            majority_classes = np.mean(classes, axis=0) > 0.5  # Majority vote for each class
            outputs.append(majority_classes)

        else:
            classes = [support_set_labels[i] for i in neighbors]
            majority_class = stats.mode(classes)[0][0]
            outputs.append(majority_class)

    outputs = torch.from_numpy(np.array(outputs))
    outputs = outputs.reshape(outputs.shape[0], -1)

    return outputs


def knn_search(nbrs, embeddings, support_set_labels, top_n):
    """
    Finds the k nearest neighbors for each embedding in "embeddings" and performs a majority vote on their labels.
    """

    # Find the k nearest neighbors for each embedding in "embeddings"
    breakpoint()
    distances, indices = nbrs.kneighbors(normalize(embeddings), top_n)

    # For each set of k nearest neighbors, find the majority class
    outputs = []
    res_sims = torch.from_numpy(distances)
    res_train_ids = torch.from_numpy(indices)
    res_labels = []
    for neighbors in indices:
        _res_ls = []
        for neighbor in neighbors:
            _res_ls.append(support_set_labels[neighbor])
        res_labels.append(_res_ls)
    res_labels = torch.from_numpy(np.array(res_labels))
    # outputs = torch.from_numpy(np.array(outputs))
    # outputs = outputs.reshape(outputs.shape[0], -1)

    return res_sims, res_train_ids, res_labels


def get_ACC(y_true: np.ndarray, y_pred: np.ndarray, task: str):
    """
    Calculates the accuracy of the prediction.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param task: Type of classification task.
    :return: Accuracy
    """

    return getACC(y_true, y_pred, task)


def get_AUC(y_true: np.ndarray, y_pred: np.ndarray, task: str):
    """
    Calculates the Area-under-the-ROC curve of the prediction.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param task: Type of classification task.
    :return: AUC score.
    """

    return getAUC(y_true, y_pred, task)


def get_ACC_kNN(y_true: np.ndarray, y_pred: np.ndarray, task: str):
    """
    Calculates the accuracy of the prediction adapted for the kNN approach.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param task: Type of classification task.
    :return: Accuracy
    """

    if task == "multi-label, binary-class":
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pred[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]

    else:
        ret = accuracy_score(y_true, y_pred)

    return ret
