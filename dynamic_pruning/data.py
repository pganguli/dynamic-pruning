"""Dataset loading for CIFAR-10/100, HAR, and KWS."""

import os
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

THIS_DIR = pathlib.Path(__file__).absolute().parent.parent

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2023, 0.1994, 0.2010)
GOOGLE_SPEECH_SAMPLE_RATE = 16000

__all__ = ["kws_dnn_model", "prepare_data", "GOOGLE_SPEECH_SAMPLE_RATE"]


def kws_dnn_model() -> pathlib.Path:
    # Clone https://github.com/ARM-software/ML-KWS-for-MCU into dnn-models/
    # to provide this file.
    path = (
        THIS_DIR
        / "dnn-models"
        / "ML-KWS-for-MCU"
        / "Pretrained_models"
        / "DNN"
        / "DNN_S.pb"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"KWS frozen graph not found at {path}.\n"
            "Clone the ARM ML-KWS-for-MCU repo into the dnn-models/ directory:\n"
            "  git clone https://github.com/ARM-software/ML-KWS-for-MCU "
            "dnn-models/ML-KWS-for-MCU"
        )
    return path


def _preprocess_kws_dataset(
    original_dataset: Any,  # noqa: ANN401 -- torchaudio dataset, an optional dep
) -> tuple[np.ndarray, list[int]]:
    # From https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Pretrained_models/labels.txt
    new_labels = "_silence_ _unknown_ yes no up down left right on off stop go".split(
        " "
    )

    import tensorflow as tf

    with open(kws_dnn_model(), "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    with tf.compat.v1.Session() as sess:
        mfcc_tensor = sess.graph.get_tensor_by_name("Mfcc:0")

        labels = []
        mfccs = None  # dimensions unknown until the first sample
        dataset_size = len(original_dataset)
        for n in range(dataset_size):
            # The first few _unknown_ samples aren't recognized by Hello
            # Edge's DNN model — use good ones instead, from the back.
            data = original_dataset[dataset_size - 1 - n]
            waveform, sample_rate, label, _, _ = data
            assert sample_rate == GOOGLE_SPEECH_SAMPLE_RATE
            decoded_wav = np.squeeze(waveform)
            # Some files in speech_commands_v0.02 are shorter than 1 second;
            # speech_commands_test_set_v0.02's counterparts pad with zeros.
            decoded_wav = np.pad(
                decoded_wav, ((0, GOOGLE_SPEECH_SAMPLE_RATE - len(decoded_wav)),)
            )
            decoded_wav = np.expand_dims(decoded_wav, axis=-1)

            # See https://github.com/tensorflow/datasets/blob/v4.6.0/tensorflow_datasets/audio/speech_commands.py#L128-L140
            if label in new_labels:
                label = new_labels.index(label)
            elif label in ("_silence_", "_background_noise_"):
                label = new_labels.index("_silence_")
            else:
                label = new_labels.index("_unknown_")
            labels.append(label)

            mfcc = sess.run(
                mfcc_tensor,
                {
                    "decoded_sample_data:0": decoded_wav,
                    "decoded_sample_data:1": GOOGLE_SPEECH_SAMPLE_RATE,
                },
            )
            if mfccs is None:
                mfccs_shape = list(mfcc.shape)
                mfccs_shape[0] = dataset_size
                mfccs = np.zeros(mfccs_shape)
            mfccs[n, :, :] = mfcc

    assert mfccs is not None, "original_dataset must be non-empty"
    return mfccs, labels


def _load_google_speech(train: bool) -> TensorDataset:
    import filelock
    import platformdirs
    import torchaudio  # noqa: F401 — torchaudio.datasets used below

    xdg_cache_home = platformdirs.user_cache_path()
    with filelock.FileLock(xdg_cache_home / "SpeechCommands.lock"):
        split = "train" if train else "test"
        cache_path = xdg_cache_home / f"SpeechCommands-cache-v1-{split}.pth"
        if cache_path.exists():
            mfccs, labels = torch.load(cache_path, weights_only=False)
        else:
            original_dataset = torchaudio.datasets.SPEECHCOMMANDS(
                root=xdg_cache_home,
                download=True,
                subset="training" if train else "testing",
            )
            mfccs, labels = _preprocess_kws_dataset(original_dataset)
            torch.save((mfccs, labels), cache_path)
    return TensorDataset(
        torch.from_numpy(np.expand_dims(mfccs.astype(np.float32), axis=1)),
        torch.tensor(labels),
    )


def _load_har(
    train_batch_size: int, test_batch_size: int
) -> tuple[DataLoader, DataLoader]:
    # Inspired by https://blog.csdn.net/bucan804228552/article/details/120143943
    har_utils_dir = THIS_DIR / "dnn-models" / "deep-learning-HAR" / "utils"
    orig_sys_path = sys.path.copy()
    try:
        sys.path.append(str(har_utils_dir))
        try:
            from utilities import read_data, standardize
        except ModuleNotFoundError:
            raise RuntimeError(
                f"HAR utilities not found at {har_utils_dir}.\n"
                "Copy the deep-learning-HAR/utils directory from the upstream project into\n"
                "  dnn-models/deep-learning-HAR/utils/\n"
                "so that dnn-models/deep-learning-HAR/utils/utilities.py exists."
            ) from None

        archive_dir = os.path.expanduser("~/.cache/UCI HAR Dataset")
        if not os.path.isdir(archive_dir):
            raise RuntimeError(
                f"UCI HAR Dataset not found at {archive_dir}.\n"
                "Download it from:\n"
                "  https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones\n"
                "Extract the zip so that ~/.cache/UCI HAR Dataset/ contains train/ and test/."
            )

        X_train_raw, train_labels, _ = read_data(archive_dir, split="train")
        _, X_train = standardize(X_train_raw, X_train_raw)
        trainset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(train_labels - 1),
        )
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

        X_test, test_labels, _ = read_data(archive_dir, split="test")
        _, X_test = standardize(X_train_raw, X_test)
        testset = TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)),
            torch.from_numpy(test_labels - 1),
        )
        testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)
        return trainloader, testloader
    finally:
        sys.path[:] = orig_sys_path


def _cifar_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ]
    )
    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    )
    return train_tf, test_tf


def prepare_data(
    dataset: str, train_batch_size: int, test_batch_size: int = 100
) -> tuple[DataLoader, DataLoader]:
    print("==> Preparing data..")

    if dataset in ("cifar10", "cifar100"):
        train_tf, test_tf = _cifar_transforms()
        cifar_cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
        root = f"./data/{dataset}"

        trainset = cifar_cls(root=root, train=True, download=True, transform=train_tf)
        trainloader = DataLoader(
            trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            multiprocessing_context="fork",
        )

        testset = cifar_cls(root=root, train=False, download=True, transform=test_tf)
        testloader = DataLoader(
            testset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            multiprocessing_context="fork",
        )
        return trainloader, testloader

    if dataset == "har":
        return _load_har(train_batch_size, test_batch_size)

    if dataset == "kws":
        trainloader = DataLoader(
            _load_google_speech(train=True), batch_size=train_batch_size, shuffle=True
        )
        testloader = DataLoader(
            _load_google_speech(train=False), batch_size=test_batch_size, shuffle=True
        )
        return trainloader, testloader

    raise ValueError(f"Unknown dataset {dataset!r}")
