import gc
import numpy as np
import torch
from torchvision import transforms as T
from atomai.trainers import EnsembleTrainer
from atomai.utils import extract_patches

from ..datasets.WSe2_defect_dataset import WSe2DefectDataset_VIA
from ..evaluate import eval_ensemble

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))


def collate_fn(batch):
    return tuple(zip(*batch))


def create_dataset(train_path=r"./datastores/WSe2/STM"):
    dataset = WSe2DefectDataset_VIA(train_path)
    dataset.set_crystal_batch("SL445")
    dataset.set_scan_size(50)

    return dataset


def gen_train_data(dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    images, masks, Z = dataset.load_all(dataloader)
    images_all, labels_all = extract_patches(
        images, masks, patch_size=256, num_patches=60
    )

    return images_all, labels_all


def read_train_data():
    images_all = np.load(
        "./datastores/WSe2/STM/WSe2-Defect-Training-Images_2023-05-01.npy"
    )
    labels_all = np.load(
        "./datastores/WSe2/STM/WSe2-Defect-Training-Labels_2023-05-01.npy"
    )

    return images_all, labels_all


def split_train_data(images_all, labels_all):
    # total dataset size
    n = len(images_all)
    # validation split size
    v = int(0.1 * n) if n < 1000 else 100
    # test split size
    m = int(0.33 * (n - v))

    images_train, labels_train = (
        images_all[0 : n - m - v],
        labels_all[0 : n - m - v],
    )
    images_val, labels_val = (
        images_all[n - m - v : n - v],
        labels_all[n - m - v : n - v],
    )
    images_test, labels_test = images_all[n - v : n], labels_all[n - v : n]

    images_train = np.expand_dims(images_train, axis=1)
    images_val = np.expand_dims(images_val, axis=1)

    return (
        (images_train, labels_train),
        (images_val, labels_val),
        (images_test, labels_test),
    )


def train_ensemble(n_models=10, regen_train_data=False, n_epochs=500):
    if regen_train_data:
        dataset = create_dataset()
        images, labels = gen_train_data(dataset)
    else:
        images, labels = read_train_data()

    train_data, val_data, test_data = split_train_data(images, labels)

    images_train, labels_train = train_data
    images_val, labels_val = val_data
    images_test, labels_test = test_data

    # Ititialize and compile ensemble trainer
    lr_scheduler = [1e-3]
    etrainer = EnsembleTrainer("Unet", nb_classes=3)
    etrainer.compile_ensemble_trainer(
        training_cycles=n_epochs, compute_accuracy=True, swa=True
    )

    # Train ensemble of n models starting every time with new randomly initialized weights
    smodel, ensemble = etrainer.train_ensemble_from_scratch(
        images_train, labels_train, images_val, labels_val, n_models=n_models
    )

    eval_ensemble(smodel, ensemble, images_test, labels_test)

    # explicitly release unreferenced memory
    del images_train
    del labels_train
    del images_val
    del labels_val
    del images_test
    del labels_test
    gc.collect()

    return smodel, ensemble, etrainer
