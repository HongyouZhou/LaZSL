import json
import numpy as np
import torch
import yaml
from torch.nn import functional as F

from descriptor_strings import *  # label_to_classname, wordify, modify_descriptor,
import pathlib

# ── Load config ──────────────────────────────────────────────
_cfg_path = pathlib.Path(__file__).parent / "config.yaml"
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

_DATA_ROOT = pathlib.Path(cfg["data_root"])
_ds = cfg["datasets"]

# ── CLI overrides ────────────────────────────────────────────
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--dataset", type=str, default=None)
_parser.add_argument("--model", type=str, default=None)
_parser.add_argument("--device", type=str, default=None)
_parser.add_argument("--output-dir", type=str, default=None)
_args, _ = _parser.parse_known_args()

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from torchvision.datasets import ImageFolder

from datasets import _transform, CUBDataset, random_crop
from collections import OrderedDict
from myclip import clip
from loading_helpers import *
from OP import *
from torchvision import datasets
from my_datasets import *

from utils import (
    imagenet_a_lt,
    imagenet_r_lt,
)


hparams = {}


def get_params(model, dataset):
    params = {}

    if model == "ViT-B/16":
        if dataset == "imagenet":
            params["alpha"] = 0.6
            params["theta"] = 0.8
            params["N"] = 90
        elif dataset == "cub":
            params["alpha"] = 0.6
            params["theta"] = 0.9
            params["N"] = 90
        elif dataset == "pets":
            params["alpha"] = 0.6
            params["theta"] = 0.2
            params["N"] = 80
        elif dataset == "food":
            params["alpha"] = 0.6
            params["theta"] = 0.8
            params["N"] = 90
        elif dataset == "place":
            params["alpha"] = 0.4
            params["theta"] = 0.8
            params["N"] = 60
        elif dataset == "imagenetv2":
            params["alpha"] = 0.5
            params["theta"] = 0.8
            params["N"] = 70
        elif dataset == "imagenet-r":
            params["alpha"] = 0.6
            params["theta"] = 0.8
            params["N"] = 90
        elif dataset == "imagenet-a":
            params["alpha"] = 0.5
            params["theta"] = 0.95
            params["N"] = 90
        elif dataset == "imagenet-s":
            params["alpha"] = 0.6
            params["theta"] = 0.8
            params["N"] = 80
    elif model == "ViT-B/32":
        if dataset == "imagenet":
            params["alpha"] = 0.6
            params["theta"] = 0.8
            params["N"] = 90
        elif dataset == "cub":
            params["alpha"] = 0.5
            params["theta"] = 0.95
            params["N"] = 80
        elif dataset == "pets":
            params["alpha"] = 0.6
            params["theta"] = 0.9
            params["N"] = 80
        elif dataset == "food":
            params["alpha"] = 0.6
            params["theta"] = 0.9
            params["N"] = 80
        elif dataset == "place":
            params["alpha"] = 0.6
            params["theta"] = 0.9
            params["N"] = 80
    elif model == "ViT-L/14":
        if dataset == "imagenet":
            params["alpha"] = 0.6
            params["theta"] = 0.8
            params["N"] = 70
        elif dataset == "cub":
            params["alpha"] = 0.5
            params["theta"] = 0.9
            params["N"] = 80
        elif dataset == "pets":
            params["alpha"] = 0.6
            params["theta"] = 0.8
            params["N"] = 60
        elif dataset == "food":
            params["alpha"] = 0.6
            params["theta"] = 0.9
            params["N"] = 70
        elif dataset == "place":
            params["alpha"] = 0.4
            params["theta"] = 0.9
            params["N"] = 70

    return params


# hyperparameters

hparams["model_size"] = _args.model or "ViT-B/16"
# Options:
# ['ViT-B/32',
#  'ViT-B/16',
#  'ViT-L/14']
hparams["dataset"] = _args.dataset or "imagenet"
params = get_params(hparams["model_size"], hparams["dataset"])
hparams["max_iter"] = 100
hparams["n_samples"] = params["N"]
# for mix
hparams["theta"] = params["theta"]
# for crop
hparams["alpha"] = params["alpha"]
# for constrain
hparams["gama"] = 0.0
hparams["constrain_type"] = "att"  # ['patch','att','const']


hparams["batch_size"] = 50
hparams["device"] = _args.device or cfg.get("device", "cuda:0")
hparams["output_dir"] = _args.output_dir
hparams["category_name_inclusion"] = "prepend"  #'append' 'prepend'

hparams["apply_descriptor_modification"] = True
hparams["is_cupl"] = False

hparams["verbose"] = False
hparams["image_size"] = 224
if hparams["model_size"] == "ViT-L/14@336px" and hparams["image_size"] != 336:
    print(
        f"Model size is {hparams['model_size']} but image size is {hparams['image_size']}. Setting image size to 336."
    )
    hparams["image_size"] = 336
elif hparams["model_size"] == "RN50x4" and hparams["image_size"] != 288:
    print(
        f"Model size is {hparams['model_size']} but image size is {hparams['image_size']}. Setting image size to 288."
    )
    hparams["image_size"] = 288
elif hparams["model_size"] == "RN50x16" and hparams["image_size"] != 384:
    print(
        f"Model size is {hparams['model_size']} but image size is {hparams['image_size']}. Setting image size to 288."
    )
    hparams["image_size"] = 384
elif hparams["model_size"] == "RN50x64" and hparams["image_size"] != 448:
    print(
        f"Model size is {hparams['model_size']} but image size is {hparams['image_size']}. Setting image size to 288."
    )
    hparams["image_size"] = 448

hparams["before_text"] = ""
hparams["label_before_text"] = ""
hparams["between_text"] = ", "
# hparams['between_text'] = ' '
# hparams['between_text'] = ''
hparams["after_text"] = ""
hparams["unmodify"] = True
# hparams['after_text'] = '.'
# hparams['after_text'] = ' which is a type of bird.'
hparams["label_after_text"] = ""
# hparams['label_after_text'] = ' which is a type of bird.'
hparams["seed"] = 1

# TODO: fix this... defining global variable to be edited in a function, bad practice
# unmodify_dict = {}

# classes_to_load = openai_imagenet_classes
hparams["descriptor_fname"] = None

IMAGENET_DIR = str(_DATA_ROOT / _ds["imagenet"])
IMAGENETV2_DIR = str(_DATA_ROOT / _ds["imagenetv2"])
IMAGENETR_DIR = str(_DATA_ROOT / _ds["imagenet-r"])
IMAGENETA_DIR = str(_DATA_ROOT / _ds["imagenet-a"])
IMAGENETS_DIR = str(_DATA_ROOT / _ds["imagenet-s"])
CUB_DIR = str(_DATA_ROOT / _ds["cub"])
EUROSAT_DIR = str(_DATA_ROOT / _ds.get("eurosat", ""))
FOOD101_DIR = str(_DATA_ROOT / _ds["food"])
PETS_DIR = str(_DATA_ROOT / _ds["pets"])
DTD_DIR = str(_DATA_ROOT / _ds.get("dtd", ""))
PLACES_DIR = str(_DATA_ROOT / _ds["place"])


# PyTorch datasets
tfms = _transform(hparams["image_size"])


def custom_loader(path: str) -> torch.Tensor:
    """Loads an image, applies a processing function, and returns augmented versions.

    Args:
        path (str): The path to the image file.
        n_samples (int): The number of augmented samples to generate.

    Returns:
        torch.Tensor: A tensor stack of the processed image and its augmented samples.
    """
    processor = tfms
    n_samples = hparams["n_samples"]
    # Load the image using the default loader
    img = datasets.folder.default_loader(path)
    # Process the image and generate additional augmented samples
    augmented_imgs = [processor(img)]
    augmented_imgs.extend(
        processor(random_crop(img, alpha=hparams["alpha"])) for _ in range(n_samples)
    )
    # Return a stacked tensor of all processed images
    return torch.stack(augmented_imgs)


if hparams["dataset"] == "imagenet":
    if hparams["dataset"] == "imagenet":
        dsclass = ImageNet
        hparams["data_dir"] = pathlib.Path(IMAGENET_DIR)
        # train_ds = ImageNet(hparams['data_dir'], split='val', transform=train_tfms)
        mydataset = dsclass(
            hparams["data_dir"], split="val", transform=None, loader=custom_loader
        )
        classes_to_load = None
        hparams["class_num"] = 1000

        if hparams["descriptor_fname"] is None:
            hparams["descriptor_fname"] = "descriptors_imagenet"
        hparams["after_text"] = hparams["label_after_text"] = "."

elif hparams["dataset"] == "imagenetv2":
    hparams["data_dir"] = pathlib.Path(IMAGENETV2_DIR)
    hparams["class_num"] = 1000

    mydataset = ImageNetV2Dataset(
        location=hparams["data_dir"],
        transform=None,
        loader=custom_loader,
    )

    classes_to_load = openai_imagenet_classes
    hparams["descriptor_fname"] = "descriptors_imagenet"
    mydataset.classes = classes_to_load

elif hparams["dataset"] == "imagenet-r":
    hparams["data_dir"] = pathlib.Path(IMAGENETR_DIR)
    dsclass = ImageFolder
    hparams["class_num"] = 200

    mydataset = dsclass(
        hparams["data_dir"],
        transform=None,
        loader=custom_loader,
    )
    hparams["descriptor_fname"] = "descriptors_imagenet"
    classes_to_load = None

elif hparams["dataset"] == "imagenet-a":
    hparams["data_dir"] = pathlib.Path(IMAGENETA_DIR)
    dsclass = ImageFolder
    hparams["class_num"] = 200

    mydataset = dsclass(
        hparams["data_dir"],
        transform=None,
        loader=custom_loader,
    )
    hparams["descriptor_fname"] = "descriptors_imagenet"
    classes_to_load = None

elif hparams["dataset"] == "imagenet-s":
    hparams["data_dir"] = pathlib.Path(IMAGENETS_DIR)
    dsclass = ImageFolder
    hparams["class_num"] = 1000

    mydataset = dsclass(
        hparams["data_dir"],
        transform=None,
        loader=custom_loader,
    )
    hparams["descriptor_fname"] = "descriptors_imagenet"
    classes_to_load = None


elif hparams["dataset"] == "cub":
    # load CUB dataset
    hparams["data_dir"] = pathlib.Path(CUB_DIR)

    mydataset = CUBDataset(
        hparams["data_dir"], train=False, transform=None, loader=custom_loader
    )
    classes_to_load = None  # dataset.classes
    hparams["descriptor_fname"] = "descriptors_cub"
    hparams["class_num"] = 200

# I recommend using VISSL https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md to download these


elif hparams["dataset"] == "place":
    hparams["class_num"] = 365
    hparams["data_dir"] = pathlib.Path(PLACES_DIR)
    mydataset = Places365(
        hparams["data_dir"],
        split="val",
        download=False,
        transform=None,
        loader=custom_loader,
    )
    # dsclass = ImageFolder
    # dataset = dsclass(hparams['data_dir'] / 'val', transform=tfms)
    hparams["descriptor_fname"] = "descriptors_places365"
    classes_to_load = None

elif hparams["dataset"] == "food":
    hparams["data_dir"] = pathlib.Path(FOOD101_DIR)
    dsclass = ImageFolder
    hparams["class_num"] = 101
    mydataset = Food101(
        hparams["data_dir"],
        transform=None,
        split="test",
        loader=custom_loader,
    )
    hparams["descriptor_fname"] = "descriptors_food101"
    classes_to_load = None

elif hparams["dataset"] == "pets":
    hparams["data_dir"] = pathlib.Path(PETS_DIR)
    dsclass = ImageFolder
    hparams["class_num"] = 37

    mydataset = OxfordIIITPet(
        hparams["data_dir"],
        transform=None,
        split="test",
        loader=custom_loader,
    )
    hparams["descriptor_fname"] = "descriptors_pets"
    classes_to_load = None


hparams["descriptor_fname"] = "./descriptors/" + hparams["descriptor_fname"]


print("Creating descriptors...")

if hparams["is_cupl"]:
    gpt_descriptions = load_cupl_descriptions(hparams, classes_to_load)
else:
    gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, classes_to_load)

label_to_classname = list(gpt_descriptions.keys())


n_classes = len(list(gpt_descriptions.keys()))


def load_clip_to_cpu():
    backbone_name = hparams["model_size"]
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def compute_description_encodings(model):
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        tokens = clip.tokenize(v).to(hparams["device"])
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    return description_encodings


def compute_label_encodings(model):
    label_encodings = F.normalize(
        model.encode_text(
            clip.tokenize(
                [
                    hparams["label_before_text"]
                    + wordify(l)
                    + hparams["label_after_text"]
                    for l in label_to_classname
                ]
            ).to(hparams["device"])
        )
    )
    return label_encodings


def aggregate_similarity(similarity_matrix_chunk, aggregation_method="mean"):
    if aggregation_method == "max":
        return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == "sum":
        return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == "mean":
        return similarity_matrix_chunk.mean(dim=1)
    else:
        raise ValueError("Unknown aggregate_similarity")


def show_from_indices(
    indices,
    images,
    labels=None,
    predictions=None,
    predictions2=None,
    n=None,
    image_description_similarity=None,
    image_labels_similarity=None,
):
    if indices is None or (len(indices) == 0):
        print("No indices provided")
        return

    if n is not None:
        indices = indices[:n]

    for index in indices:
        show_single_image(images[index])
        print(f"Index: {index}")
        if labels is not None:
            true_label = labels[index]
            true_label_name = label_to_classname[true_label]
            print(f"True label: {true_label_name}")
        if predictions is not None:
            predicted_label = predictions[index]
            predicted_label_name = label_to_classname[predicted_label]
            print(f"Predicted label (ours): {predicted_label_name}")
        if predictions2 is not None:
            predicted_label2 = predictions2[index]
            predicted_label_name2 = label_to_classname[predicted_label2]
            print(f"Predicted label 2 (CLIP): {predicted_label_name2}")

        print("\n")

        if image_labels_similarity is not None:
            if labels is not None:
                print(
                    f"Total similarity to {true_label_name} (true label) labels: {image_labels_similarity[index][true_label].item()}"
                )
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name:
                    print("Predicted label (ours) matches true label")
                else:
                    print(
                        f"Total similarity to {predicted_label_name} (predicted label) labels: {image_labels_similarity[index][predicted_label].item()}"
                    )
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) matches true label")
                elif (
                    predictions is not None
                    and predicted_label_name == predicted_label_name2
                ):
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else:
                    print(
                        f"Total similarity to {predicted_label_name2} (predicted label 2) labels: {image_labels_similarity[index][predicted_label2].item()}"
                    )

            print("\n")

        if image_description_similarity is not None:
            if labels is not None:
                print_descriptor_similarity(
                    image_description_similarity,
                    index,
                    true_label,
                    true_label_name,
                    "true",
                )
                print("\n")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name:
                    print("Predicted label (ours) same as true label")
                    # continue
                else:
                    print_descriptor_similarity(
                        image_description_similarity,
                        index,
                        predicted_label,
                        predicted_label_name,
                        "descriptor",
                    )
                print("\n")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) same as true label")
                    # continue
                elif (
                    predictions is not None
                    and predicted_label_name == predicted_label_name2
                ):
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else:
                    print_descriptor_similarity(
                        image_description_similarity,
                        index,
                        predicted_label2,
                        predicted_label_name2,
                        "CLIP",
                    )
            print("\n")


def print_descriptor_similarity(
    image_description_similarity, index, label, label_name, label_type="provided"
):
    # print(f"Total similarity to {label_name} ({label_type} label) descriptors: {aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    print(f"Total similarity to {label_name} ({label_type} label) descriptors:")
    print(
        f"Average:\t\t{100.0 * aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}"
    )
    label_descriptors = gpt_descriptions[label_name]
    for k, v in sorted(
        zip(label_descriptors, image_description_similarity[label][index]),
        key=lambda x: x[1],
        reverse=True,
    ):
        k = unmodify_dict[label_name][k]
        # print("\t" + f"matched \"{k}\" with score: {v}")
        print(f"{k}\t{100.0 * v}")


def print_max_descriptor_similarity(
    image_description_similarity, index, label, label_name
):
    max_similarity, argmax = image_description_similarity[label][index].max(dim=0)
    label_descriptors = gpt_descriptions[label_name]
    print(
        f"I saw a {label_name} because I saw {unmodify_dict[label_name][label_descriptors[argmax.item()]]} with score: {max_similarity.item()}"
    )


def show_misclassified_images(
    images,
    labels,
    predictions,
    n=None,
    image_description_similarity=None,
    image_labels_similarity=None,
    true_label_to_consider: int = None,
    predicted_label_to_consider: int = None,
):
    misclassified_indices = yield_misclassified_indices(
        images,
        labels=labels,
        predictions=predictions,
        true_label_to_consider=true_label_to_consider,
        predicted_label_to_consider=predicted_label_to_consider,
    )
    if misclassified_indices is None:
        return
    show_from_indices(
        misclassified_indices,
        images,
        labels,
        predictions,
        n=n,
        image_description_similarity=image_description_similarity,
        image_labels_similarity=image_labels_similarity,
    )


def yield_misclassified_indices(
    images,
    labels,
    predictions,
    true_label_to_consider=None,
    predicted_label_to_consider=None,
):
    misclassified_indicators = predictions.cpu() != labels.cpu()
    if true_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (
            labels.cpu() == true_label_to_consider
        )
    if predicted_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (
            predictions.cpu() == predicted_label_to_consider
        )

    if misclassified_indicators.sum() == 0:
        output_string = "No misclassified images found"
        if true_label_to_consider is not None:
            output_string += (
                f" with true label {label_to_classname[true_label_to_consider]}"
            )
        if predicted_label_to_consider is not None:
            output_string += f" with predicted label {label_to_classname[predicted_label_to_consider]}"
        print(output_string + ".")

        return

    misclassified_indices = torch.arange(images.shape[0])[misclassified_indicators]
    return misclassified_indices


from PIL import Image


def predict_and_show_explanations(
    images,
    model,
    labels=None,
    description_encodings=None,
    label_encodings=None,
    device=None,
):
    if type(images) == Image:
        images = tfms(images)

    if images.device != device:
        images = images.to(device)
        labels = labels.to(device)

    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)

    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)

    n_classes = len(description_encodings)
    image_description_similarity = [None] * n_classes
    image_description_similarity_cumulative = [None] * n_classes
    for i, (k, v) in enumerate(
        description_encodings.items()
    ):  # You can also vectorize this; it wasn't much faster for me
        dot_product_matrix = image_encodings @ v.T

        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(
            image_description_similarity[i]
        )

    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)

    descr_predictions = cumulative_tensor.argmax(dim=1)

    show_from_indices(
        torch.arange(images.shape[0]),
        images,
        labels,
        descr_predictions,
        clip_predictions,
        image_description_similarity=image_description_similarity,
        image_labels_similarity=image_labels_similarity,
    )
