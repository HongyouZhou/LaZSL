import torch

from load_OP import *
import torchmetrics
from tqdm import tqdm


seed_everything(hparams["seed"])

bs = hparams["batch_size"]
dataloader = DataLoader(mydataset, bs, shuffle=True, num_workers=16, pin_memory=True)

print("Loading model...")

device = torch.device(hparams["device"])
# load model
model = load_clip_to_cpu()
model.to(device)
model.eval()
model.requires_grad_(False)
op_d = OP_d(
    max_iter=hparams["max_iter"],
    gama=hparams["gama"],
    constrain_type=hparams["constrain_type"],
    theta=hparams["theta"],
)

print("Encoding descriptions...")

description_encodings = compute_description_encodings(model)

label_encodings = compute_label_encodings(model)

print(
    "n_samples: %d \nalpha: %f \ntheta: %f"
    % (hparams["n_samples"], hparams["alpha"], hparams["theta"])
)
print("constrain_type: %s " % (hparams["constrain_type"]))

total_batches = len(dataloader)
print(f"Evaluating... (total_batches={total_batches})")


langOPD_accuracy_metric = torchmetrics.Accuracy(
    task="multiclass", num_classes=hparams["class_num"]
).to(device)

lang_accuracy_metric = torchmetrics.Accuracy(
    task="multiclass", num_classes=hparams["class_num"]
).to(device)

for batch_number, batch in enumerate(tqdm(dataloader)):
    images = batch[0]
    labels = batch[1]

    images = images.to(device)
    labels = labels.to(device)

    bs, n_region, n_chaneel, h, w = images.shape

    images = torch.reshape(images, (bs * n_region, n_chaneel, h, w))

    image_encodings = model.encode_image(images)

    image_encodings = torch.reshape(image_encodings, (bs, n_region, -1))
    image_encodings = torch.permute(image_encodings, (1, 0, 2))

    image_encodings = F.normalize(image_encodings, dim=2)

    image_encodings_global = image_encodings[0]

    image_encodings_region = image_encodings[1:]
    sim_rg = torch.einsum("nbd,bd->bn", image_encodings_region, image_encodings_global)

    image_description_similarity = [None] * n_classes
    image_description_similarity_cumulative_OP_c = [None] * n_classes
    image_description_similarity_cumulative = [None] * n_classes

    for i, (k, v) in enumerate(description_encodings.items()):
        """#region +cost&sim_global"""
        image_description_similarity_cumulative_OP_c[i] = op_d.get_OP_distence(
            image_features=image_encodings,
            text_features=v,
            sim_rg=sim_rg,
            is_cost_global=True,
            is_sim_global=True,
        )

        dot_product_matrix = image_encodings_global @ v.T

        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(
            image_description_similarity[i]
        )

    cumulative_tensor = torch.stack(image_description_similarity_cumulative, dim=1)
    cumulative_tensor_OP_C = torch.stack(
        image_description_similarity_cumulative_OP_c, dim=1
    )

    if hparams["dataset"] == "imagenet-a":
        cumulative_tensor = cumulative_tensor[:, imagenet_a_lt]
        cumulative_tensor_OP_C = cumulative_tensor_OP_C[:, imagenet_a_lt]

    if hparams["dataset"] == "imagenet-r":
        cumulative_tensor = cumulative_tensor[:, imagenet_r_lt]
        cumulative_tensor_OP_C = cumulative_tensor_OP_C[:, imagenet_a_lt]

    lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    langOPC_acc = langOPD_accuracy_metric(
        cumulative_tensor_OP_C.softmax(dim=-1), labels
    )

    # Emit parseable progress line for runner
    print(f"\n[PROGRESS] {batch_number + 1}/{total_batches}")
    print(
        "\nbaseline: %.2f\nbaseline+op+cost&sim_global: %.2f" % (lang_acc, langOPC_acc)
    )


print("\n")

accuracy_logs = {}

accuracy_logs["Total Description-based Top-1 Accuracy: "] = (
    100 * lang_accuracy_metric.compute().item()
)

accuracy_logs["Total DescriptionOP-Cost&SimGlobal-based Top-1 Accuracy: "] = (
    100 * langOPD_accuracy_metric.compute().item()
)

print("\n" + "=" * 60)
print("Final Results:")
print("=" * 60)
for key, value in accuracy_logs.items():
    print(f"  {key}{value:.2f}%")

# Save results to JSON if output_dir is set
import json, os
from datetime import datetime

if hparams.get("output_dir"):
    os.makedirs(hparams["output_dir"], exist_ok=True)
    model_tag = hparams["model_size"].replace("/", "-")
    fname = f"{hparams['dataset']}_{model_tag}.json"

    results = {
        "dataset": hparams["dataset"],
        "model": hparams["model_size"],
        "device": hparams["device"],
        "accuracy": {
            "baseline_top1": accuracy_logs["Total Description-based Top-1 Accuracy: "],
            "lazsl_top1": accuracy_logs[
                "Total DescriptionOP-Cost&SimGlobal-based Top-1 Accuracy: "
            ],
        },
        "hparams": {
            "seed": hparams["seed"],
            "batch_size": hparams["batch_size"],
            "n_samples": hparams["n_samples"],
            "alpha": hparams["alpha"],
            "theta": hparams["theta"],
            "constrain_type": hparams["constrain_type"],
            "gama": hparams["gama"],
        },
        "timestamp": datetime.now().isoformat(),
    }

    result_path = os.path.join(hparams["output_dir"], fname)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_path}")
