"""
Fine-tune a dynamic model with decision units held in deterministic mode.

Loads a decision-model checkpoint and fine-tunes the backbone weights with the
pruning gates frozen at their deterministic outputs. Requires a CUDA-capable GPU.

Usage: python train/finetune.py --arch resnet10 --dataset cifar10
"""

import torch.nn.functional as F
import numpy as np
import torch
import argparse
import os

from decision import default_graph, apply_func, set_deterministic_value
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--arch", "-a", default="resnet56", type=str)
parser.add_argument("--sparsity_level", default=0.7, type=float)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--epochs", default=160, type=int)
parser.add_argument("--log_interval", default=100, type=int)
parser.add_argument("--train_batch_size", default=128, type=int)

args = parser.parse_args()

args.num_classes = {"cifar10": 10, "cifar100": 100, "har": 6, "kws": 12}.get(args.dataset, 10)

args.device = "cuda"
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.logdir = "finetune-decision-%d/%s-%s/sparsity-%.2f" % (
    misc.action_num(args.arch),
    args.dataset,
    args.arch,
    args.sparsity_level,
)
misc.prepare_logging(args)

trainloader, testloader = misc.prepare_data(args.dataset, args.train_batch_size)

model = misc.initialize_model(args.dataset, args.arch, args.num_classes)
model_params = list(model.parameters())

misc.transform_model(model, args.arch, misc.action_num(args.arch))

print("==> Loading pretrained decision model...")
ckpt = torch.load(
    "logs/decision-%d/%s-%s/sparsity-%.2f/model.pth.tar"
    % (misc.action_num(args.arch), args.dataset, args.arch, args.sparsity_level),
    weights_only=True,
)
model.load_state_dict(ckpt["state_dict"])

model = model.to(args.device)

optimizer = torch.optim.SGD(
    model_params, lr=misc.learning_rate(args.arch), momentum=0.9, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[80, 120], gamma=0.1
)


def train(epoch):
    model.train()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            selected_channels = default_graph.get_tensor_list("selected_channels")
            concat_channels = torch.cat(selected_channels, dim=1)
            sparsity = (concat_channels != 0).float().mean()
            acc = (output.max(1)[1] == target).float().mean()

            print(
                "Train Epoch: %d [%d/%d]\tLoss: %.4f, "
                "Sparsity: %.4f, Accuracy: %.4f"
                % (epoch, i, len(trainloader), loss.item(), sparsity.item(), acc.item())
            )


def test():
    model.eval()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
    test_loss = []
    test_sparsity = []
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            default_graph.clear_all_tensors()

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            selected_channels = default_graph.get_tensor_list("selected_channels")
            concat_channels = torch.cat(selected_channels, dim=1)

            test_loss.append(F.cross_entropy(output, target).item())
            test_sparsity.append((concat_channels != 0).float().mean().item())

            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    acc = correct / len(testloader.dataset)
    print(
        "Test set: Loss: %.4f, "
        "Sparsity: %.4f, Accuracy: %.4f\n"
        % (np.mean(test_loss), np.mean(test_sparsity), acc)
    )
    return acc, np.mean(test_sparsity)


for epoch in range(args.epochs):
    train(epoch)
    acc, sparsity = test()
    torch.save(model.state_dict(), os.path.join(args.logdir, "checkpoint.pth"))
    print(
        "Save checkpoint @ Epoch %d, Accuracy = %.4f, Sparsity = %.4f\n"
        % (epoch, acc, sparsity)
    )
