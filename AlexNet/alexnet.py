"""
AlexNet网络训练算法复现
"""

from collections.abc import Iterable, Sequence
from typing import cast

import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import (  # pyright: ignore[reportMissingTypeStubs]
    datasets,
    models,
    transforms,
)

Sample = tuple[Tensor, int]
Batch = Sequence[Tensor]

# ====== 超参数 ======

DATA_DIR = "./data"
BATCH_SIZE = 64
NUM_CLASSES = 10

# SGD的动量系数
MOMENTUM = 0.9

# SGD前 1 ~ 7 轮的学习率
LR = 0.01

# 训练总轮数
EPOCHS = 20

# 模型文件保存地址
SAVE_PATH = "alexnet_difar10.pth"

# ====== 设备 ======

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====== 数据预处理 ======


def compute_mean_std(
    dataset: Dataset[Sample],
) -> tuple[list[float], list[float]]:
    """统计整个训练集各通道的均值和总体标准差"""
    loader: DataLoader[Sample] = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 各通道所有像素值的和
    channel_sum: Tensor = torch.zeros(3, dtype=torch.float64)

    # 各通道所有像素值的平方和
    channel_squared_sum: Tensor = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for images, _ in cast(Iterable[Batch], loader):
        images = images.to(torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += images.square().sum(dim=(0, 2, 3))
        pixel_count += images.size(0) * images.size(2) * images.size(3)

    if pixel_count == 0:
        raise ValueError("训练集为空，无法计算 mean 和 std.")

    mean: Tensor = channel_sum / pixel_count
    # clamp(min=0) 处理浮点计算理论误差: 理论上方差不会小于零，但计算机可能算出 -0.000000000000001 这样的结果.
    std: torch.Tensor = (
        (channel_squared_sum / pixel_count - mean.square()).clamp(min=0).sqrt()
    )

    if (std == 0).any():
        raise ValueError("存在标准差为 0 的通道，无法直接用于 Normalize")

    return (
        [float(mean[c].item()) for c in range(3)],
        [float(std[c].item()) for c in range(3)],
    )


train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    ),
)

print("Computing training-set mean and std...")
mean, std = compute_mean_std(cast(Dataset[Sample], train_dataset))
print(f"mean = {mean}, std = {std}")

train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

train_dataset.transform = train_transform

val_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=val_transform,
)

train_loader: DataLoader[Sample] = DataLoader(
    dataset=cast(Dataset[Sample], train_dataset),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader: DataLoader[Sample] = DataLoader(
    dataset=cast(Dataset[Sample], val_dataset),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ====== 模型 ======

model = models.alexnet(weights=None)

# 调整模型结构，适配数据集 CIFAR-10
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
model = model.to(device)

# 交叉熵损失函数
criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

# 优化器
optimizer: optim.SGD = optim.SGD(
    params=model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
)

# 学习率调度器
scheduler: StepLR = StepLR(optimizer, step_size=7, gamma=0.1)

# ====== 一轮训练 ======


def train_one_epoch(epoch: int) -> tuple[float, float]:
    """
    一次函数训练训练一轮epoch
    epoch参数用于表示是第几轮训练，函数内用于print
    """
    model.train()
    running_loss: float = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(cast(Iterable[Batch], train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs: Tensor = cast(Tensor, model(images))
        loss: Tensor = cast(Tensor, criterion(outputs, labels))
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        predicted: Tensor = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += int(predicted.eq(labels).sum().item())

        if (i + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], "
                f"Step [{i + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, "
                f"Acc: {100.0 * correct / total:.2f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ====== 一轮验证 ======


def evaluate() -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in cast(Iterable[Batch], val_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = cast(Tensor, model(images))
            loss = cast(Tensor, criterion(outputs, labels))

            running_loss += float(loss.item()) * images.size(0)
            predicted: Tensor = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += int(predicted.eq(labels).sum().item())

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


# ====== 主函数 ======


def main() -> None:
    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = evaluate()
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model to {SAVE_PATH} (Val Acc: {best_acc:.2f}%)")

    print(f"Training finished. Best Val Acc: {best_acc:.2f}")


if __name__ == "__main__":
    main()
