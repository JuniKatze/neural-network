"""
AlexNet网络训练流程:
1. 数据加载
2. 模型定义
3. 损失函数
4. 优化器
5. 训练循环
6. 验证
7. 保存模型
"""

from collections.abc import Iterable, Sequence
from typing import cast

import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

# torchvision 0.18.1 未提供完整的类型标记；仍使用其源码推断具体接口。
from torchvision import (  # pyright: ignore[reportMissingTypeStubs]
    datasets,
    models,
    transforms,
)

# 单个样本的标签为 int；默认拼接后的 batch 中图片和标签均为 Tensor。
Sample = tuple[Tensor, int]
Batch = Sequence[Tensor]

# ====== 超参数 ======

BATCH_SIZE = 64
DATA_DIR = "./data"
NUM_CLASSES = 10
LR = 0.01
MOMENTUM = 0.9
EPOCHS = 20
SAVE_PATH = "alexnet_cifar10.pth"

# ====== 设备 ======

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====== 数据预处理 ======


def compute_mean_std(dataset: Dataset[Sample]) -> tuple[list[float], list[float]]:
    """统计整个训练集各通道的均值和总体标准差。"""
    loader: DataLoader[Sample] = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    channel_sum: Tensor = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum: Tensor = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for images, _ in cast(Iterable[Batch], loader):
        # images 的形状为 [B, C, H, W]，保留 C，累加其余维度。
        images = images.to(torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += images.square().sum(dim=(0, 2, 3))
        pixel_count += images.size(0) * images.size(2) * images.size(3)

    if pixel_count == 0:
        raise ValueError("训练集为空，无法计算 mean 和 std")

    mean: Tensor = channel_sum / pixel_count
    # Var(X) = E[X²] - E[X]²；clamp 避免浮点误差产生负数。
    std: torch.Tensor = (
        (channel_squared_sum / pixel_count - mean.square()).clamp(min=0).sqrt()
    )
    if (std == 0).any():
        raise ValueError("存在标准差为 0 的通道，无法直接用于 Normalize")
    return (
        [float(mean[c].item()) for c in range(3)],
        [float(std[c].item()) for c in range(3)],
    )


# 只统计训练集；保留固定尺寸调整，不做随机增强或 Normalize。
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
# ToTensor 保证样本图片为 Tensor；cast 只补充类型信息，不转换数据。
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

# ====== 数据集 ======

# 统计完成后，为同一训练集切换到训练预处理。
train_dataset.transform = train_transform

val_dataset = datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=val_transform
)

train_loader: DataLoader[Sample] = DataLoader(
    cast(Dataset[Sample], train_dataset),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader: DataLoader[Sample] = DataLoader(
    cast(Dataset[Sample], val_dataset),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ====== 模型 ======

model: models.AlexNet = models.alexnet(weights=None)

model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
model = model.to(device)

criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.SGD = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
)
scheduler: StepLR = StepLR(optimizer, step_size=7, gamma=0.1)


def train_one_epoch(epoch: int) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(cast(Iterable[Batch], train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cast(Tensor, model(images))
        loss = cast(Tensor, criterion(outputs, labels))
        # torch 2.3.1 的 backward 参数类型未补全；这里使用默认参数。
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
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


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
            # torch 2.3.1 的 save 中 PathLike 泛型不完整；这里传入的是 str。
            torch.save(model.state_dict(), SAVE_PATH)  # pyright: ignore[reportUnknownMemberType]
            print(f"Saved best model to {SAVE_PATH} (Val Acc: {best_acc:.2f}%)")

    print(f"Training finished. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
