"""
猫狗混淆问题实验 V4 - Focal-Guided Adaptive Sampling

核心原则：
1. Focal Loss 的 scale 动态调整，使得 Σ weights = N
   scale = 1 / E[(1-p_t)^gamma]
2. 每 5 个 epoch 更新一次 scale

实验配置：
1. CrossEntropy (baseline)
2. DynamicFocalLoss (gamma=2, scale动态)
3. CE + Focal-Guided Sampler
4. DynamicFocalLoss + Focal-Guided Sampler
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Sampler, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import CIFAR10_MEAN, CIFAR10_STD
from src.model import CNN
from torchvision import datasets, transforms

USE_GPUS = [4, 5, 6, 7]
DEVICE = torch.device(f'cuda:{USE_GPUS[0]}')
torch.manual_seed(42)
np.random.seed(42)


class DynamicFocalLoss(nn.Module):
    """
    动态 Focal Loss - scale 根据当前模型预测自适应调整

    scale = 1 / E[(1-p_t)^gamma]，使得总权重等于样本数量
    """
    def __init__(self, gamma=2.0, update_interval=5):
        super().__init__()
        self.gamma = gamma
        self.update_interval = update_interval
        self.scale = 2.0  # 初始值
        self.epoch = 0

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        probs = nn.functional.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        return (focal_loss * self.scale).mean()

    def update_scale(self, model, device, dataset):
        """
        根据当前模型预测更新 scale

        Args:
            model: 当前模型
            device: 计算设备
            dataset: 训练数据集
        """
        self.epoch += 1

        if self.epoch % self.update_interval != 0:
            return

        model.eval()

        # 计算整个训练集的 (1-p_t)^gamma 均值
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

        focal_weights = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                pt = probs.gather(1, targets.unsqueeze(1).to(device)).squeeze(1)
                focal_weights.append(((1 - pt) ** self.gamma).cpu())

        focal_weights = torch.cat(focal_weights)
        mean_weight = focal_weights.mean().item()

        # 计算 scale (无限制)
        new_scale = 1.0 / mean_weight

        self.scale = new_scale

        print(f"[DynamicFocalLoss] Epoch {self.epoch}: "
              f"mean_weight={mean_weight:.3f}, scale={self.scale:.2f}")

        model.train()


class FocalGuidedAdaptiveSampler(Sampler):
    """
    Focal Loss 引导的自适应采样器

    采样权重 w_i = (1 - p_t)^gamma
    (采样概率自动归一化，不需要额外 scale)
    """
    def __init__(
        self,
        dataset,
        gamma=2.0,
        warmup_epochs=3,
        min_weight=0.1,
        smoothing=0.5
    ):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight
        self.smoothing = smoothing

        self.weights = np.ones(self.num_samples) / self.num_samples
        self.epoch = 0

        print(f"[FocalGuidedSampler] γ={gamma}, warmup={warmup_epochs}, "
              f"min_weight={min_weight}, smoothing={smoothing}")

    def update_weights(self, model, device):
        """根据当前模型预测更新采样权重"""
        self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            self.weights = np.ones(self.num_samples) / self.num_samples
            print(f"[FocalGuidedSampler] Epoch {self.epoch}: Warmup, uniform sampling")
            return

        model.eval()

        loader = DataLoader(
            self.dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
                all_targets.append(targets)

        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        targets_expanded = all_targets.unsqueeze(1)
        pt = all_probs.gather(1, targets_expanded).squeeze(1).numpy()

        focal_weights = (1 - pt) ** self.gamma
        focal_weights = np.maximum(focal_weights, self.min_weight)

        uniform_weights = np.ones(self.num_samples)
        self.weights = (1 - self.smoothing) * focal_weights + self.smoothing * uniform_weights
        self.weights = self.weights / self.weights.sum()

        easy_ratio = (pt > 0.9).mean() * 100
        hard_ratio = (pt < 0.5).mean() * 100
        avg_focal_weight = focal_weights.mean()

        print(f"[FocalGuidedSampler] Epoch {self.epoch}: "
              f"easy={easy_ratio:.1f}%, hard={hard_ratio:.1f}%, "
              f"avg_focal_weight={avg_focal_weight:.3f}")

        model.train()

    def __iter__(self):
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            replace=True,
            p=self.weights
        )
        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_dataloaders(with_sampler=None):
    """获取数据加载器"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_full = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    indices = list(range(len(train_full)))
    train_dataset = torch.utils.data.Subset(train_full, indices[:45000])
    val_dataset = torch.utils.data.Subset(train_full, indices[45000:50000])

    if with_sampler:
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            sampler=with_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=True)

    return train_loader, val_loader, train_dataset


def train_one_config(config_name, criterion, use_sampler, num_epochs=50):
    """训练一个配置"""
    print(f"\n{'='*70}")
    print(f"配置: {config_name}")
    print(f"{'='*70}")

    train_loader, val_loader, train_dataset = get_dataloaders(with_sampler=None)

    sampler = None
    if use_sampler:
        sampler = FocalGuidedAdaptiveSampler(
            dataset=train_dataset,
            gamma=2.0,
            warmup_epochs=3,
            min_weight=0.1,
            smoothing=0.5
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

    model = CNN(dropout_rate=0.5).to(DEVICE)
    if len(USE_GPUS) > 1:
        model = nn.DataParallel(model, device_ids=USE_GPUS, output_device=USE_GPUS[0])
    print(f"[Model] 使用 {len(USE_GPUS)} 张 GPU: {USE_GPUS}")

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # 更新 Focal Loss 的 scale（如果是 DynamicFocalLoss）
        if isinstance(criterion, DynamicFocalLoss):
            criterion.update_scale(model, DEVICE, train_dataset)

        # 更新采样器权重
        if sampler is not None:
            sampler.update_weights(model, DEVICE)

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= train_total
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:2d}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    cat_as_dog = cm[3, 5] / cm[3].sum() * 100
    dog_as_cat = cm[5, 3] / cm[5].sum() * 100
    confusion_rate = (cat_as_dog + dog_as_cat) / 2

    print(f"\n[猫狗混淆] 猫→狗: {cat_as_dog:.2f}%, 狗→猫: {dog_as_cat:.2f}%, "
          f"平均混淆率: {confusion_rate:.2f}%")

    return history, confusion_rate


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    print("="*70)
    print("猫狗混淆问题实验 V4 - 动态 Focal Loss + 自适应采样")
    print("="*70)
    print(f"GPU 配置: {USE_GPUS}")
    print(f"Epochs: {args.epochs}")
    print("="*70)

    results = {}

    # 1. CrossEntropy (baseline)
    history, confusion = train_one_config(
        "1. CrossEntropy (baseline)",
        criterion=nn.CrossEntropyLoss(),
        use_sampler=False,
        num_epochs=args.epochs
    )
    results['CE'] = {'history': history, 'confusion': confusion}

    # 2. DynamicFocalLoss
    history, confusion = train_one_config(
        "2. DynamicFocalLoss (γ=2, scale∈[2,5], 每5 epoch更新)",
        criterion=DynamicFocalLoss(gamma=2.0, update_interval=5),
        use_sampler=False,
        num_epochs=args.epochs
    )
    results['DFL'] = {'history': history, 'confusion': confusion}

    # 3. CE + Focal-Guided Sampler
    history, confusion = train_one_config(
        "3. CE + FocalGuidedSampler (γ=2)",
        criterion=nn.CrossEntropyLoss(),
        use_sampler=True,
        num_epochs=args.epochs
    )
    results['CE+Sampler'] = {'history': history, 'confusion': confusion}

    # 4. DynamicFocalLoss + Focal-Guided Sampler
    history, confusion = train_one_config(
        "4. DynamicFL + FocalGuidedSampler",
        criterion=DynamicFocalLoss(gamma=2.0, update_interval=5),
        use_sampler=True,
        num_epochs=args.epochs
    )
    results['DFL+Sampler'] = {'history': history, 'confusion': confusion}

    # 结果对比
    print("\n" + "="*70)
    print("最终结果对比")
    print("="*70)
    print(f"{'配置':<40} {'Val Acc':<12} {'混淆率':<12} {'vs CE':<15}")
    print("-"*70)

    baseline_acc = results['CE']['history']['val_acc'][-1]
    baseline_conf = results['CE']['confusion']

    for key in ['CE', 'DFL', 'CE+Sampler', 'DFL+Sampler']:
        acc = results[key]['history']['val_acc'][-1]
        conf = results[key]['confusion']
        if key == 'CE':
            print(f"{key:<40} {acc:<12.2f}% {conf:<12.2f}% -")
        else:
            acc_delta = acc - baseline_acc
            conf_delta = baseline_conf - conf
            print(f"{key:<40} {acc:<12.2f}% {conf:<12.2f}% "
                  f"Acc:{acc_delta:+.2f}% 混淆:{conf_delta:+.2f}%")

    print("="*70)

    plot_results(results, args.epochs)


def plot_results(results, num_epochs):
    """绘制结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cat-Dog Confusion V4: Dynamic Focal Loss + Adaptive Sampling', fontsize=16)

    labels = ['CE', 'DFL', 'CE+Sampler', 'DFL+Sampler']
    display_names = [
        'CrossEntropy',
        'Dynamic FL (scale∈[2,5])',
        'CE + FocalSampler',
        'DynamicFL + Sampler'
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    epochs = range(1, num_epochs + 1)

    axes[0, 0].set_title('Training Loss')
    for i, key in enumerate(labels):
        axes[0, 0].plot(epochs, results[key]['history']['train_loss'],
                       label=display_names[i], color=colors[i], linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss')
    for i, key in enumerate(labels):
        axes[0, 1].plot(epochs, results[key]['history']['val_loss'],
                       label=display_names[i], color=colors[i], linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Validation Accuracy')
    for i, key in enumerate(labels):
        axes[1, 0].plot(epochs, results[key]['history']['val_acc'],
                       label=display_names[i], color=colors[i], linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    x = np.arange(len(labels))
    width = 0.35
    final_accs = [results[k]['history']['val_acc'][-1] for k in labels]
    confusions = [results[k]['confusion'] for k in labels]

    axes[1, 1].bar(x - width/2, final_accs, width, label='Val Acc (%)', color='#2ecc71')
    axes[1, 1].bar(x + width/2, confusions, width, label='Confusion (%)', color='#e74c3c')
    axes[1, 1].set_ylabel('%')
    axes[1, 1].set_title('Final Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(display_names, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./models/cat_dog_experiment_v4.png', dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: ./models/cat_dog_experiment_v4.png")
    plt.close()


if __name__ == "__main__":
    main()
