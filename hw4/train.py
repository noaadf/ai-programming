"""
MNIST 训练脚本：使用自定义 C++ 框架 (mytensor) 训练 CNN
"""
import sys
import numpy as np

sys.path.insert(0, "build")
import mytensor as mt


def load_mnist(data_dir: str = "./data"):
    """用 torchvision 加载 MNIST，返回 numpy 数组"""
    from torchvision import datasets, transforms

    train_set = datasets.MNIST(data_dir, train=True, download=True)
    test_set = datasets.MNIST(data_dir, train=False, download=True)

    # 归一化到 [0, 1]，reshape 为 [N, 1, 28, 28]
    x_train = train_set.data.numpy().astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 1, 28, 28)
    y_train = train_set.targets.numpy().astype(np.float32)

    x_test = test_set.data.numpy().astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 1, 28, 28)
    y_test = test_set.targets.numpy().astype(np.float32)

    return x_train, y_train, x_test, y_test


class CNN:
    """
    Conv(1→16) → ReLU → Pool(28→14)
    Conv(16→32) → ReLU → Pool(14→7)
    Flatten → FC(32*7*7→128) → ReLU → FC(128→10)
    """

    def __init__(self):
        self.conv1 = mt.Conv(1, 16)
        self.conv2 = mt.Conv(16, 32)
        self.relu1 = mt.ReLU()
        self.relu2 = mt.ReLU()
        self.relu3 = mt.ReLU()
        self.pool1 = mt.Pooling()
        self.pool2 = mt.Pooling()
        self.fc1 = mt.FC(32 * 7 * 7, 128)
        self.fc2 = mt.FC(128, 10)
        self.softmax = mt.SoftMax()
        self.loss_fn = mt.CrossEntropyLoss()

        # 用小随机数初始化权重
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化"""
        for conv, fan_in in [(self.conv1, 1 * 9), (self.conv2, 16 * 9)]:
            std = np.sqrt(2.0 / fan_in)
            conv.weight = mt.from_numpy(
                np.random.randn(*conv.weight.shape).astype(np.float32) * std
            )
            conv.bias = mt.from_numpy(np.zeros(conv.bias.shape, dtype=np.float32))

        for fc, fan_in in [(self.fc1, 32 * 7 * 7), (self.fc2, 128)]:
            std = np.sqrt(2.0 / fan_in)
            fc.weight = mt.from_numpy(
                np.random.randn(*fc.weight.shape).astype(np.float32) * std
            )
            fc.bias = mt.from_numpy(np.zeros(fc.bias.shape, dtype=np.float32))

    def forward(self, x: mt.Tensor) -> mt.Tensor:
        """前向传播，返回 softmax 概率"""
        # Conv block 1
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        # Conv block 2
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        # Flatten: [N, 32, 7, 7] → [N, 32*7*7]
        x_np = x.numpy().reshape(x.shape[0], -1)
        x = mt.from_numpy(x_np)

        # FC block
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)

        # Softmax
        x = self.softmax.forward(x)
        return x

    def backward(self, labels: mt.Tensor, lr: float = 0.01):
        """反向传播 + SGD 参数更新"""
        # softmax + cross entropy 合并梯度
        grad = self.loss_fn.backward(labels)

        # FC2 反向
        grad_w2 = self.fc2.backward_weight(grad)
        grad_b2 = self.fc2.backward_bias(grad)
        grad = self.fc2.backward(grad)

        # ReLU3 反向
        grad = self.relu3.backward(grad)

        # FC1 反向
        grad_w1 = self.fc1.backward_weight(grad)
        grad_b1 = self.fc1.backward_bias(grad)
        grad = self.fc1.backward(grad)

        # Unflatten: [N, 32*7*7] → [N, 32, 7, 7]
        N = grad.shape[0]
        grad_np = grad.numpy().reshape(N, 32, 7, 7)
        grad = mt.from_numpy(grad_np)

        # Pool2 反向
        grad = self.pool2.backward(grad)

        # ReLU2 反向
        grad = self.relu2.backward(grad)

        # Conv2 反向
        grad_conv2_w = self.conv2.backward_weight(grad)
        grad_conv2_out = grad  # 保存，用于 bias 梯度
        grad = self.conv2.backward(grad)

        # Pool1 反向
        grad = self.pool1.backward(grad)

        # ReLU1 反向
        grad = self.relu1.backward(grad)

        # Conv1 反向
        grad_conv1_w = self.conv1.backward_weight(grad)
        grad_conv1_out = grad  # 保存，用于 bias 梯度

        # SGD 参数更新
        self._sgd_update(self.fc2, grad_w2, grad_b2, lr)
        self._sgd_update(self.fc1, grad_w1, grad_b1, lr)
        self._conv_sgd_update(self.conv2, grad_conv2_w, grad_conv2_out, lr)
        self._conv_sgd_update(self.conv1, grad_conv1_w, grad_conv1_out, lr)

    @staticmethod
    def _sgd_update(fc, grad_w, grad_b, lr: float):
        """FC 层的 SGD 更新：w = w - lr * grad"""
        w = fc.weight.numpy() - lr * grad_w.numpy()
        b = fc.bias.numpy() - lr * grad_b.numpy()
        fc.weight = mt.from_numpy(w)
        fc.bias = mt.from_numpy(b)

    @staticmethod
    def _conv_sgd_update(conv, grad_w, grad_output, lr: float):
        """Conv 层的 SGD 更新（bias 梯度 = grad_output 对空间维求和）"""
        w = conv.weight.numpy() - lr * grad_w.numpy()
        conv.weight = mt.from_numpy(w)
        # conv bias 梯度：对 N, H, W 维度求和
        g = grad_output.numpy()  # [N, C_out, H, W]
        grad_b = g.sum(axis=(0, 2, 3))
        b = conv.bias.numpy() - lr * grad_b.astype(np.float32)
        conv.bias = mt.from_numpy(b)


def evaluate(model: CNN, x: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    """计算分类准确率"""
    correct = 0
    n = x.shape[0]
    for i in range(0, n, batch_size):
        xb = mt.from_numpy(x[i:i + batch_size])
        probs = model.forward(xb)
        preds = probs.numpy().argmax(axis=1)
        correct += (preds == y[i:i + batch_size]).sum()
    return correct / n


def train():
    """主训练流程"""
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = CNN()
    batch_size = 64
    lr = 0.01
    epochs = 5
    n = x_train.shape[0]

    for epoch in range(epochs):
        # 每个 epoch 打乱数据
        perm = np.random.permutation(n)
        x_train, y_train = x_train[perm], y_train[perm]

        total_loss = 0.0
        num_batches = 0

        for i in range(0, n, batch_size):
            xb = mt.from_numpy(x_train[i:i + batch_size])
            yb = mt.from_numpy(y_train[i:i + batch_size])

            # 前向
            probs = model.forward(xb)
            loss = model.loss_fn.forward(probs, yb)
            total_loss += loss
            num_batches += 1

            # 反向 + 更新
            model.backward(yb, lr)

            if num_batches % 100 == 0:
                print(f"  Epoch {epoch+1}, batch {num_batches}, loss: {loss:.4f}")

        avg_loss = total_loss / num_batches
        train_acc = evaluate(model, x_train, y_train)
        test_acc = evaluate(model, x_test, y_test)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}, "
              f"train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    train()
