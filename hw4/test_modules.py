"""
单元测试：用 PyTorch 对比自定义模块的 forward/backward 输出
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "build")
import mytensor as mt

np.random.seed(42)
ATOL = 1e-4


def check(name: str, ours: np.ndarray, ref: np.ndarray, atol: float = ATOL) -> bool:
    diff = np.max(np.abs(ours - ref))
    ok = diff < atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}  (max_diff={diff:.6e})")
    return ok


# ==================== ReLU ====================
def test_relu() -> bool:
    print("=== ReLU ===")
    x_np = np.random.randn(4, 8).astype(np.float32)
    x_t = torch.tensor(x_np, requires_grad=True)

    relu = mt.ReLU()
    out_mt = relu.forward(mt.from_numpy(x_np)).numpy()
    ok = check("forward", out_mt, F.relu(x_t).detach().numpy())

    grad_np = np.random.randn(4, 8).astype(np.float32)
    grad_in_mt = relu.backward(mt.from_numpy(grad_np)).numpy()
    F.relu(x_t).backward(torch.tensor(grad_np))
    ok &= check("backward", grad_in_mt, x_t.grad.numpy())
    return ok


# ==================== Sigmoid ====================
def test_sigmoid() -> bool:
    print("=== Sigmoid ===")
    x_np = np.random.randn(4, 8).astype(np.float32)
    x_t = torch.tensor(x_np, requires_grad=True)

    sig = mt.Sigmoid()
    out_mt = sig.forward(mt.from_numpy(x_np)).numpy()
    ok = check("forward", out_mt, torch.sigmoid(x_t).detach().numpy())

    grad_np = np.random.randn(4, 8).astype(np.float32)
    grad_in_mt = sig.backward(mt.from_numpy(grad_np)).numpy()
    torch.sigmoid(x_t).backward(torch.tensor(grad_np))
    ok &= check("backward", grad_in_mt, x_t.grad.numpy())
    return ok


# ==================== FC ====================
def test_fc() -> bool:
    print("=== FC ===")
    N, Ci, Co = 4, 8, 6
    x_np = np.random.randn(N, Ci).astype(np.float32)
    w_np = np.random.randn(Co, Ci).astype(np.float32)
    b_np = np.random.randn(Co).astype(np.float32)

    x_t = torch.tensor(x_np, requires_grad=True)
    w_t = torch.tensor(w_np, requires_grad=True)
    b_t = torch.tensor(b_np, requires_grad=True)
    out_pt = F.linear(x_t, w_t, b_t)

    fc = mt.FC(Ci, Co)
    fc.weight = mt.from_numpy(w_np)
    fc.bias = mt.from_numpy(b_np)
    ok = check("forward", fc.forward(mt.from_numpy(x_np)).numpy(),
               out_pt.detach().numpy())

    grad_np = np.random.randn(N, Co).astype(np.float32)
    out_pt.backward(torch.tensor(grad_np))
    g = mt.from_numpy(grad_np)
    ok &= check("grad_input", fc.backward(g).numpy(), x_t.grad.numpy())
    ok &= check("grad_weight", fc.backward_weight(g).numpy(), w_t.grad.numpy())
    ok &= check("grad_bias", fc.backward_bias(g).numpy(), b_t.grad.numpy())
    return ok


# ==================== Conv ====================
def test_conv() -> bool:
    print("=== Conv ===")
    N, Ci, Co, H, W = 2, 3, 4, 8, 8
    x_np = np.random.randn(N, Ci, H, W).astype(np.float32)
    w_np = np.random.randn(Co, Ci, 3, 3).astype(np.float32)
    b_np = np.random.randn(Co).astype(np.float32)

    x_t = torch.tensor(x_np, requires_grad=True)
    w_t = torch.tensor(w_np, requires_grad=True)
    b_t = torch.tensor(b_np, requires_grad=True)
    out_pt = F.conv2d(x_t, w_t, b_t, padding=1)

    conv = mt.Conv(Ci, Co)
    conv.weight = mt.from_numpy(w_np)
    conv.bias = mt.from_numpy(b_np)
    ok = check("forward", conv.forward(mt.from_numpy(x_np)).numpy(),
               out_pt.detach().numpy())

    grad_np = np.random.randn(N, Co, H, W).astype(np.float32)
    out_pt.backward(torch.tensor(grad_np))
    g = mt.from_numpy(grad_np)
    ok &= check("grad_input", conv.backward(g).numpy(), x_t.grad.numpy())
    ok &= check("grad_weight", conv.backward_weight(g).numpy(),
                w_t.grad.numpy())
    return ok


# ==================== Pooling ====================
def test_pooling() -> bool:
    print("=== Pooling ===")
    N, C, H, W = 2, 3, 8, 8
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    x_t = torch.tensor(x_np, requires_grad=True)

    out_pt = F.max_pool2d(x_t, 2)

    pool = mt.Pooling()
    out_mt = pool.forward(mt.from_numpy(x_np)).numpy()
    ok = check("forward", out_mt, out_pt.detach().numpy())

    grad_np = np.random.randn(N, C, H // 2, W // 2).astype(np.float32)
    out_pt.backward(torch.tensor(grad_np))
    grad_in_mt = pool.backward(mt.from_numpy(grad_np)).numpy()
    ok &= check("backward", grad_in_mt, x_t.grad.numpy())
    return ok


# ==================== SoftMax ====================
def test_softmax() -> bool:
    print("=== SoftMax ===")
    x_np = np.random.randn(4, 10).astype(np.float32)

    sm = mt.SoftMax()
    out_mt = sm.forward(mt.from_numpy(x_np)).numpy()
    out_pt = F.softmax(torch.tensor(x_np), dim=1).numpy()
    return check("forward", out_mt, out_pt)


# ==================== CrossEntropyLoss ====================
def test_cross_entropy() -> bool:
    print("=== CrossEntropyLoss ===")
    N, C = 4, 10
    # 先生成 softmax 概率作为输入
    logits_np = np.random.randn(N, C).astype(np.float32)
    probs_np = np.exp(logits_np) / np.exp(logits_np).sum(axis=1, keepdims=True)
    labels_np = np.array([2, 5, 0, 7], dtype=np.float32)

    # 自定义
    ce = mt.CrossEntropyLoss()
    loss_mt = ce.forward(mt.from_numpy(probs_np), mt.from_numpy(labels_np))
    grad_mt = ce.backward(mt.from_numpy(labels_np)).numpy()

    # PyTorch: cross_entropy 接受 logits，我们手动算
    labels_int = labels_np.astype(np.int64)
    ref_loss = -np.mean(np.log(probs_np[np.arange(N), labels_int] + 1e-12))
    ok = check("forward loss", np.array([loss_mt]), np.array([ref_loss]))

    # backward: d(CE)/d(probs) = probs - one_hot(labels) / N
    one_hot = np.zeros_like(probs_np)
    one_hot[np.arange(N), labels_int] = 1.0
    ref_grad = (probs_np - one_hot) / N
    ok &= check("backward", grad_mt, ref_grad.astype(np.float32))
    return ok


# ==================== main ====================
if __name__ == "__main__":
    tests = [
        test_relu, test_sigmoid, test_fc,
        test_conv, test_pooling, test_softmax,
        test_cross_entropy,
    ]
    passed = sum(t() for t in tests)
    print(f"\n{'='*40}")
    print(f"结果: {passed}/{len(tests)} 模块全部通过")
    if passed == len(tests):
        print("All tests passed!")
    else:
        print("Some tests FAILED.")
        sys.exit(1)
