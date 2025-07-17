import torch
import triton
import triton.language as tl
import pytest
import math

# 用于性能分析
from triton.testing import do_bench

# --------------------------------------------------
# 1. Fused AdamW Kernel 
# --------------------------------------------------
@triton.jit
def fused_adamw_kernel(
    p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    n_elements, lr,
    bias_correction1, sqrt_bias_correction2,
    beta1, beta2,
    one_minus_beta1, one_minus_beta2,
    eps, weight_decay,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

    p = p * (1.0 - lr * weight_decay)

    grad_update_m = grad * one_minus_beta1
    exp_avg = tl.fma(exp_avg, beta1, grad_update_m)

    grad_sq = grad * grad
    grad_update_v = grad_sq * one_minus_beta2
    exp_avg_sq = tl.fma(exp_avg_sq, beta2, grad_update_v)

    step_size = lr / bias_correction1

    denom = (tl.sqrt(exp_avg_sq) / sqrt_bias_correction2) + eps

    p_new = p - (exp_avg / denom) * step_size

    tl.store(p_ptr + offsets, p_new, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


# --------------------------------------------------
# 2. 严格的性能分析函数
# --------------------------------------------------
def profile_kernel_strict(n_elements, device='cuda'):
    """使用 torch.cuda.Event 对 Triton Kernel 和 PyTorch 原生 AdamW 进行严格的性能分析。"""
    
    # --- 1. 预先分配所有内存 ---
    # Triton 输入
    p = torch.randn(n_elements, device=device, dtype=torch.float32)
    g = torch.randn(n_elements, device=device, dtype=torch.float32)
    m = torch.randn(n_elements, device=device, dtype=torch.float32)
    v = torch.rand(n_elements, device=device, dtype=torch.float32).abs()

    # PyTorch 输入 (为保证公平，使用不同的内存)
    p_torch_param = torch.nn.Parameter(torch.randn_like(p))
    g_torch = torch.randn_like(g)
    m_torch = torch.randn_like(m)
    v_torch = torch.rand_like(v).abs()

    # --- 2. 准备两个优化器 ---
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 0.01
    step = 10
    
    # PyTorch Fused AdamW
    optimizer_torch = torch.optim.AdamW([p_torch_param], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, fused=True)
    p_torch_param.grad = g_torch
    
    # --- 3. 定义计时函数 (不含内存拷贝) ---
    def adamw_triton():
        
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        sqrt_bias_correction2 = math.sqrt(bias_correction2)
        one_minus_beta1 = 1.0 - beta1
        one_minus_beta2 = 1.0 - beta2
        grid = (triton.cdiv(n_elements, 1024),)
        
        fused_adamw_kernel[grid](
            p, g, m, v,
            n_elements, lr,
            bias_correction1, sqrt_bias_correction2,
            beta1, beta2,
            one_minus_beta1, one_minus_beta2,
            eps, weight_decay,
            BLOCK_SIZE=1024,
        )

    def adamw_torch():
        # 在循环外设置一次状态即可，循环内只调用step
        optimizer_torch.step()

    # --- 4. 使用 torch.cuda.Event 进行精确计时 ---
    def benchmark_gpu(func, warmup=25, rep=100):
        # 预热
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()

        # 正式计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(rep):
            func()
        end_event.record()
        torch.cuda.synchronize()
        
        # 返回每次运行的平均毫秒数
        return start_event.elapsed_time(end_event) / rep

    # 设置PyTorch优化器的初始状态
    optimizer_torch.state[p_torch_param]['step'] = torch.tensor(float(step-1), device=device)
    optimizer_torch.state[p_torch_param]['exp_avg'] = m_torch
    optimizer_torch.state[p_torch_param]['exp_avg_sq'] = v_torch
    
    print(f"\nStrict Profiling for {n_elements} elements...")
    
    torch_ms = benchmark_gpu(adamw_torch)
    triton_ms = benchmark_gpu(adamw_triton)

    bytes_per_run = n_elements * 4 * 7 # 3 reads, 4 writes for 32-bit floats
    torch_gbps = bytes_per_run / (torch_ms * 1e-3) / 1e9
    triton_gbps = bytes_per_run / (triton_ms * 1e-3) / 1e9

    print(f"  PyTorch Fused: {torch_ms:.4f} ms, {torch_gbps:.2f} GB/s")
    print(f"  Triton Fused:  {triton_ms:.4f} ms, {triton_gbps:.2f} GB/s")
    
    speedup = torch_ms / triton_ms
    print(f"  Speedup: {speedup:.2f}x")


# --------------------------------------------------
# 5. 主程序入口
# --------------------------------------------------
if __name__ == "__main__":
    for n_elements in [2**20, 2**22, 2**24, 2**26]: # 从小到大测试
        profile_kernel_strict(n_elements)
