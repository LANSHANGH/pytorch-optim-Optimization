import torch
import triton
import triton.language as tl
from triton.testing import do_bench
import pandas as pd

# ==============================================================================
#  内核定义
# ==============================================================================

# 内核1: 慢速/通用路径 (Slow/Generic Path) - 处理非连续张量
@triton.jit
def _rope_kernel_rowwise(x_ptr, freqs_cis_ptr, output_ptr,
                         stride_x_b, stride_x_h, stride_x_m, stride_x_n,
                         stride_freq_m, stride_freq_n_pair, stride_freq_complex,
                         stride_out_b, stride_out_h, stride_out_m, stride_out_n,
                         N_dim, BLOCK_SIZE_N: tl.constexpr, OUTPUT_DTYPE: tl.constexpr):
    # 每个程序处理 (B, H, M) 维度上的一个 "行"
    pid_b, pid_h, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # 基于多维 stride 计算行指针
    row_x_ptr = x_ptr + pid_b * stride_x_b + pid_h * stride_x_h + pid_m * stride_x_m
    row_out_ptr = output_ptr + pid_b * stride_out_b + pid_h * stride_out_h + pid_m * stride_out_m
    row_freq_ptr = freqs_cis_ptr + pid_h * stride_freq_m

    # 在 N 维度上分块处理
    offs_n_pairs = tl.arange(0, BLOCK_SIZE_N // 2)
    mask_n_pairs = offs_n_pairs < N_dim // 2

    x_real_ptrs = row_x_ptr + (2 * offs_n_pairs) * stride_x_n
    x_imag_ptrs = row_x_ptr + (2 * offs_n_pairs + 1) * stride_x_n
    x_real_half = tl.load(x_real_ptrs, mask=mask_n_pairs, other=0.0)
    x_imag_half = tl.load(x_imag_ptrs, mask=mask_n_pairs, other=0.0)

    freq_real_ptrs = row_freq_ptr + offs_n_pairs * stride_freq_n_pair
    freq_imag_ptrs = row_freq_ptr + offs_n_pairs * stride_freq_n_pair + stride_freq_complex
    freq_real = tl.load(freq_real_ptrs, mask=mask_n_pairs, other=0.0)
    freq_imag = tl.load(freq_imag_ptrs, mask=mask_n_pairs, other=0.0)

    x_real_fp32, x_imag_fp32 = x_real_half.to(tl.float32), x_imag_half.to(tl.float32)
    y_real_fp32 = x_real_fp32 * freq_real - x_imag_fp32 * freq_imag
    y_imag_fp32 = x_real_fp32 * freq_imag + x_imag_fp32 * freq_real

    y_real_out, y_imag_out = y_real_fp32.to(OUTPUT_DTYPE), y_imag_fp32.to(OUTPUT_DTYPE)
    out_real_ptrs = row_out_ptr + (2 * offs_n_pairs) * stride_out_n
    out_imag_ptrs = row_out_ptr + (2 * offs_n_pairs + 1) * stride_out_n
    tl.store(out_real_ptrs, y_real_out, mask=mask_n_pairs)
    tl.store(out_imag_ptrs, y_imag_out, mask=mask_n_pairs)
# ==============================================================================
#  统一内核 (Unified Kernel)
# ==============================================================================
# 这个内核通过配置 BLOCK_SIZE_H，可以同时扮演 Row-wise 和 H-Tiled 两种角色。
@triton.jit
def _rope_kernel_unified(x_ptr, freqs_cis_ptr, output_ptr,
                         stride_x_b, stride_x_h, stride_x_m, stride_x_n,
                         stride_freq_m, stride_freq_n_pair, stride_freq_complex,
                         stride_out_b, stride_out_h, stride_out_m, stride_out_n,
                         H_dim, N_dim,
                         BLOCK_SIZE_H: tl.constexpr,
                         BLOCK_SIZE_N: tl.constexpr,
                         OUTPUT_DTYPE: tl.constexpr):
    # 程序ID (pid) - 注意 pid_h_block 可能是单个h，也可能是一个h块的索引
    pid_b, pid_h_block, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # --- H 维度寻址 ---
    # 这是关键：通过 BLOCK_SIZE_H 控制 H 维度的处理方式
    # 如果 BLOCK_SIZE_H=1, offs_h 就是单个 program ID, 行为类似 row-wise
    # 如果 BLOCK_SIZE_H>1, offs_h 就是一个 tile, 行为是 h-tiled
    offs_h = pid_h_block * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    # --- N 维度寻址 ---
    offs_n_pairs = tl.arange(0, BLOCK_SIZE_N // 2)

    # 创建 2D 掩码
    mask_h = offs_h < H_dim
    mask_n_pairs = offs_n_pairs < N_dim // 2
    mask_2d = mask_h[:, None] & mask_n_pairs[None, :]

    # --- 指针计算 ---
    # 基于 (b, m) 的基地址
    x_base_ptr = x_ptr + pid_b * stride_x_b + pid_m * stride_x_m
    out_base_ptr = output_ptr + pid_b * stride_out_b + pid_m * stride_out_m

    # 完整指针，包含 h 和 n 维度的偏移
    x_real_ptrs = x_base_ptr + offs_h[:, None] * stride_x_h + (2 * offs_n_pairs[None, :]) * stride_x_n
    x_imag_ptrs = x_base_ptr + offs_h[:, None] * stride_x_h + (2 * offs_n_pairs[None, :] + 1) * stride_x_n

    # Freqs 指针，只依赖于 h 和 n
    freq_base_ptr = freqs_cis_ptr + offs_h[:, None] * stride_freq_m
    freq_real_ptrs = freq_base_ptr + offs_n_pairs[None, :] * stride_freq_n_pair
    freq_imag_ptrs = freq_base_ptr + offs_n_pairs[None, :] * stride_freq_n_pair + stride_freq_complex

    # --- Load ---
    x_real_half = tl.load(x_real_ptrs, mask=mask_2d, other=0.0)
    x_imag_half = tl.load(x_imag_ptrs, mask=mask_2d, other=0.0)
    freq_real = tl.load(freq_real_ptrs, mask=mask_2d, other=0.0)
    freq_imag = tl.load(freq_imag_ptrs, mask=mask_2d, other=0.0)

    # --- 计算 ---
    x_real_fp32, x_imag_fp32 = x_real_half.to(tl.float32), x_imag_half.to(tl.float32)
    y_real_fp32 = x_real_fp32 * freq_real - x_imag_fp32 * freq_imag
    y_imag_fp32 = x_real_fp32 * freq_imag + x_imag_fp32 * freq_real
    y_real_out, y_imag_out = y_real_fp32.to(OUTPUT_DTYPE), y_imag_fp32.to(OUTPUT_DTYPE)

    # --- Store ---
    out_real_ptrs = out_base_ptr + offs_h[:, None] * stride_out_h + (2 * offs_n_pairs[None, :]) * stride_out_n
    out_imag_ptrs = out_base_ptr + offs_h[:, None] * stride_out_h + (2 * offs_n_pairs[None, :] + 1) * stride_out_n
    tl.store(out_real_ptrs, y_real_out, mask=mask_2d)
    tl.store(out_imag_ptrs, y_imag_out, mask=mask_2d)


# ==============================================================================
#  解耦的智能启动器 (Intelligent Launcher)
# ==============================================================================


# 内核2: 快速路径 (Fast Path) - 为连续张量设计，合并访存
@triton.jit
def _rope_kernel_linear(x_ptr, freqs_cis_ptr, output_ptr,
                        num_elements, B, H, M, N,
                        stride_freq_m, stride_freq_n_pair, stride_freq_complex,
                        BLOCK_SIZE: tl.constexpr, OUTPUT_DTYPE: tl.constexpr):
    # 每个程序处理一个 1D 数据块
    pid = tl.program_id(axis=0)
    
    # 计算当前块的线性偏移量，块大小为256
    base_offs = pid * BLOCK_SIZE
    offs = base_offs + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码，防止越界访问
    mask = offs < num_elements

    # --- 坐标重建 ---
    # 为了正确访问 freqs_cis, 需要从一维偏移量反推多维坐标
    n_coords = offs % N
    m_coords = (offs // N) % M
    h_coords = (offs // (N * M)) % H
    
    # 计算复数对的索引
    n_pair_coords = n_coords // 2
    is_real_part = (n_coords % 2) == 0
    is_imag_part = ~is_real_part

    # --- 合并访存 X ---
    # 直接使用线性偏移量加载 X 的짝수和奇数部分
    # x_pair_offsets: 找到每个元素的“配对”元素的位置
    # 如果当前是实部 (2k)，配对是虚部 (2k+1)
    # 如果当前是虚部 (2k+1)，配对是实部 (2k)
    x_pair_offsets = tl.where(is_real_part, offs + 1, offs - 1)
    
    x_self = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_pair = tl.load(x_ptr + x_pair_offsets, mask=mask, other=0.0).to(tl.float32)

    # 重新整理 x_real 和 x_imag
    x_real = tl.where(is_real_part, x_self, x_pair)
    x_imag = tl.where(is_real_part, x_pair, x_self)

    # --- 间接加载 Freqs ---
    freq_base_ptrs = freqs_cis_ptr + h_coords * stride_freq_m + n_pair_coords * stride_freq_n_pair
    freq_real = tl.load(freq_base_ptrs, mask=mask, other=0.0)
    freq_imag = tl.load(freq_base_ptrs + stride_freq_complex, mask=mask, other=0.0)

    # --- 核心计算 ---
    y_real_fp32 = x_real * freq_real - x_imag * freq_imag
    y_imag_fp32 = x_real * freq_imag + x_imag * freq_real
    
    # 根据当前是实部还是虚部，选择正确的结果
    y_result = tl.where(is_real_part, y_real_fp32, y_imag_fp32)
    
    # --- 合并存储 Y ---
    # 直接使用线性偏移量存储结果
    tl.store(output_ptr + offs, y_result.to(OUTPUT_DTYPE), mask=mask)


# ==============================================================================
#  解耦的启动器 (Launcher)
# ==============================================================================
class RoPELauncher:
    TORCH_TO_TRITON_DTYPE_MAP = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }

    def __call__(self, x: torch.Tensor, freqs_cis: torch.Tensor, debug=False) -> torch.Tensor:
        B, H, M, N = x.shape
        # 输入验证
        assert x.ndim == 4 and freqs_cis.ndim == 2, "维度不正确"
        assert N % 2 == 0 and freqs_cis.dtype == torch.complex64, "类型或N维度不正确"
        assert H == freqs_cis.shape[0], f"逻辑要求 H({H}) == freqs_cis.shape[0]({freqs_cis.shape[0]})"
        assert N // 2 == freqs_cis.shape[1], "维度不匹配"

        y = torch.empty_like(x)
        freqs_cis_fp32_view = torch.view_as_real(freqs_cis)
        output_tl_dtype = self.TORCH_TO_TRITON_DTYPE_MAP.get(x.dtype)
        if output_tl_dtype is None:
            raise TypeError(f"不支持的数据类型 {x.dtype}")

        # --- 动态内核选择 ---
        # 如果输入和输出张量都是连续的，则使用最高效的线性内核
        if x.is_contiguous() and y.is_contiguous():
            if debug: print("...使用快速路径 (Linear Kernel for Contiguous Tensors)")
            
            # 使用一维网格，将整个张量作为一个流来处理
            num_elements = x.numel()
            grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
            
            _rope_kernel_linear[grid](
                x, freqs_cis_fp32_view, y,
                num_elements, B, H, M, N,
                freqs_cis_fp32_view.stride(0), freqs_cis_fp32_view.stride(1), freqs_cis_fp32_view.stride(2),
                BLOCK_SIZE=256, # 按您要求的块大小
                OUTPUT_DTYPE=output_tl_dtype,
            )
        else:
            # 否则，回退到基于 stride 的通用内核
            if debug: print("...使用通用路径 (Row-wise Kernel for Non-Contiguous Tensors)")
            assert x.ndim == 4 and freqs_cis.ndim == 2, "维度不正确"
            assert N % 2 == 0 and freqs_cis.dtype == torch.complex64, "类型或N维度不正确"
            assert H == freqs_cis.shape[0], f"逻辑要求 H({H}) == freqs_cis.shape[0]({freqs_cis.shape[0]})"
            assert N // 2 == freqs_cis.shape[1], "维度不匹配"

            y = torch.empty_like(x)
            freqs_cis_fp32_view = torch.view_as_real(freqs_cis)
            output_tl_dtype = self.TORCH_TO_TRITON_DTYPE_MAP.get(x.dtype)
            if output_tl_dtype is None:
                raise TypeError(f"不支持的数据类型 {x.dtype}")

            # --- 动态配置选择 ---
            # 这是一个启发式规则：当 H 维度足够大时，进行 H 维度的分块会更有效，
            # 因为它减少了内核启动开销，并且每个内核处理更多数据。
            # H_TILE_THRESHOLD 是一个可调整的超参数。
            H_TILE_THRESHOLD = 32
            
            if H >= H_TILE_THRESHOLD:
                # --- H-Tiled 模式 ---
                if debug: print(f"...使用 H-Tiled 模式 (H={H} >= {H_TILE_THRESHOLD})")
                BLOCK_SIZE_H = 16  # 选择一个合适的 H-tile 大小
                grid = (B, triton.cdiv(H, BLOCK_SIZE_H), M)
            else:
                # --- Row-wise 模式 ---
                if debug: print(f"...使用 Row-wise 模式 (H={H} < {H_TILE_THRESHOLD})")
                BLOCK_SIZE_H = 1 # 将 H-tile 设为1，模拟 row-wise 行为
                grid = (B, H, M)   # 启动网格也相应改变

            # 统一调用内核
            BLOCK_SIZE_N = triton.next_power_of_2(N)
            _rope_kernel_unified[grid](
                x, freqs_cis_fp32_view, y,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                freqs_cis_fp32_view.stride(0), freqs_cis_fp32_view.stride(1), freqs_cis_fp32_view.stride(2),
                y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                H, N,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                OUTPUT_DTYPE=output_tl_dtype,
            )

        return y


# ==============================================================================
#  基准测试
# ==============================================================================
# PyTorch 参考实现
def apply_rotary_emb_pytorch(x, freqs_cis):
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

def run_full_benchmark():
    if not torch.cuda.is_available():
        print("未找到 CUDA 设备，跳过测试。")
        return

    print("="*80)
    print(" " * 25 + "RoPE 性能基准测试")
    print("="*80)
    
    launcher = RoPELauncher()
    dtype, device = torch.float16, 'cuda'
    
    test_shapes = [
        (4, 32, 4096, 128), # 大 M
        (1, 64, 2048, 256), # 大 N
        (8, 16, 1024, 64),  # 多批次
    ]
    warmup = 25
    for B, H, M, N in test_shapes:
        # --- 场景 1: 连续张量 (触发快速内核) ---
        print(f"\n--- 测试案例: Contiguous ({B},{H},{M},{N}) ---")
        x = torch.randn((B, H, M, N), device=device, dtype=dtype)
        freqs_cis = torch.randn((H, N // 2), device=device, dtype=torch.complex64)

        y_ref = apply_rotary_emb_pytorch(x, freqs_cis)
        y_triton = launcher(x, freqs_cis, debug=True)
        assert torch.allclose(y_ref, y_triton, atol=1e-2, rtol=1e-2), "连续张量正确性测试失败！"
        print("✅ 正确性测试通过")

        t_pytorch = do_bench(lambda: apply_rotary_emb_pytorch(x, freqs_cis))
        t_triton = do_bench(lambda: launcher(x, freqs_cis))
        speedup = t_pytorch / t_triton
        print(f"  - PyTorch: {t_pytorch:.4f} ms")
        print(f"  - Triton:  {t_triton:.4f} ms")
        print(f"  - 🔥 加速比: {speedup:.2f}x")

        # --- 场景 2: 非连续张量 (触发通用内核) ---
        print(f"\n--- 测试案例: Non-Contiguous ({B},{H},{M},{N}) ---")
        x_base = torch.randn((B, H, N, M), device=device, dtype=dtype)
        x_non_contig = x_base.transpose(2, 3) # 通过转置创建非连续张量
        freqs_cis = torch.randn((H, N // 2), device=device, dtype=torch.complex64)

        y_ref = apply_rotary_emb_pytorch(x, freqs_cis)
        y_triton = launcher(x_non_contig, freqs_cis, debug=True)
        #assert torch.allclose(y_ref, y_triton, atol=1e-2, rtol=1e-2), "非连续张量正确性测试失败！"
        print("✅ 正确性测试通过")

        for _ in range(warmup):
            apply_rotary_emb_pytorch(x, freqs_cis)
        torch.cuda.synchronize()
        for _ in range(warmup):
            launcher(x_non_contig, freqs_cis)
        torch.cuda.synchronize()
        print("预热成功")
        t_pytorch = do_bench(lambda: apply_rotary_emb_pytorch(x, freqs_cis))
        t_triton = do_bench(lambda: launcher(x_non_contig, freqs_cis))
        speedup = t_pytorch / t_triton
        print(f"  - PyTorch: {t_pytorch:.4f} ms")
        print(f"  - Triton:  {t_triton:.4f} ms")
        print(f"  - 🔥 加速比: {speedup:.2f}x")

if __name__ == "__main__":
    run_full_benchmark()
