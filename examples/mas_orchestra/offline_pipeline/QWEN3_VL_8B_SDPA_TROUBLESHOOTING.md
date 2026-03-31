# Qwen3-VL 8B 在 rllm 里的 Attention 兼容性处理

这份说明记录一下这次问题是怎么定位和处理的，也把环境侧和运行参数侧需要改的地方一起整理下来，后面再遇到类似情况可以直接按这份文档走。

## 背景

这次的问题不在数据，也不在模型权重本身，而是在 attention backend 的选择上。

`Qwen3-VL 8B` 的视觉分支里，单个 attention head 的维度是 `72`。这个值本身没有问题，但我们这套 `rllm + veRL + vLLM` 组合里默认会优先走 Flash Attention 相关路径，而这条路径在当前环境下对 head dim 有更严格的要求，最后就会报出类似下面的错误：

- `This flash attention build does not support headdim not being a multiple of 32`
- `FlashAttention2 has been toggled on, but it cannot be used`

所以这次的核心思路不是继续硬怼 Flash Attention，而是统一把这条链路切到更稳的 `SDPA` 路径上。

## 解决思路

逻辑上可以把问题拆成两段：

1. `veRL` 负责 actor / critic 这条训练链路，它在当前版本里会默认把 `attn_implementation` 设成 `flash_attention_2`。
2. `vLLM` 负责 rollout，这边的视觉编码器也会优先回到 Flash Attention 路径。

只改一边是不够的。前者不改，会在模型初始化阶段报 `flash_attn` 相关错误；后者不改，会在视觉编码器真正跑起来时继续撞上 `head dim = 72` 的限制。

这次的处理方式就是：

- `veRL` 这边默认从 `flash_attention_2` 改成 `sdpa`
- `vLLM` 这边把视觉 attention backend 固定到 `TORCH_SDPA`
- `rllm` 启动参数里也显式避开会重新触发 Flash Attention 的配置

## 推荐环境

如果是新建环境，建议尽量沿着项目当前这条版本线来，不要在同一个环境里频繁来回切大版本：

- Python `3.11`
- `torch 2.8.x`
- `torchvision 0.23.x`
- `torchaudio 2.8.x`
- `verl==0.6.1`
- `vllm==0.11.0`

这一套不是唯一能跑的组合，但和仓库当前依赖更接近，后面踩到兼容性坑的概率会低一些。

如果已经有现成环境，建议先把环境路径抽成变量，后面改文件时就不用把本机路径写死在命令里了：

```bash
export CONDA_ENV="${CONDA_PREFIX:-/path/to/your/conda/env}"
export SITE_PKGS="$CONDA_ENV/lib/python3.11/site-packages"
```

如果 `CONDA_PREFIX` 已经有值，第一行会直接复用当前激活的 conda 环境。

## 环境侧建议

### 1. 尽量不要让这个环境继续优先走 Flash Attention

如果这个环境就是专门拿来跑 `Qwen3-VL 8B` 的训练，建议尽量保持简单，不要把太多会干扰 backend 选择的包混在一起。

`flash-attn` 和 `flashinfer` 不是一定要卸，但这次问题里它们都可能把程序重新带回 Flash Attention 路径。如果只是想先把训练稳定跑通，保守一点反而更省事。

如果要清理这类包，可以在当前环境里执行：

```bash
pip uninstall -y flash-attn flash_attn flashinfer-python flashinfer-cubin
```

如果后面别的模型还需要这些包，最好单独分一个环境，不要和这套 `Qwen3-VL 8B` 训练环境混在一起。

### 2. `verl==0.6.1` 记得配合 `numpy<2`

这个不是这次主问题，但如果环境里 `numpy` 已经被拉到 `2.x`，`verl 0.6.1` 会额外报依赖不兼容。

```bash
pip install "numpy<2"
```

## 代码热修复

下面这些改动本质上是临时 hotfix。只要上游还没把这条路径修顺，当前环境里就需要保留这些修改。

### 1. 先备份原文件

```bash
cp "$SITE_PKGS/verl/workers/fsdp_workers.py" \
   "$SITE_PKGS/verl/workers/fsdp_workers.py.bak"

cp "$SITE_PKGS/vllm/model_executor/models/qwen3_vl.py" \
   "$SITE_PKGS/vllm/model_executor/models/qwen3_vl.py.bak"

cp "$SITE_PKGS/vllm/model_executor/models/qwen2_5_vl.py" \
   "$SITE_PKGS/vllm/model_executor/models/qwen2_5_vl.py.bak"
```

### 2. 把 veRL 的默认 attention 实现改成 `sdpa`

当前版本的 `fsdp_workers.py` 里会把默认值写成 `flash_attention_2`，需要改掉：

```bash
sed -i '318s/flash_attention_2/sdpa/' \
  "$SITE_PKGS/verl/workers/fsdp_workers.py"

sed -i '1255s/flash_attention_2/sdpa/' \
  "$SITE_PKGS/verl/workers/fsdp_workers.py"

sed -i '1672s/flash_attention_2/sdpa/' \
  "$SITE_PKGS/verl/workers/fsdp_workers.py"
```

这样改的目的很简单：actor / critic 在通过 `transformers.from_pretrained(...)` 初始化时，不要再默认去找 `FlashAttention2`。

### 3. 把 vLLM 视觉分支固定到 `TORCH_SDPA`

`Qwen3-VL` 和 `Qwen2.5-VL` 这两个文件里都会根据环境再去选视觉 attention backend。我们这里直接把它固定到 `TORCH_SDPA`，避免它再次回退到 Flash Attention。

```bash
sed -i '323,330c\
        self.attn_backend = _Backend.TORCH_SDPA\
        use_upstream_fa = False' \
  "$SITE_PKGS/vllm/model_executor/models/qwen3_vl.py"

sed -i '620,626c\
        self.attn_backend = _Backend.TORCH_SDPA\
        use_upstream_fa = False' \
  "$SITE_PKGS/vllm/model_executor/models/qwen2_5_vl.py"
```

这一步的作用是让 rollout 阶段的视觉编码器别再走 `vllm_flash_attn` 那条路径。

### 4. 改完后做一次语法检查

```bash
python -m py_compile \
  "$SITE_PKGS/verl/workers/fsdp_workers.py" \
  "$SITE_PKGS/vllm/model_executor/models/qwen3_vl.py" \
  "$SITE_PKGS/vllm/model_executor/models/qwen2_5_vl.py"
```

如果这一条能通过，说明至少文件结构没有被改坏。

## rllm 运行参数

除了环境和代码本身，训练命令里的参数也要一起收紧，不然程序还是有机会重新绕回 Flash Attention。

### 1. 运行前的环境变量

```bash
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
unset VLLM_USE_FLASHINFER_SAMPLER
```

这里的意思是：

- `VLLM_ATTENTION_BACKEND=TORCH_SDPA`：告诉 rollout 这边直接走 PyTorch 的 `SDPA`
- `unset VLLM_USE_FLASHINFER_SAMPLER`：避免 FlashInfer 相关逻辑继续插进来，减少干扰项

如果看到类似下面的 warning，一般是正常的：

```text
FlashInfer is not available. Falling back to the PyTorch-native implementation...
```

这说明它已经在走更保守的 fallback 路径了，不是新的故障。

### 2. `train_offline.sh` 建议这样传

`Qwen3-VL 8B` 这边，最重要的是不要把 `use_remove_padding` 再打开。

推荐启动方式：

```bash
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
unset VLLM_USE_FLASHINFER_SAMPLER

CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
N_GPUS_PER_NODE=2 \
TP_SIZE=1 \
./examples/mas_orchestra/offline_pipeline/train_offline.sh \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.rollout.enforce_eager=True
```

### 3. 这些参数为什么要改

#### `actor_rollout_ref.model.use_remove_padding=False`

这个开关在很多模型上有助于性能，但在这次问题里，它很容易把 `transformers` 侧又带回 Flash Attention 相关路径。对于 `Qwen3-VL 8B`，先把它关掉更稳。

#### `actor_rollout_ref.rollout.enforce_eager=True`

这个不是为了修复 head dim 本身，而是为了让 rollout 初始化过程更稳定一些，减少 graph capture 或 backend 选择带来的额外变量。问题先跑通，再考虑把它放开做性能优化。

#### `TP_SIZE=1`

先用最简单的张量并行设置验证问题是否消失。注意力 backend 这类问题本来就比较绕，先减少变量比较容易定位。

## 重启 worker

改完代码后，一定要把旧的 Ray 和 vLLM 进程清掉。否则很容易出现文件已经改了，但老进程还在跑旧代码的情况。

```bash
ray stop --force
pkill -f vllm
pkill -f ray
```

然后重新 export 环境变量，再重新启动训练。

## 如果还要回滚

如果后面上游版本已经修掉这个问题，或者想回到原始文件，可以直接用备份恢复：

```bash
cp "$SITE_PKGS/verl/workers/fsdp_workers.py.bak" \
   "$SITE_PKGS/verl/workers/fsdp_workers.py"

cp "$SITE_PKGS/vllm/model_executor/models/qwen3_vl.py.bak" \
   "$SITE_PKGS/vllm/model_executor/models/qwen3_vl.py"

cp "$SITE_PKGS/vllm/model_executor/models/qwen2_5_vl.py.bak" \
   "$SITE_PKGS/vllm/model_executor/models/qwen2_5_vl.py"
```

## 一句话总结

这次不是模型有问题，而是 `Qwen3-VL 8B` 的视觉 head dim 是 `72`，当前默认的 Flash Attention 路径对这种情况不友好。解决办法不是继续硬配 Flash Attention，而是把 `veRL` 和 `vLLM` 两边都明确切到 `SDPA`，再把训练参数里容易把程序带回 Flash Attention 的开关关掉。
