# 课程09：KV Cache 原理与实现

> 自回归解码每一步只新增一个 token，却若每次都从头重算整段序列的 K/V，会浪费 \(O(T^2)\) 级别的重复算力；KV Cache 把历史位置的 K、V 存下来，让解码步近似 \(O(T)\) 地增长。

## 本课目标

- 说清楚 **为何需要 KV Cache**（相对「每步全量重算」）。
- 背熟并会推导 **显存估算公式**（与 nano-vllm 张量形状一致）。
- 读懂 **`allocate_kv_cache`**：块数、可用显存、`torch.empty` 六维张量。
- 区分 **Prefill** 与 **Decode** 阶段对 KV Cache 的读写模式。
- 整理 **面试高频追问**（量化、多请求、GQA 对公式的影响）。

## 核心概念

### 1. 注意力里重复计算从何而来

对长度 \(T\) 的序列，第 \(t\) 步若从零计算 attention，需要所有位置 \(1..t\) 的 K、V 参与。但 **第 \(t\) 步新增的只是位置 \(t\) 的 query**；位置 \(1..t-1\) 的 K、V 与上一步相比 **不变**（模型权重与已生成 token 固定时）。

因此可把 **过去所有步已算过的 K、V** 缓存在 GPU 上，本步只算当前 token 的 K、V 并 **追加** 到 cache，再与历史一起做注意力（常配合因果掩码或「只 attend 到过去」的实现）。

**省的是什么**：避免对历史 token 重复做 K/V 投影与（在部分实现中）重复写入中间结果。复杂度从「每步像做一次长序列 prefill」降为「每步常数级或线性于当前上下文」的增量更新（具体常数依赖 kernel 与 head 配置）。

### 2. 经典显存估算公式（与课程一致）

对每层、每个序列位置，需要存 **K** 与 **V** 两个张量，形状与 `num_kv_heads`、`head_dim` 相关。

粗算 **总 KV Cache 字节数**（与下面 nano-vllm 实现维度一致时可写作）：

\[
\text{KV\_bytes} \approx 2 \times L \times T \times H_{kv} \times D \times S_{\mathrm{dtype}} \times B
\]

其中：

- **2**：K 与 V 两份；
- **\(L\)**：`num_hidden_layers`；
- **\(T\)**：序列长度（或当前占用的 token 数上限）；
- **\(H_{kv}\)**：`num_key_value_heads`（注意 **张量并行后要按 rank 取本地头数**）；
- **\(D\)**：`head_dim`；
- **\(S_{\mathrm{dtype}}\)**：单元素字节（fp16=2，bf16=2 等）；
- **\(B\)**：batch 或并行序列数（依系统是否共享/分槽而定）。

面试时强调：**GQA 用 \(H_{kv}\) 而非 \(H\)**，这是与 MHA 公式的重要区别。

### 3. nano-vllm 中的「块」与全局张量

nano-vllm 不是为「每个请求 malloc 一块连续 KV」的简单模式，而是预先分配 **固定块数 × 块大小** 的大池，由 **BlockManager** 管理映射（下一课）。`allocate_kv_cache` 负责 **池子本体** 的显存与 **按层绑定** 到各 attention 模块的 `k_cache` / `v_cache` 视图。

### 4. Prefill vs Decode

- **Prefill（提示阶段）**：一次性处理 prompt，可并行算多个 token 的 Q/K/V，向 KV Cache **写入** 一段连续区间；计算形态常为「大张量、高并行」。
- **Decode（生成阶段）**：每步通常只处理 **1 个新 token**（或少量），对 KV Cache **增量追加**，计算形态常为「小 batch、内存带宽敏感」。

KV Cache 在 decode 阶段的收益最大：若无 cache，每步都要对全长重算历史 K/V，延迟爆炸。

---

## 源码解析：`ModelRunner.allocate_kv_cache`

下面与 `nano-vllm-main/nanovllm/engine/model_runner.py` 一致。

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

### 显存余量：`total * gpu_memory_utilization - used - peak + current`

- **`mem_get_info`**：当前设备「空闲/总」显存。
- **`used = total - free`**：非空闲部分（含框架缓存等，语义以 CUDA 运行时为准）。
- **`peak` / `current`**：分配器统计的峰值与当前分配，用于修正「warmup 已分配但未必常驻」等差异。

整体意图：**在不超过用户设定利用率** 的前提下，估算还能容纳多少 **完整 KV block**。

这是 **vLLM / Nano-vLLM 等 LLM 推理框架的核心显存调度逻辑**，专门用于**精准计算还能分配多少个 KV Cache 块**，避免 OOM，同时最大化利用显存。

## 先记住核心前提
这套公式解决的最大痛点：
**CUDA 系统统计的显存 ≠ 框架实际使用的显存**
（PyTorch 有显存缓存机制，分配后不还给系统，导致 CUDA 统计虚高）

---

## 一、先定义所有变量（来源 + 精准语义）
## 1. 硬件/系统层（来自 CUDA 原生 API `cudaMemGetInfo`）
| 变量 | 含义 | 计算/来源 |
|------|------|-----------|
| `total` | 显卡**总物理显存** | GPU 硬件固定值（如 24GB、48GB） |
| `free` | CUDA 统计的**当前空闲显存** | `mem_get_info` 直接返回 |
| `used` | **CUDA 视角的已用显存** | `used = total - free` |
| `gpu_memory_utilization` | 用户配置的显存利用率上限 | 如 0.9 → 只使用 90% 显存，留 10% 安全余量 |

> `used` 坑点：
> 它包含 **模型权重 + CUDA 上下文 + 框架临时缓存 + 真正分配的张量**，
> 只要框架向 CUDA 申请过，哪怕不用了，也会被算成 `used`，**数值虚高**。

## 2. 框架分配器层（PyTorch 自定义显存分配器统计）
| 变量 | 含义 | 作用 |
|------|------|------|
| `peak` | 框架**历史峰值分配显存** | 框架曾经申请过的最大显存（含已释放、留在缓存里的） |
| `current` | 框架**当前真实常驻显存** | 真正正在使用的显存（KV Cache、激活值、权重等） |

---

# 二、公式完整拆解
## 原始公式
```
显存余量 = total × gpu_memory_utilization  -  used  -  peak  +  current
```

## 分步翻译（每一项的意义）
### 第 1 项：`total × gpu_memory_utilization`
→ **框架允许使用的最大显存上限**
例：24GB 显卡 × 0.9 = 21.6GB
遵守用户配置，不占满显存，防止系统/驱动占用导致 OOM。

### 第 2 项：`- used`
→ 减去 **CUDA 系统视角的总已用显存**
这是系统层面已经被占用的所有显存（包含虚高的缓存）。

### 第 3/4 项：`- peak + current` = **`-(peak - current)`**
→ **核心修正项！消除显存缓存的虚高误差**
`peak - current` = **框架可回收的显存缓存**
（框架曾经分到峰值，现在只用 current，多出来的部分是**闲置缓存，可以释放复用**）

---

# 三、公式等价简化（最容易理解的版本）
```
可分配显存 = 允许使用上限 - (CUDA已用 - 可回收缓存)
```

### 直白解释：
1. CUDA 统计的 `used` 是**假的高**，因为里面藏了框架不用的缓存；
2. 我们把这部分**可回收缓存减掉**，得到**真正常驻、无法释放的显存**；
3. 用允许上限 - 真实常驻显存 = **纯纯可以分给 KV Cache 的空闲显存**。

---

# 四、举个栗例子
已知：
- 显卡总显存 `total = 24GB`
- 显存利用率 `gpu_memory_utilization = 0.9` → 允许上限 `21.6GB`
- CUDA 空闲 `free = 10GB` → `used = 24-10 = 14GB`
- 框架峰值 `peak = 12GB`，当前真实使用 `current = 8GB`
- 可回收缓存 = `12-8 = 4GB`

代入计算：
```
显存余量 = 21.6 - 14 - 12 + 8 = 3.6GB
```
✅ **结果：还有 3.6GB 显存可以分配新的 KV Cache 块**

---

# 五、为什么要这么算？（整体意图）
这套计算是 **LLM 推理 PagedAttention 调度的核心**：
1. KV Cache 是显存占用的**最大动态部分**（请求越多，KV block 越多）；
2. 框架必须**实时精准计算**还能装多少个 KV block；
3. 直接用 CUDA 统计会因为**显存缓存**导致估算错误（以为没空间，其实有）；
4. 用 `peak/current` 修正后，得到**真实可用显存**，从而决定：
   - 还能接收多少个用户请求
   - 最大 Batch 能设多少
   - 不会触发 OOM，同时最大化硬件利用率

---

# 六、终极一句话总结
1. `total×util`：设定**安全显存红线**
2. `used`：系统统计的**粗略已用**（含缓存，不准）
3. `-peak+current`：**修正缓存误差**，得到真实常驻占用
4. 最终结果：**纯可用显存 → 计算还能分配多少 KV block**

这是 LLM 推理框架解决**显存统计失真、精准调度 KV Cache**的标准工程方案。


### `block_bytes` 的含义

单块、单层、单 rank 的 KV 双线？注意公式：

```text
2 * num_layers * block_size * num_kv_heads * head_dim * itemsize
```

这是 **一个 KV cache block slot** 占用的字节数：横跨 **所有层**（\(2 \times L\) 因子把 K/V 与层数都折进「每块成本」），从而 `num_kvcache_blocks = 可用字节 // block_bytes` 得到 **块槽位数**。

（若从维度上理解：`self.kv_cache` 第一维 2 为 K/V；第二维为 layer；块在第三、四维。`block_bytes` 把「一层一块」扩展为「所有层同一 block_id 的总占用」，与 `empty` 形状一致。）

### 六维张量 `self.kv_cache` 形状解析

```text
(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
```

| 维 | 含义 |
|----|------|
| **2** | K 与 V 两个池（索引 0/1） |
| **num_hidden_layers** | 每层独立子张量，便于按层绑定模块 |
| **num_kvcache_blocks** | PagedAttention 的块个数 |
| **block_size** | 每块容纳的 token 槽位数 |
| **num_kv_heads** | 本 rank 上的 KV 头数（已除以 `world_size`） |
| **head_dim** | 每头维度 |

### 按层绑定

遍历 `self.model.modules()`，凡同时具有 `k_cache`、`v_cache` 的模块（各层 Attention），把 **该层** 对应 `layer_id` 的视图指过去：

```text
k_cache = kv_cache[0, layer_id]   # 形状去掉前两维中的 K/V 与 layer
v_cache = kv_cache[1, layer_id]
```

这样前向时模块直接写自己的层切片，无需每层单独 `torch.empty`。

---

## 图解

### KV 随时间追加（概念）

```text
step 0:  [K0]
step 1:  [K0, K1]
...
step t:  [K0 ... Kt]
```

V 同理；实现上落在 block 池的离散块中，而非简单向量追加。

### Prefill vs Decode（对比）

```text
Prefill:  一次写入多个 token 的 KV（并行度高）
Decode:   每步写入 1 个 token（带宽敏感，强依赖 cache）
```

### 与块管理器的关系（预告）

```text
allocate_kv_cache  -->  一大块物理池
BlockManager         -->  逻辑块 <-> 序列 token 的映射表
```

---

## 面试考点

### 为何公式里用 `num_kv_heads` 而不是 `num_heads`

GQA/MQA 下多个 query 头共享 KV 头，缓存只存 **物理 KV 头**。

### 张量并行如何进入公式

每 rank 只存 **本分片** 的 KV 头：`num_kv_heads // world_size`（代码变量名 `num_kv_heads` 已除过）。

### 量化 KV Cache（追问）

INT8/FP8 等降低 \(S_{\mathrm{dtype}}\)，但需反量化或专用 kernel；公式结构不变，改 **每元素字节数** 与 **精度损失** 讨论。

### `assert config.num_kvcache_blocks > 0`

配置过大 `max_model_len`、过高利用率、或显存过小时可能为 0；工程上要报错提示用户调参。

---

## 常见面试题

1. **只有 KV Cache，没有 Q Cache？**  
   每步只需求当前位置的 Q；历史 Q 不参与当前步 attention 的「与过去 token 匹配」时不需要存全历史 Q（标准自回归解码）。

2. **KV Cache 会和梯度一起反传吗？**  
   推理路径无梯度；训练时通常用 FlashAttention 等变体，cache 语义不同。

3. **块大小 `block_size` 影响什么？**  
   粒度 vs 碎片：小块更灵活但元数据开销大；大块可能浪费尾部空间。

4. **为什么要 `warmup_model` 再分配 KV？**  
   先触发峰值分配与 cudnn/cublas 工作区，再扣减 `peak`，使块数估计更接近真实运行（与源码顺序一致）。

5. **batch 变大时 KV 显存线性涨吗？**  
   多序列各占槽位，总占用随并发序列数增加；具体是否线性取决于是否共享前缀、是否分页等。

---

## 小结

KV Cache 避免对历史 token 重复计算 K/V，是低延迟解码的核心；显存可按「层 × 长 × KV 头 × 头维 × 精度 × 2」估算；nano-vllm 用 **单大六维张量 + 按层视图** 管理池，`allocate_kv_cache` 根据 GPU 余量与块成本计算 **可用块数**。Prefill 批量写、Decode 增量写，二者对系统瓶颈（算力 vs 带宽）影响不同。

## 下一课预告

下一课 **PagedAttention 与 BlockManager**：操作系统分页类比、xxhash 前缀块复用、`allocate`/`may_append` 与引用计数，把「块池」真正接到「多请求并发」上。
