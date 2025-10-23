# 超参数配置对比 - 2080Ti (11GB VRAM)

## 📊 三个配置版本对比

| 参数 | sft_v1.sh (推荐) | sft_v2.sh (激进) | sft_v3.sh (保守) |
|------|------------------|------------------|------------------|
| **策略** | 平衡优化 | 最大性能 | 稳定训练 |
| **Learning Rate** | 3e-5 | 5e-5 | 2.5e-5 |
| **Epochs** | 5 | 6 | 4 |
| **Batch Size/GPU** | 2 | 4 | 1 |
| **Grad Accumulation** | 64 | 32 | 128 |
| **Sequence Length** | 1536 | 1280 | 2048 |
| **Warmup Ratio** | 0.15 | 0.1 | 0.2 |
| **预期训练时间** | ~2.5 小时 | ~2 小时 | ~3.5 小时 |
| **显存使用** | ~9GB | ~10GB | ~8GB |
| **目标准确率** | >23% | >25% | >22% |

## 🎯 推荐使用顺序

### 1️⃣ 首先尝试: **sft_v1.sh** (平衡版本)
**原因**: 
- 在速度和性能之间取得最佳平衡
- 针对 2080Ti 11GB 优化的参数配置
- 使用较高学习率 (3e-5) 加速收敛
- 5 个 epoch 足够学习指令遵循能力
- Batch size = 2 充分利用显存但不会 OOM

**运行方式**:
```bash
cd d:\code\COMP_4901B\Assignment2
bash scripts/sft_v1.sh
```

**预期结果**: Strict accuracy 23-25%

---

### 2️⃣ 如果 v1 结果不理想: **sft_v2.sh** (激进版本)
**适用场景**:
- v1 的结果在 22-23% 之间,需要进一步提升
- 愿意承担更高学习率可能带来的不稳定性
- 想要追求最高分数 (25-35 分档)

**特点**:
- 最高学习率 (5e-5) - 学习更快但可能不稳定
- 6 个 epoch - 确保充分学习
- Batch size = 4 - 更大批次但需要更短序列
- Sequence length = 1280 - 牺牲一些上下文换取更大批次

**运行方式**:
```bash
cd d:\code\COMP_4901B\Assignment2
bash scripts/sft_v2.sh
```

**预期结果**: Strict accuracy 25-28% (如果稳定训练)

**风险**: 高学习率可能导致训练不稳定,需要监控 loss 曲线

---

### 3️⃣ 如果追求稳定: **sft_v3.sh** (保守版本)
**适用场景**:
- v1 或 v2 训练过程中 loss 震荡
- 优先保证 > 22% 的基准分数
- 时间充足,可以接受较慢的训练

**特点**:
- 中等学习率 (2.5e-5) - 最稳定
- 4 个 epoch - 标准训练长度
- 完整序列长度 (2048) - 最大上下文
- 长 warmup (0.2) - 避免早期震荡

**运行方式**:
```bash
cd d:\code\COMP_4901B\Assignment2
bash scripts/sft_v3.sh
```

**预期结果**: Strict accuracy 22-24%

---

## 🚀 完整工作流程

### 步骤 1: 训练模型
```bash
# 使用推荐的 v1 配置
cd d:\code\COMP_4901B\Assignment2
bash scripts/sft_v1.sh
```

训练期间监控:
- Training loss 应该持续下降
- 如果 loss 震荡或上升,说明学习率过高
- 正常情况下,loss 会从 ~2.5 降到 ~1.5-1.8

### 步骤 2: 评估模型
```bash
cd ifeval

# 评估基础模型 (对比用)
python run_ifeval_local.py --mode all --model_path ../SmolLM2-135M --output_dir ./results/base

# 评估训练后的模型
python run_ifeval_local.py --mode all --model_path ../ckpt/HW2_v1 --output_dir ./results/HW2_v1
```

### 步骤 3: 查看结果
```powershell
# 查看基础模型结果
Get-Content .\results\base\summary.json | ConvertFrom-Json | Format-List

# 查看微调模型结果
Get-Content .\results\HW2_v1\summary.json | ConvertFrom-Json | Format-List

# 对比
$base = Get-Content .\results\base\summary.json | ConvertFrom-Json
$ft = Get-Content .\results\HW2_v1\summary.json | ConvertFrom-Json

Write-Host "Base Model Strict Accuracy: $($base.strict_accuracy_percentage)%"
Write-Host "Fine-tuned Model Strict Accuracy: $($ft.strict_accuracy_percentage)%"
Write-Host "Improvement: +$($ft.strict_accuracy_percentage - $base.strict_accuracy_percentage)%"
```

### 步骤 4: 根据结果调整

| 结果 | 下一步行动 |
|------|------------|
| **Strict Accuracy > 25%** | 🎉 完美! 已达到满分标准 |
| **Strict Accuracy 23-25%** | ✅ 很好! 可以提交,或尝试 v2 冲击更高分 |
| **Strict Accuracy 22-23%** | 🔄 达标但可改进,建议尝试 v2 (激进版本) |
| **Strict Accuracy 20-22%** | ⚠️ 需要改进,尝试 v2 或检查实现 |
| **Strict Accuracy < 20%** | ❌ 检查 loss 函数和 loss masking 实现 |

---

## 💡 关键超参数解释

### Learning Rate (学习率)
- **2e-5** (原始): 安全但可能学习慢
- **2.5e-5** (v3): 稳定的中等速度
- **3e-5** (v1 推荐): 较快学习,仍然稳定
- **5e-5** (v2): 最快学习,可能不稳定

### Epochs (训练轮数)
- 太少 (< 3): 欠拟合,学习不充分
- 适中 (4-5): 最佳性能
- 太多 (> 6): 可能过拟合,在小数据集上风险较高

### Batch Size
- **实际批次** (BSZPERDEV): 每个 GPU 同时处理的样本数
- **有效批次** (TOTALBSZ): 通过梯度累积模拟的总批次大小
- 更大批次 = 更稳定的梯度,但需要更多显存

### Sequence Length
- **1280**: 适合 batch_size=4,牺牲一些上下文
- **1536**: 平衡选择,batch_size=2
- **2048**: 最大上下文,但只能 batch_size=1

### Warmup Ratio
- 前 N% 的训练步骤用于学习率预热
- 避免一开始就用高学习率导致不稳定
- 0.1-0.2 是常见选择

---

## 📈 预期性能对比

基于类似硬件的经验数据:

| 配置 | 预期 Strict Acc | 训练时间 | 稳定性 | 推荐度 |
|------|----------------|----------|--------|--------|
| **sft_v1** | 23-25% | ~2.5h | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **sft_v2** | 25-28% | ~2h | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **sft_v3** | 22-24% | ~3.5h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **原始 sft** | 20-22% | ~2h | ⭐⭐⭐⭐ | ⭐⭐ |

---

## 🔍 故障排查

### 问题 1: CUDA Out of Memory
**解决方案**:
```bash
# 降低 batch size
# 在脚本中修改: BSZPERDEV=1

# 或降低序列长度
# 在脚本中修改: --model_max_length 1024
```

### 问题 2: Training Loss 不下降
**可能原因**:
- 学习率太低 → 尝试 v1 或 v2
- Loss 函数实现错误 → 检查 `loss_functions.py`
- Loss masking 错误 → 检查 `conversation_func.py`

### 问题 3: Training Loss 震荡
**可能原因**:
- 学习率太高 → 尝试 v3 (保守版本)
- Batch size 太小 → 增加 TOTALBSZ 到 256

### 问题 4: 评估时间太长
**解决方案**:
```bash
# 使用更大的 batch size 进行推理
python run_ifeval_local.py \
    --mode all \
    --model_path ../ckpt/HW2_v1 \
    --output_dir ./results/HW2_v1 \
    --batch_size 8  # 增加到 8
```

---

## 📝 报告中需要包含的内容

记录所有实验结果,用于报告的 Part 5:

### 超参数调优表格示例:

| 配置 | LR | Epochs | Batch | Seq Len | Train Loss | Strict Acc | Loose Acc |
|------|-----|--------|-------|---------|------------|------------|-----------|
| Base Model | - | - | - | - | - | ~22% | ~25% |
| v1 | 3e-5 | 5 | 2 | 1536 | 1.65 | 24.5% | 27.8% |
| v2 | 5e-5 | 6 | 4 | 1280 | 1.58 | 26.2% | 29.1% |

### 分析要点:
1. **哪个超参数影响最大**: Learning rate 和 epochs
2. **为什么这些参数有效**: 更高的 LR 加速学习,更多 epochs 确保收敛
3. **改进的指令类型**: 格式化要求 (段落数、字数限制)
4. **仍然困难的指令**: 复杂的多步骤推理

---

## 🎓 总结

**最佳策略**: 
1. 先跑 **sft_v1.sh** (推荐配置)
2. 如果结果 > 23%,直接提交或尝试 v2 冲击更高分
3. 如果结果 < 23%,检查代码实现或尝试其他配置

**时间规划**:
- 每个配置训练 2-3.5 小时
- 评估每个模型 15-30 分钟
- 建议至少尝试 2 个配置进行对比

祝训练顺利! 🚀
