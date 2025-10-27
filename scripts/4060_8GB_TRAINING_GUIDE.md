# 4060 Laptop 8GB GPU 训练指南

## 🖥️ 硬件配置
- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **CPU**: Intel i7
- **限制**: 显存仅8GB,需要激进的显存优化

---

## ⚠️ 关键显存优化技术

### 1. **必须开启的优化**
- ✅ `gradient_checkpointing=True` - 牺牲20%速度,节省约40%显存
- ✅ `fp16=True` - 使用FP16混合精度,节省50%显存
- ✅ `model_max_length=1024` - 减少序列长度(从2048降到1024)
- ✅ `BSZPERDEV=1` - 单GPU batch size保持最小

### 2. **显存占用估算**
```
SmolLM2-135M 模型:       ~520MB
FP16激活值(batch=1):     ~1.5GB
梯度和优化器状态:         ~2GB
Gradient checkpointing:  节省~1GB
序列长度1024:            节省~2GB
-----------------------------------
总计:                    约4-5GB (安全范围内)
```

---

## 📋 推荐训练策略

### 策略 A: 稳健优化版 (首选)

**配置文件**: `scripts/sft_4060_8gb.sh`

**关键参数**:
```bash
TOTALBSZ=64              # 有效batch size
BSZPERDEV=1              # 每GPU batch size
GRADACC=64               # 梯度累积步数
learning_rate=4e-5       # 适中的学习率
num_train_epochs=5       # 5个epochs
model_max_length=1024    # 序列长度1024
gradient_checkpointing=True
fp16=True
max_rounds=6
```

**预期效果**:
- 显存占用: ~4.5GB
- 训练时间: 约2-3小时 (4060性能)
- IFEval目标: 23-27%

**运行命令**:
```bash
cd d:\code\COMP_4901B\Assignment2
bash scripts/sft_4060_8gb.sh
```

---

### 策略 B: 激进版 (如果A不够25%)

**配置文件**: `scripts/sft_4060_aggressive.sh`

**关键参数**:
```bash
TOTALBSZ=48              # 更小batch size
BSZPERDEV=1
GRADACC=48
learning_rate=6e-5       # 更高学习率
num_train_epochs=6       # 6个epochs
model_max_length=768     # 更短序列
max_rounds=8             # 更多对话轮次
weight_decay=0.02
```

**预期效果**:
- 显存占用: ~4GB
- 训练时间: 约3-4小时
- IFEval目标: 25-30%

**运行命令**:
```bash
bash scripts/sft_4060_aggressive.sh
```

---

## 🚀 训练步骤

### 1. 准备环境
```bash
cd d:\code\COMP_4901B\Assignment2

# 确认GPU可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 2. 开始训练 (策略A)
```bash
# Windows PowerShell
bash scripts/sft_4060_8gb.sh

# 或者直接用python
python train_hw_parallel.py `
    --model_name_or_path SmolLM2-135M `
    --data_path smol-smoltalk-6k.json `
    --output_dir ckpt/HW2-4060-8GB `
    --num_train_epochs 5 `
    --per_device_train_batch_size 1 `
    --gradient_accumulation_steps 64 `
    --learning_rate 4e-5 `
    --warmup_ratio 0.2 `
    --model_max_length 1024 `
    --fp16 True `
    --gradient_checkpointing True `
    --save_steps 20 `
    --logging_steps 1 `
    --max_rounds 6
```

### 3. 监控训练
观察以下指标:
- **GPU显存使用**: 应该在4-5GB范围内
- **Loss**: 应该从1.7降到1.3-1.5
- **Gradient norm**: 应该稳定在0.7-1.0

如果遇到OOM (Out of Memory):
```bash
# 进一步降低配置
--model_max_length 768
--per_device_train_batch_size 1
--gradient_accumulation_steps 48
```

### 4. 评估模型
```bash
cd ifeval

# 评估最佳checkpoint
bash run.sh ../ckpt/HW2-4060-8GB/checkpoint-100 results/4060-strategy-A

# 试试不同的checkpoints
bash run.sh ../ckpt/HW2-4060-8GB/checkpoint-80 results/4060-cp80
bash run.sh ../ckpt/HW2-4060-8GB/checkpoint-120 results/4060-cp120
```

---

## 📊 性能优化建议

### 1. **加速训练**
- 关闭不必要的后台程序
- 确保笔记本使用高性能模式
- 确保充电器已连接(避免功率限制)
- 关闭浏览器(释放显存和RAM)

### 2. **Windows特定优化**
```powershell
# PowerShell中设置环境变量
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
$env:CUDA_LAUNCH_BLOCKING="0"

# 然后运行训练
python train_hw_parallel.py ...
```

### 3. **如果仍然OOM**
最小配置:
```bash
--model_max_length 512
--per_device_train_batch_size 1
--gradient_accumulation_steps 32
--max_rounds 4
```

---

## 🔍 故障排查

### OOM错误
```
RuntimeError: CUDA out of memory
```
**解决方案**:
1. 降低 `model_max_length` (1024 → 768 → 512)
2. 确保 `gradient_checkpointing=True`
3. 确保 `fp16=True`
4. 重启Python kernel释放显存

### 训练很慢
**原因**: 4060 Laptop GPU + gradient checkpointing
**正常速度**: 约 0.5-1.0 step/秒
**预计时间**: 
- 策略A (320步): 约2-3小时
- 策略B (384步): 约3-4小时

### Loss不下降
**解决方案**:
1. 检查loss_functions.py是否正确实现
2. 检查conversation_func.py的mask是否正确
3. 尝试提高学习率到 5e-5 或 6e-5
4. 增加训练epochs到6-7

---

## 🎯 达到25%的关键

基于8GB显存限制,以下因素最重要:

### 1. **训练轮次** ⭐⭐⭐⭐⭐
- 5-6个epochs通常足够
- 不要训练太多(避免过拟合)

### 2. **学习率** ⭐⭐⭐⭐⭐
- 4e-5 到 6e-5 之间
- warmup_ratio 0.2-0.25

### 3. **序列长度** ⭐⭐⭐⭐
- 1024已经足够大部分任务
- 如果显存允许,试试1280

### 4. **Checkpoint选择** ⭐⭐⭐⭐⭐
- 不要只用最后的checkpoint!
- 试试中间的 (如80%, 90%训练进度)
- 有时候中间的效果最好

### 5. **多轮对话** ⭐⭐⭐
- max_rounds=6-8
- 帮助模型学习复杂指令

---

## 📈 预期结果

### 策略A (稳健版)
- **Strict Accuracy**: 23-27%
- **显存使用**: 4-5GB
- **训练时间**: 2-3小时
- **成功率**: 80%

### 策略B (激进版)
- **Strict Accuracy**: 25-30%
- **显存使用**: 4-4.5GB
- **训练时间**: 3-4小时
- **成功率**: 70% (更激进,风险略高)

---

## 💡 最终建议

1. **先运行策略A** - 稳定可靠
2. **评估多个checkpoints** - 找最佳的
3. **如果<25%,运行策略B** - 更激进的学习
4. **训练时保持笔记本散热良好** - 避免降频

**8GB显存虽然有限,但SmolLM2-135M这个小模型完全可以训练好!**

祝训练顺利! 🚀
