# IFEval with Transformers (No vLLM Required)

这个文档说明如何在不使用 vLLM 的情况下运行 IFEval 评估。

## 概述

原始的 IFEval 实现依赖于 vLLM 进行推理,但 vLLM 在某些环境下(如 Windows、旧版 GPU 等)可能无法正常工作。我们提供了一个基于 HuggingFace Transformers 的替代方案,可以在任何支持 PyTorch 的环境中运行。

## 文件说明

- `run_ifeval_transformers.py` - 主要的评估脚本(使用 Transformers)
- `inference_transformers.py` - 独立的推理脚本(使用 Transformers)
- `run_transformers.sh` - 便捷的运行脚本
- `run_ifeval.py` - 原始的评估脚本(使用 vLLM)
- `inference_vllm.py` - 原始的推理脚本(使用 vLLM)
- `run.sh` - 原始的运行脚本(使用 vLLM)

## 快速开始

### 方法 1: 使用便捷脚本(推荐)

在 Git Bash 中运行:

```bash
cd ifeval

# 评估基础模型
bash run_transformers.sh ../SmolLM2-135M results/base

# 评估微调后的模型
bash run_transformers.sh /path/to/your/checkpoint results/finetuned
```

### 方法 2: 直接使用 Python 脚本

```bash
cd ifeval

# 运行完整流程(推理 + 评估)
python run_ifeval_transformers.py \
    --mode all \
    --model_path ../SmolLM2-135M \
    --output_dir results/base \
    --max_new_tokens 1024

# 仅运行推理
python run_ifeval_transformers.py \
    --mode inference \
    --model_path ../SmolLM2-135M \
    --output_dir results/base

# 仅运行评估(需要已有的响应文件)
python run_ifeval_transformers.py \
    --mode eval \
    --input_response_data results/base/responses_SmolLM2-135M.jsonl \
    --output_dir results/base
```

## 参数说明

### 必需参数

- `--mode`: 运行模式
  - `all`: 运行推理和评估(默认)
  - `inference`: 仅运行推理
  - `eval`: 仅运行评估
  
- `--model_path`: 模型路径(推理模式必需)
  - 可以是本地路径或 HuggingFace Hub 模型名称
  
- `--input_response_data`: 响应文件路径(仅评估模式必需)

### 可选参数

- `--input_data`: 输入数据路径(默认: `./data/sampled_input_data.jsonl`)
- `--output_dir`: 输出目录(默认: `./results`)
- `--max_new_tokens`: 最大生成 token 数(默认: 1024)
- `--temperature`: 采样温度,0.0 表示贪婪解码(默认: 0.0)
- `--top_p`: Nucleus 采样参数(默认: 1.0)
- `--system_message`: 可选的系统消息
- `--device`: 使用的设备(`cuda`, `cpu`, 或 `None` 自动检测)
- `--dtype`: 模型数据类型(`auto`, `float16`, `bfloat16`, `float32`)

## 输出文件

运行完成后,会在输出目录中生成以下文件:

```
results/
├── responses_<model_name>.jsonl      # 模型生成的响应
├── eval_results_strict.jsonl         # 严格模式评估结果
├── eval_results_loose.jsonl          # 宽松模式评估结果
└── summary.json                       # 评估结果摘要
```

### summary.json 格式

```json
{
  "response_file": "results/base/responses_SmolLM2-135M.jsonl",
  "strict_accuracy": 0.22,
  "loose_accuracy": 0.35,
  "strict_accuracy_percentage": 22.0,
  "loose_accuracy_percentage": 35.0
}
```

## 与 vLLM 版本的区别

### 优点

1. **兼容性更好**: 可以在 Windows、macOS 和 Linux 上运行
2. **无需额外依赖**: 只需要 PyTorch 和 Transformers
3. **更容易调试**: 代码更简单,更容易理解和修改
4. **支持 CPU**: 可以在没有 GPU 的环境中运行(虽然会很慢)

### 缺点

1. **速度较慢**: vLLM 针对推理做了大量优化,速度更快
2. **内存效率较低**: vLLM 使用了更高效的内存管理
3. **不支持批处理**: 当前实现是逐个处理样本(可以改进)

### 性能对比

在 NVIDIA 2080 Ti 上的估计时间:

- **vLLM**: 约 2-3 分钟(200 个样本)
- **Transformers**: 约 10-15 分钟(200 个样本)

## 常见问题

### Q: 如何在 CPU 上运行?

```bash
python run_ifeval_transformers.py \
    --mode all \
    --model_path ../SmolLM2-135M \
    --device cpu \
    --dtype float32
```

注意: CPU 推理会非常慢,建议只在调试时使用。

### Q: 出现内存不足错误怎么办?

1. 使用更小的 `max_new_tokens`:
   ```bash
   --max_new_tokens 512
   ```

2. 使用半精度:
   ```bash
   --dtype float16
   ```

3. 如果使用 GPU,尝试清理 CUDA 缓存:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Q: 生成的响应质量不好怎么办?

1. 检查模型是否正确加载
2. 尝试调整温度和 top_p 参数
3. 确保 chat template 正确应用
4. 检查是否需要添加 system message

### Q: 出现 "Failed to apply chat template" 警告怎么办?

这个警告通常出现在模型的 tokenizer 没有预设的 chat template 时。解决方法:

1. **对于 SmolLM2 模型**: 代码会自动设置 Qwen3 chat template,不需要手动处理
2. **对于其他模型**: 警告不影响运行,会使用原始 prompt(但可能影响效果)
3. **如果需要手动设置**: 可以在代码中添加 chat template:
   ```python
   from utils import QWEN3CHATTEMPLATE
   tokenizer.chat_template = QWEN3CHATTEMPLATE
   ```

**注意**: 最新版本的代码已经自动处理了 SmolLM2 的 chat template 设置,这个警告应该不会出现。

### Q: 如何使用不同的数据集?

```bash
python run_ifeval_transformers.py \
    --mode all \
    --model_path ../SmolLM2-135M \
    --input_data /path/to/your/data.jsonl \
    --output_dir results/custom
```

数据格式应该与 IFEval 标准格式一致(每行一个 JSON 对象,包含 `prompt` 字段)。

## 代码实现细节

### 推理流程

1. **加载模型和 Tokenizer**
   ```python
   model = AutoModelForCausalLM.from_pretrained(model_path, ...)
   tokenizer = AutoTokenizer.from_pretrained(model_path, ...)
   ```

2. **应用 Chat Template**
   ```python
   formatted = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=True
   )
   ```

3. **生成响应**
   ```python
   outputs = model.generate(
       input_ids=input_ids,
       max_new_tokens=max_new_tokens,
       temperature=temperature,
       ...
   )
   ```

4. **解码并保存**
   ```python
   response = tokenizer.decode(generated_ids, skip_special_tokens=True)
   ```

### 评估流程

评估部分与原始实现完全相同,使用 IFEval 的标准评估库:

1. 加载提示和响应
2. 清理响应(移除推理标记)
3. 运行严格和宽松模式评估
4. 计算准确率并保存结果

## 进阶用法

### 自定义系统消息

```bash
python run_ifeval_transformers.py \
    --mode all \
    --model_path ../SmolLM2-135M \
    --system_message "You are a helpful assistant that follows instructions carefully."
```

### 使用采样而非贪婪解码

```bash
python run_ifeval_transformers.py \
    --mode all \
    --model_path ../SmolLM2-135M \
    --temperature 0.7 \
    --top_p 0.9
```

### 分步运行(推理和评估分开)

```bash
# 步骤 1: 运行推理
python run_ifeval_transformers.py \
    --mode inference \
    --model_path ../SmolLM2-135M \
    --output_dir results/base

# 步骤 2: 运行评估
python run_ifeval_transformers.py \
    --mode eval \
    --input_response_data results/base/responses_SmolLM2-135M.jsonl \
    --output_dir results/base
```

这种方式的好处是可以在不同的机器上运行推理和评估,或者对同一组响应运行多次评估。

## 技术支持

如果遇到问题:

1. 检查 Python 版本(推荐 Python 3.8+)
2. 检查依赖版本:
   ```bash
   pip list | grep -E "torch|transformers|tqdm"
   ```
3. 查看日志输出,通常会包含详细的错误信息
4. 在 Canvas 上提问

## 与作业相关

在作业 Part 5 中,你需要:

1. 评估基础模型:
   ```bash
   bash run_transformers.sh ../SmolLM2-135M results/base
   ```

2. 评估微调后的模型:
   ```bash
   bash run_transformers.sh /path/to/checkpoint results/finetuned
   ```

3. 比较结果:
   - 查看 `results/base/summary.json`
   - 查看 `results/finetuned/summary.json`
   - 在报告中包含严格和宽松准确率
   - 目标: 严格准确率 > 22%

## 许可证

本代码基于 Google Research 的 IFEval 实现,遵循 Apache 2.0 许可证。
