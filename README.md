# LLM-Smart-Water

**给你的 AI Agent 装一个"智能水表" -- 用 XEPV 四字母表解读 Agent 行为基因**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Your%20Agent%20Has%20a%20Genome-orange.svg)](#citation)

**[English](README_EN.md) | [中文](README.md)**

---

## 一句话介绍

> 你的 AI Agent 每天在"探索-执行-规划-验证"之间来回跳转，但你有没有想过：**它到底在干什么？干得好不好？有没有在做无用功？**
>
> LLM-Smart-Water 就是 Agent 世界的"智能水表" -- 它把 Agent 的每一步操作编码成 4 个字母（X、E、P、V），像 DNA 碱基一样排列成序列，然后自动帮你发现 Agent 的行为习惯、效率瓶颈和失败风险。

---

## 这东西解决什么问题？

现在 AI Agent 满天飞 -- ReAct、Reflexion、SWE-agent、各种自主编程助手 -- 它们能读文件、写代码、跑测试、搜网页。但有个尴尬的现实：

**我们只看结果，不看过程。**

两个 Agent 都完成了 90% 的任务，看起来一样好？不一定。一个可能高效利落地"看一眼、写一笔、跑个测试"就搞定了；另一个可能在"搜了半天、想了又想、改了又改"之后才磕磕绊绊地完成。结果一样，但第二个在遇到新问题时大概率会翻车。

**LLM-Smart-Water 就是让你看清这个"过程"的工具。**

### 核心思路：把 Agent 行为变成"基因序列"

灵感来自生物学：DNA 用 4 个碱基字母（A、T、C、G）编码了整个生命的蓝图。我们用类似的思路，用 4 个字母来编码 AI Agent 的每一步行为：

| 字母 | 含义 | 通俗解释 | 典型操作 |
|------|------|----------|----------|
| **X** | 探索 (eXplore) | Agent 在"看东西"、"找资料" | 读文件、搜网页、浏览目录、grep 搜索 |
| **E** | 执行 (Execute) | Agent 在"干活"、"改东西" | 写文件、改代码、跑命令、调 API |
| **P** | 规划 (Plan) | Agent 在"想事情"、"定计划" | 纯推理、任务分解、Reflexion 反思 |
| **V** | 验证 (Verify) | Agent 在"检查成果"、"跑测试" | 运行 pytest、检查输出、写完之后再读一遍 |

这样，一次任务执行就变成了一条**行为序列** -- 比如 `XEEVXEV`（看了看 -> 写了代码 -> 又写了 -> 验证了 -> 再看看 -> 再写 -> 再验证）。

**这就是你的 Agent 的"行为基因"，而 LLM-Smart-Water 就是解读这条基因的工具。**

---

## 我们发现了什么？（论文核心发现）

我们分析了 **2,347+ 条真实 Agent 执行轨迹**（347 条来自生产环境 DunCrew 系统，2,000 条来自开源 SWE-agent），得到了一些非常有意思的发现：

### 发现 1: Agent 几乎不检查自己的工作

> **E -> V 的转移概率只有 2.1%**

Agent 在执行完操作后，只有 2% 的概率会去验证结果。就像一个程序员写完代码从来不跑测试 -- 这在我们的数据中是系统性的问题。而在 SWE-agent 数据中，成功解决问题的轨迹，E->V 转移率是未解决轨迹的**将近 2 倍**（54.2% vs. 28.1%）。

**一句话总结：写完代码就跑测试的 Agent，成功率远高于写完就走的。**

### 发现 2: "想太多"是最强的失败信号

> **P 比率（规划占比）与成功率的相关系数 r = -0.256，p < 0.0001**

不是说规划不好，而是**相对于执行来说，规划太多是最清晰的失败标志**。失败任务中，规划占比是成功任务的 **2 倍**（23.4% vs. 11.9%），而执行占比只有成功任务的 **一半**（23.4% vs. 38.4%）。

**一句话总结：Agent 陷入"规划-探索-再规划"的死循环时，基本就凉了。**

### 发现 3: P-X-P 是唯一的"高危模式"

> **P-X-P 三连出现时，成功率下降 10.4%（83.3% vs. 全局 92.5%）**

P-X-P 意味着 Agent "想了想 -> 去看了看 -> 又开始想"，这说明第一次规划没能消化探索到的信息，不得不重新规划。在我们测试的所有三元组模式中，这是**唯一**具有统计显著性的高风险模式。

有趣的是，连续探索 X-X-X 的成功率（94.4%）反而**高于**平均水平 -- 问题不在于探索太多，而在于**探索完不去执行，反而回头重新规划**。

### 发现 4: 不同模型有不同的"行为指纹"

在 SWE-bench 上的跨模型分析揭示了惊人的差异：

| 模型 | 探索占比(X) | 执行占比(E) | 验证占比(V) | 解决率 |
|------|------------|------------|------------|--------|
| Llama-405B | 34.0% | 39.9% | **26.1%** | **42.5%** |
| Llama-70B | 43.0% | 40.0% | 17.0% | 16.2% |
| Llama-8B | 44.8% | 38.2% | 17.0% | 18.0% |

**更大的模型天然更爱"验证"（V 比率高出 50%+），解决率也大幅领先。** 行为序列可以作为模型的"行为身份证"。

---

## Governor: 零成本的 Agent "教练"

光发现问题不够，还要能解决问题。所以我们造了 **Governor** -- 一个实时监控 Agent 行为并在关键时刻"提醒"它的系统。

### 它是怎么工作的？

```
Agent 每执行一步 -> Governor 看一下行为序列 -> 发现问题模式 -> 注入一段提示词 -> Agent 调整行为
```

**零 LLM 开销** -- Governor 是纯规则逻辑，不需要额外调用大模型，延迟 < 1ms。

### 三层架构

| 层级 | 干什么 | 怎么理解 |
|------|--------|----------|
| **Layer 1: 规则引擎** | 7 条规则 + 8 维特征向量，检测到问题就注入提示 | 相当于"即时教练"，发现问题马上指出来 |
| **Layer 2: 统计累加器** | 记录每次干预的效果，按特征分桶统计 | 相当于"数据记录员"，持续收集反馈 |
| **Layer 3: 自适应** | 定期用卡方检验评估规则有效性，自动调整阈值 | 相当于"策略优化师"，根据数据调整教练的方法 |

### 7 条内置规则（全部从数据中发现，不是拍脑袋想的）

| 规则 | 做什么 | 数据依据 |
|------|--------|----------|
| `x_brake` | 连续探索太久就喊停 | X->X 自环概率 57.2%，不打断会无限循环 |
| `miss_verify` | 执行了好几步还没验证就提醒 | E->V 只有 2.1%，Agent 天生不爱验证 |
| `switch_warn` | 策略来回切换就提醒专注 | switch_rate 与失败负相关 (r=-0.134) |
| `explore_dom` | 探索太多执行太少就推一把 | X/(X+E) 失衡表明 Agent 在"空转" |
| `late_plan` | 任务过半还在重新规划就催促执行 | 后期规划与失败负相关 |
| `div_collapse` | 行为完全单一化就打破循环 | 多样性崩溃是死循环的标志 |
| `step_fuse` | ~~任务步数太长就终止~~ (已禁用) | 数据发现 >15 步的任务成功率 97.4%，长不是问题！ |

> `step_fuse` 被禁用的故事很有意思：我们最初以为长任务=失败，结果数据告诉我们恰恰相反。Governor 的第三层自适应检测到这条规则在损害性能，自动标记出来，最终被人工审核后停用。这就是数据驱动规则管理的价值。

### 实际效果

在生产环境中（101 条 Governor 前 vs. 246 条 Governor 后轨迹）：

| 指标 | Governor 前 | Governor 后 | 变化 |
|------|------------|------------|------|
| **任务成功率** | 88.1% | **94.3%** | **+6.2%** |
| **平均 Token 消耗** | 275K | **154K** | **-44%** |
| **平均执行步数** | 22.2 | **14.0** | **-37%** |

**同时提升成功率和降低成本** -- 这不是此消彼长，而是因为 Governor 阻止了浪费的探索螺旋（这才是 token 成本的大头）。

其中 `x_brake`（探索制动）一条规则就贡献了大部分收益：它在 60% 的任务中触发，触发后成功率高达 98.6%。

---

## 安装

```bash
pip install base-sequence-toolkit

# 如果要分析 SWE-agent 轨迹（需要 HuggingFace datasets）：
pip install base-sequence-toolkit[swe-agent]

# 开发模式：
pip install base-sequence-toolkit[dev]
```

从源码安装：

```bash
git clone https://github.com/FatBy/base-sequence-toolkit.git
cd base-sequence-toolkit
pip install -e ".[dev,swe-agent]"
```

## 快速上手

### 1. 分析 SWE-agent 轨迹

```python
from base_sequence_toolkit.adapters.swe_agent import load_from_huggingface
from base_sequence_toolkit.core.analyzer import run_full_analysis

# 从 HuggingFace 加载 500 条 SWE-agent 轨迹并自动分类
results = load_from_huggingface(max_records=500)

# 提取序列和结果
sequences = [r.base_sequence for r in results]
outcomes = [r.resolved for r in results]

# 一键跑完整分析：N-gram、转移矩阵、风险画像等
report = run_full_analysis(sequences, outcomes)
print(report.format_summary())
```

### 2. 给你自己的 Agent 接入分类器

```python
from base_sequence_toolkit import classify_step, create_context, StepClassification

ctx = create_context()

steps = [
    StepClassification(tool_name="readFile", args={"path": "src/main.py"}),
    StepClassification(tool_name="writeFile", args={"path": "src/main.py", "content": "..."}, status="success"),
    StepClassification(tool_name="readFile", args={"path": "src/main.py"}),  # 写后再读 -> 自动识别为 V（验证）
    StepClassification(tool_name="runCmd", shell_command="pytest tests/"),
]

for step in steps:
    base = classify_step(step, ctx)
    print(f"{step.tool_name:15s} -> {base}")

# 输出:
# readFile        -> X  (第一次访问，是探索)
# writeFile       -> E  (写文件，是执行)
# readFile        -> V  (刚写完就回头读，是验证!)
# runCmd          -> V  (写完代码跑测试，也是验证!)
```

### 3. 分析自定义序列

```python
from base_sequence_toolkit.core.analyzer import (
    compute_sequence_stats,
    compute_transition_matrix,
    extract_risk_profile,
    find_discriminative_patterns,
)

# 单条序列统计
stats = compute_sequence_stats("XEEVXEV")
print(f"长度: {stats.length}")
print(f"EV 配对数: {stats.ev_pairs}")
print(f"有验证: {stats.has_verification}")
print(f"切换率: {stats.switch_rate}")

# 多条序列的转移矩阵
matrix = compute_transition_matrix(["XEEVXEV", "XXXEEE", "XEVEV"])
print(matrix.format())

# 风险评估
risk = extract_risk_profile("XXXEEEEE")
print(f"风险标记: {risk.flags()}")
# -> ['consecutive_x>=3', 'no_verification']

# 找到区分成功/失败的模式
patterns = find_discriminative_patterns(
    resolved_sequences=["XEVEV", "XEEVE"],
    unresolved_sequences=["XXXEE", "XXXXX"],
    min_count=1,
)
for p in patterns[:5]:
    print(f"  {p.pattern}: lift={p.lift:.2f}")
```

### 4. 使用 Governor 实时监控

```python
from base_sequence_toolkit import BaseSequenceGovernor, InterventionRecord

governor = BaseSequenceGovernor()

# 实时评估（每次工具调用后）
bases = ["X", "E", "E", "X", "X", "X", "X"]
signal = governor.evaluate(bases)

if signal.triggered:
    print(f"触发规则: {signal.triggered_rules}")
    print(f"注入提示: {signal.prompt_injection}")
    # -> 把 signal.prompt_injection 注入到 LLM 的系统提示中

# 查看 8 维特征
print(f"连续探索次数: {signal.features.consecutive_x}")
print(f"探索执行比: {signal.features.xe_ratio}")
```

```python
# 任务结束后记录结果，让 Governor 学习
adjustments = governor.record_trace(
    base_sequence="X-E-E-X-X-X-X",
    success=False,
    interventions=[
        InterventionRecord(
            rule="consecutive_x_brake",
            step_index=5,
            features=signal.features,
        )
    ],
)

# Layer 3 积累足够数据后自动调整阈值
if adjustments:
    print(f"阈值调整: {adjustments}")
```

## CLI 命令行一键分析

```bash
# 分析 SWE-agent 轨迹
bst-analyze swe-agent -n 500 -o results/

# 分析 DunCrew 执行轨迹
bst-analyze duncrew /path/to/exec_traces/ -o results/

# 分析自定义 JSON 数据
bst-analyze json data/my_sequences.json
```

## 接入你自己的 Agent 框架

实现一个适配器，把你的 Agent 轨迹转换成 XEPV 序列：

```python
from base_sequence_toolkit import classify_step, create_context, StepClassification

def classify_my_agent_trace(trace: dict) -> str:
    """把你的 Agent 轨迹转换为 XEPV 行为序列"""
    ctx = create_context()
    bases = []

    for action in trace["actions"]:
        step = StepClassification(
            tool_name=action["tool"],
            args=action.get("arguments", {}),
            status="success" if action["succeeded"] else "error",
            shell_command=action.get("command"),
        )
        base = classify_step(step, ctx)
        bases.append(str(base))

    return "".join(bases)
```

适配器只需要满足两个条件：
1. **完备性**：你的 Agent 的每个动作都能映射到 X/E/P/V 中的一种
2. **语义一致性**：收集信息 -> X，修改状态 -> E，验证结果 -> V，纯推理 -> P

## 8 维特征向量

每条行为序列被提取为 8 个数值特征，用于实时风险评估：

```
consecutive_x        -- 尾部连续 X 计数（探索惯性）
step_count           -- 总步数（任务复杂度）
x_ratio_last5        -- 最近 5 步中 X 的比例（局部探索密度）
switch_rate          -- 相邻步骤类型不同的比率（行为稳定性）
p_in_late_half       -- P 是否出现在后半段（晚期规划信号）
last_p_followed_by_v -- 最近的 P 后面是否跟了 V（验证覆盖）
max_e_run_length     -- 最长连续 E 长度（执行动量）
xe_ratio             -- X / (X + E)（探索执行比）
```

---

## 论文背景：更深入的理解

LLM-Smart-Water 的理论基础来自论文 *"Your Agent Has a Genome: Sequence-Level Behavioral Analysis and Runtime Governance of LLM Autonomous Agents"*。以下是论文提出的更深层次的洞察：

### "小脑假说"

论文提出了一个有趣的架构类比：
- **LLM** 是 Agent 的"大脑" -- 负责推理、创造力、理解语义
- **工具框架**（ReAct 循环、技能系统）是"四肢" -- 负责对世界施加行动
- **行为序列治理（Governor）** 是"小脑" -- 负责运动协调和时间序列编排

人类小脑有 690 亿个神经元（比大脑皮层还多），因为运动协调需要从大量运动经验中学习。类似地，一个成熟的行为序列治理系统需要从海量执行轨迹中学习，才能发现那些决定成功的复杂行为模式。

### 六个未来研究方向

| 方向 | 内容 | 所需数据规模 |
|------|------|------------|
| 行为序列语言模型 | 把 XEPV 序列当作一种"语言"来建模，预测下一步行为 | ~3K-5K 轨迹 |
| 行为条件解码 | 在 LLM 推理时用行为序列信号引导工具选择 | ~50K 轨迹 |
| 序列异常检测 | 自动发现未知的异常行为模式 | ~3K-5K 轨迹 |
| 双流 Agent 架构 | 专门的"序列协调器"与 LLM 并行工作 | ~100 万+ 轨迹 |
| 行为序列奖励模型 | 为 Agent RL 训练提供密集的步级奖励信号 | ~50K 轨迹 |
| 行为指纹识别 | 从匿名轨迹中识别底层模型和框架 | 跨系统数据 |

### 跨系统验证

我们在 SWE-bench 上验证了 2,000 条 SWE-agent 公开轨迹，确认了两个核心发现在完全不同的 Agent 框架中复现：
- **E->V 验证缺失**：已解决问题的 E->V 转移率是未解决问题的 2 倍
- **探索螺旋**：未解决问题的平均最大连续探索长度是已解决问题的 2.3 倍（11.0 vs. 4.8）

这意味着这些不是某个特定系统的 bug，而是 **LLM Agent 的通用行为特征** -- 自回归生成天然倾向于继续当前动作类型，而不会主动切换到验证。

---

## 项目结构

```
base-sequence-toolkit/
├── base_sequence_toolkit/
│   ├── __init__.py                 # 包入口，导出核心 API
│   ├── cli.py                      # CLI 命令行工具
│   ├── core/
│   │   ├── classifier.py           # XEPV 分类器（有状态、上下文感知）
│   │   ├── analyzer.py             # 分析原语：N-gram、转移矩阵、风险画像
│   │   └── governor.py             # 三层自适应 Governor
│   └── adapters/
│       ├── swe_agent.py            # SWE-agent 轨迹适配器
│       └── duncrew.py              # DunCrew 轨迹适配器
├── examples/
│   └── swe_agent_analysis.py       # 端到端示例
├── tests/                          # 测试套件
├── data/                           # 预计算结果（可选）
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

如果这个工具对你的研究有帮助，请引用：

```bibtex
@software{llm_smart_water,
  title = {LLM-Smart-Water: XEPV Behavioral Sequence Analysis Framework for AI Agent Execution Traces},
  author = {Deng, Sidi},
  year = {2026},
  url = {https://github.com/FatBy/base-sequence-toolkit}
}
```

## License

MIT License. 详见 [LICENSE](LICENSE)。

## Related

- **XEPV Framework** -- 行为序列分类体系，详见论文
- **DunCrew** -- 集成了 XEPV 治理的 AI Agent 操作系统 ([duncrew.com](https://duncrew.com))
- **SWE-bench** -- AI 软件工程基准测试
