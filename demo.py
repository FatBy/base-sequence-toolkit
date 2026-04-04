"""
base-sequence-toolkit 快速演示
直接运行: python demo.py
不需要任何外部数据，所有示例都是内置的。
"""

from base_sequence_toolkit.core.classifier import (
    BaseType,
    StepClassification,
    classify_step,
    create_context,
)
from base_sequence_toolkit.core.analyzer import (
    compute_sequence_stats,
    compute_transition_matrix,
    extract_risk_profile,
    find_discriminative_patterns,
    run_full_analysis,
)


def demo_1_classifier():
    """
    演示 1: 分类器怎么用

    想象一个 AI agent 在修 bug，它依次执行了这些操作：
      1. 读 src/main.py        → 第一次看这个文件，属于「探索」
      2. 读 src/utils.py       → 第一次看，还是「探索」
      3. 写 src/main.py        → 改代码了，属于「执行」
      4. 读 src/main.py        → 刚写完又读回来看，属于「验证」
      5. 跑 pytest tests/      → 跑测试确认改得对不对，属于「验证」
      6. 搜索 webSearch         → 测试没过，上网搜解决方案，属于「探索」
      7. 写 src/main.py        → 第二次修改，属于「执行」
      8. 跑 pytest tests/      → 再跑一次测试，属于「验证」

    分类器会自动根据上下文判断每一步是 X/E/P/V 哪一种。
    """
    print("=" * 60)
    print("演示 1: 分类器 — 把 agent 的每步操作分类为 X/E/P/V")
    print("=" * 60)

    # 创建上下文（每个任务一个，分类器会记住之前的操作）
    ctx = create_context()

    # 模拟 agent 的 8 步操作
    steps = [
        StepClassification(tool_name="readFile",  args={"path": "src/main.py"},                      status="success", order=0),
        StepClassification(tool_name="readFile",  args={"path": "src/utils.py"},                     status="success", order=1),
        StepClassification(tool_name="writeFile", args={"path": "src/main.py", "content": "fix..."},  status="success", order=2),
        StepClassification(tool_name="readFile",  args={"path": "src/main.py"},                      status="success", order=3),
        StepClassification(tool_name="runCmd",    args={}, shell_command="pytest tests/",             status="success", order=4),
        StepClassification(tool_name="webSearch", args={"query": "python asyncio timeout"},           status="success", order=5),
        StepClassification(tool_name="writeFile", args={"path": "src/main.py", "content": "fix2.."}, status="success", order=6),
        StepClassification(tool_name="runCmd",    args={}, shell_command="pytest tests/",             status="success", order=7),
    ]

    descriptions = [
        "读 main.py     — 第一次看，不知道里面什么",
        "读 utils.py    — 第一次看，也是探索",
        "写 main.py     — 开始改代码了",
        "读 main.py     — 刚写完就读回来检查",
        "跑 pytest      — 写完代码跑测试验证",
        "搜索 asyncio   — 测试没过，上网查资料",
        "写 main.py     — 根据搜索结果再改一次",
        "跑 pytest      — 改完再验证一次",
    ]

    bases = []
    for step, desc in zip(steps, descriptions):
        base = classify_step(step, ctx)
        bases.append(str(base))
        print(f"  步骤 {step.order}: {desc}")
        print(f"         → 分类结果: {base} ({base.name})")
        print()

    sequence = "".join(bases)
    print(f"  最终碱基序列: {sequence}")
    print(f"  含义: {'→'.join(bases)}")
    print()

    # 解释为什么这样分类
    print("  为什么这样分？")
    print("  · readFile 第一次访问某文件 → X（探索未知）")
    print("  · writeFile                → E（执行修改）")
    print("  · readFile 读刚写过的文件   → V（写后验证）")
    print("  · pytest 在有写操作之后     → V（测试验证）")
    print("  · webSearch                → X（总是探索）")
    print()
    return sequence


def demo_2_analyzer():
    """
    演示 2: 分析器怎么用

    假设你有一批 agent 执行记录，有的成功有的失败。
    分析器能告诉你：成功和失败的序列有什么区别？
    """
    print("=" * 60)
    print("演示 2: 分析器 — 从碱基序列中找规律")
    print("=" * 60)

    # 模拟数据：10 个成功 + 10 个失败的碱基序列
    success_sequences = [
        "XEVEV",      # 探索→执行→验证→执行→验证（标准成功模式）
        "XEEVE",      # 探索→执行→执行→验证→执行
        "XXEVEV",     # 多探索后执行→验证循环
        "XEEV",       # 简洁路径
        "XEVEE",      # 验证后继续执行
        "XEXEVEV",    # 中途探索后回到执行验证
        "XEEVEV",     # 双执行后验证
        "XEVEEV",     # 执行验证执行执行验证
        "XXEEV",      # 探索后执行验证
        "XEVEVE",     # 经典循环
    ]

    failure_sequences = [
        "XXXEEE",     # 探索太多，没有验证
        "XXXXXX",     # 纯探索螺旋，什么都没做
        "XEEEEEE",   # 猛冲不验证
        "XXXEXE",     # 不断探索
        "XEXXXE",     # 中途又陷入探索
        "XXXEEX",     # 探索螺旋
        "XXXXEEE",   # 前期全探索
        "XEEEEE",    # 无验证的执行
        "XXEXXEX",   # 碎片化
        "XXXXXE",    # 探索过度
    ]

    # ── 2a: 单序列统计 ──
    print("\n--- 2a: 单条序列的统计 ---")
    seq = "XEVEV"
    stats = compute_sequence_stats(seq)
    print(f"  序列: {seq}")
    print(f"  长度: {stats.length} 步")
    print(f"  首碱基: {stats.first_base}  末碱基: {stats.last_base}")
    print(f"  E→V 配对数: {stats.ev_pairs}  （执行后紧跟验证的次数）")
    print(f"  切换频率: {stats.switch_rate}  （相邻步骤不同的比例）")
    print(f"  有验证(V): {stats.has_verification}")
    print(f"  碱基分布: X={stats.distribution.X} E={stats.distribution.E} "
          f"P={stats.distribution.P} V={stats.distribution.V}")

    # ── 2b: 风险画像 ──
    print("\n--- 2b: 风险画像 ---")
    for seq in ["XXXEEE", "XEVEV", "XEEEEEEEEEEE"]:
        risk = extract_risk_profile(seq)
        flags = risk.flags()
        status = "高风险" if flags else "正常"
        print(f"  {seq:<20s} → {status}: {', '.join(flags) if flags else '无风险标志'}")

    # ── 2c: 转移矩阵 ──
    print("\n--- 2c: 转移矩阵（从某碱基到下一个碱基的概率）---")
    all_seqs = success_sequences + failure_sequences
    matrix = compute_transition_matrix(all_seqs)
    print(matrix.format())
    print("  关键：E→V 的概率越高越好（执行后验证），E→E 占主导说明验证不足")

    # ── 2d: 判别性模式 ──
    print("\n--- 2d: 成功 vs 失败的差异化模式 ---")
    patterns = find_discriminative_patterns(success_sequences, failure_sequences, min_count=1)

    print("\n  与成功强相关的模式（lift > 1 = 在成功中更常见）:")
    for p in patterns[:5]:
        print(f"    {p.pattern:6s}  lift={p.lift:.2f}  (成功出现{p.resolved_count}次, 失败{p.unresolved_count}次)")

    print("\n  与失败强相关的模式（lift < 1 = 在失败中更常见）:")
    for p in sorted(patterns, key=lambda x: x.lift)[:5]:
        print(f"    {p.pattern:6s}  lift={p.lift:.2f}  (成功出现{p.resolved_count}次, 失败{p.unresolved_count}次)")

    # ── 2e: 完整分析报告 ──
    print("\n--- 2e: 一键完整报告 ---")
    all_sequences = success_sequences + failure_sequences
    outcomes = [True] * len(success_sequences) + [False] * len(failure_sequences)

    report = run_full_analysis(all_sequences, outcomes, min_pattern_count=1)
    print(report.format_summary())


def demo_3_swe_agent():
    """
    演示 3: SWE-agent 适配器

    不需要下载数据集，这里模拟一个 SWE-agent 轨迹格式的数据。
    """
    print("\n" + "=" * 60)
    print("演示 3: SWE-agent 适配器 — 把 SWE-agent 轨迹转成碱基序列")
    print("=" * 60)

    from base_sequence_toolkit.adapters.swe_agent import classify_trajectory

    # 模拟一条 SWE-agent 轨迹（真实格式）
    fake_trajectory = {
        "instance_id": "django__django-11099",
        "model_name": "gpt-4",
        "target": True,  # 解决了
        "trajectory": [
            {"role": "ai", "text": "Let me find the relevant file.\n```\nfind_file forms.py\n```"},
            {"role": "environment", "text": "Found 5 files matching 'forms.py'"},
            {"role": "ai", "text": "Let me open the main forms file.\n```\nopen django/forms/forms.py\n```"},
            {"role": "environment", "text": "[File content...]"},
            {"role": "ai", "text": "I see the bug. Let me fix it.\n```\nedit 150:155\n    def clean(self):\n        return self.cleaned_data\nend_of_edit\n```"},
            {"role": "environment", "text": "Edit applied successfully."},
            {"role": "ai", "text": "Now let me run the tests to verify.\n```\npython -m pytest tests/forms_tests/\n```"},
            {"role": "environment", "text": "All tests passed."},
            {"role": "ai", "text": "All tests pass. Let me submit.\n```\nsubmit\n```"},
            {"role": "environment", "text": "Submitted."},
        ],
    }

    result = classify_trajectory(fake_trajectory)

    print(f"\n  实例: {result.instance_id}")
    print(f"  模型: {result.model_name}")
    print(f"  结果: {'已解决' if result.resolved else '未解决'}")
    print(f"  步数: {result.step_count}")
    print(f"  碱基序列: {result.base_sequence}")
    print()

    # 逐步解释
    step_texts = [
        "find_file forms.py    → X（探索：搜索文件）",
        "open django/forms/... → X（探索：打开文件看内容）",
        "edit 150:155          → E（执行：修改代码）",
        "pytest tests/         → V（验证：跑测试）",
        "submit                → V（验证：提交解决方案）",
    ]
    print("  逐步分类:")
    for i, (base_char, text) in enumerate(zip(result.base_sequence, step_texts)):
        print(f"    {i+1}. [{base_char}] {text}")

    print(f"\n  碱基占比: X={result.x_ratio:.0%} E={result.e_ratio:.0%} "
          f"P={result.p_ratio:.0%} V={result.v_ratio:.0%}")

    # 如果要分析真实的 SWE-agent 数据集:
    print("\n  --- 如果要分析真实 SWE-agent 数据集 ---")
    print("  pip install base-sequence-toolkit[swe-agent]")
    print("  然后:")
    print("    from base_sequence_toolkit.adapters.swe_agent import load_from_huggingface")
    print("    results = load_from_huggingface(max_records=500)")
    print("  它会自动从 HuggingFace 下载 nebius/SWE-agent-trajectories 数据集")


if __name__ == "__main__":
    demo_1_classifier()
    print()
    demo_2_analyzer()
    demo_3_swe_agent()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("""
总结：
  1. 分类器: 把 agent 的每步操作 → X/E/P/V
     classify_step(step, ctx) → BaseType

  2. 分析器: 从一批碱基序列中找规律
     run_full_analysis(sequences, outcomes) → 完整报告

  3. 适配器: 把特定 agent 的数据格式 → 碱基序列
     classify_trajectory(record) → 结果

工作流:  原始轨迹 →(适配器)→ 碱基序列 →(分析器)→ 报告/洞察
""")
