"""Rebuild bilingual cross-experiment analysis from actual completed artifacts."""
import csv
import json
from pathlib import Path

from openrubrics_lab.config import load_config
from openrubrics_lab.io import digest, file_hash, read_jsonl, source_fingerprint, write_json
from openrubrics_lab.metrics import majority, parse_winner

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT/'experiments/m1-cpu-training-20261116'
SUPP = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return 'n/a' if value is None else f'{100*value:.1f}%'


def vote_outcomes(path):
    counts = {'correct_votes': 0, 'wrong_votes': 0, 'invalid_votes': 0,
              'invalid_at_limit': 0, 'invalid_before_limit': 0, 'invalid_without_generation': 0}
    for row in read_jsonl(path):
        stats = row.get('generation_stats', [])
        for index, prediction in enumerate(row['predictions']):
            winner = parse_winner(prediction)
            if winner is not None:
                counts['correct_votes' if winner == row['label'] else 'wrong_votes'] += 1
            else:
                counts['invalid_votes'] += 1
                key = ('invalid_without_generation' if index >= len(stats) else
                       'invalid_at_limit' if stats[index]['hit_token_limit'] else 'invalid_before_limit')
                counts[key] += 1
    return counts


def main():
    manifest = read(PRIMARY/'manifest.json')
    assert manifest['status'] == 'completed_small_scale_protocol'
    assert manifest['source_files'] == source_fingerprint(ROOT), 'Scientific source changed after run'
    assert manifest['config'] == load_config(ROOT/'configs/m1.json')
    for name, expected in manifest['inputs'].items():
        assert file_hash(ROOT/'data/m1'/name) == expected, name
    assert manifest['signature'] == digest({'config': manifest['config'], 'inputs': manifest['inputs'],
                                            'code': manifest['source_files']})
    supplementary_inputs = read(SUPP/'input-receipt.json')['files']
    for relative, expected in supplementary_inputs.items():
        assert file_hash(ROOT/relative) == expected, relative
    checked = 0
    for stage in manifest['stages'].values():
        assert stage['status'] == 'completed'
        for relative, expected in stage['artifacts'].items():
            assert file_hash(PRIMARY/relative) == expected, relative
            checked += 1
    device_receipts = [p for p in PRIMARY.rglob('execution_device.json')
                       if '.failed-' not in str(p) and '.superseded-' not in str(p)]
    assert device_receipts
    for path in device_receipts:
        assert read(path)['actual_device'] == 'cpu' and read(path)['trainable'], path
    write_json(ROOT/'experiments/acceptance/final-artifact-verification.json',
               {'stages': len(manifest['stages']), 'artifacts': checked,
                'source_fingerprint_matches': True, 'all_hashes_match': True,
                'config_signature_and_input_hashes_match': True,
                'cpu_training_checkpoint_receipts': len(device_receipts),
                'supplementary_input_hashes_verified': len(supplementary_inputs)})
    metrics = []
    for path in sorted(PRIMARY.rglob('predictions.jsonl')):
        if '.failed-' in str(path) or '.superseded-' in str(path):
            continue
        metrics.append({'experiment': str(path.parent.relative_to(ROOT/'experiments')),
                        **read(path.parent/'summary.json'), **vote_outcomes(path)})
    for name in ['long-decode-sft', 'long-two-stage-sft', 'order-sft', 'order-arm',
                 'eval-helpsteer3', 'eval-ppe-ifeval', 'eval-ifbench']:
        metrics.append({'experiment': f'supplementary/{name}', **read(SUPP/name/'summary.json'),
                        **vote_outcomes(SUPP/name/'predictions.jsonl')})
    public = {key: sum(row[key] for row in metrics if '/11-public-benchmarks/' in row['experiment'])
              for key in ['correct_votes', 'wrong_votes', 'invalid_votes', 'invalid_at_limit',
                          'invalid_before_limit', 'invalid_without_generation']}
    phases = []
    for variant in ['07-arm', '09-ablation-reverse', '09-ablation-no_format']:
        for item in read(PRIMARY/variant/'schedule.json'):
            folder = PRIMARY/variant/f"round-{item['round']}-{item['phase']}"
            trace = read_jsonl(folder/'training.jsonl')
            rollouts = read_jsonl(folder/'rollouts.jsonl')
            correct = [r.get('accuracy_reward', r['reward']) for r in rollouts]
            phases.append({'variant': variant, 'round': item['round'], 'phase': item['phase'],
                'mean_correctness_reward': sum(correct)/len(correct), 'reward_mean': item['reward_mean'],
                'zero_variance_groups': item['zero_variance_groups'],
                'max_gradient_norm': max(t['grad_norm'] for t in trace)})
    order = {}
    for variant, stage in [('sft', '06-sft-evaluation'), ('arm', '08-arm-evaluation')]:
        original = {r['id']: r for r in read_jsonl(PRIMARY/stage/'predictions.jsonl')}
        comparisons = []
        for row in read_jsonl(SUPP/f'order-{variant}'/'predictions.jsonl'):
            forward = original[row['original_id']]
            assert row['instruction'] == forward['instruction']
            assert row['response_a'] == forward['response_b'] and row['response_b'] == forward['response_a']
            assert row['label'] != forward['label']
            a = majority([parse_winner(x) for x in forward['predictions']])
            b = majority([parse_winner(x) for x in row['predictions']])
            same_rubric = row['rubric'] == forward['rubric']
            comparisons.append({'id': row['original_id'], 'forward_vote': a, 'reverse_vote': b,
                                'rubric_identical': same_rubric,
                                'comparable': a is not None and b is not None and same_rubric,
                                'consistent': a is not None and b is not None and a != b})
        valid = [r for r in comparisons if r['comparable']]
        order[variant] = {'pairs': len(comparisons), 'valid_comparisons': len(valid),
            'identical_rubric_pairs': sum(r['rubric_identical'] for r in comparisons),
            'consistency_among_valid': sum(r['consistent'] for r in valid)/len(valid) if valid else None,
            'comparisons': comparisons}
    policy_names = [r['name'] for r in read(SUPP/'policy-variants.json')['policies']]
    policy = {name: read(SUPP/f'ifeval-{name}'/'summary.json') for name in policy_names}
    dpo = read(PRIMARY/'12-dpo-arm/adapter/summary.json')
    source_dpo = read(PRIMARY/'13-dpo-source-baseline/summary.json')
    iterations = read(PRIMARY/'14-iterative-dpo/summary.json')
    online = read(PRIMARY/'15-online-grpo/summary.json')
    online_rollouts = read_jsonl(PRIMARY/'15-online-grpo/rollouts.jsonl')
    online['reward_diagnostics'] = {
        'full_wins': sum(r['reward'] == 1 for r in online_rollouts),
        'half_credit': sum(r['reward'] == .5 for r in online_rollouts),
        'half_credit_with_one_valid_order': sum(r['reward'] == .5 and
            sum(parse_winner(t) is not None for t in r['judge_predictions']) == 1 for r in online_rollouts),
        'max_pre_clip_gradient_norm': max(r['grad_norm'] for r in read_jsonl(PRIMARY/'15-online-grpo/training.jsonl')),
    }
    sft_rubrics = read_jsonl(PRIMARY/'06-sft-evaluation/rubrics.jsonl')
    rubric_limits = {'unique_prompts': len(sft_rubrics),
                     'at_token_limit': sum(r['hit_token_limit'] for r in sft_rubrics),
                     'token_cap': manifest['config']['rubric_tokens']}
    result = {'analysis_source_sha256': file_hash(Path(__file__)),
        'primary_stages': len(manifest['stages']), 'artifact_hashes_verified': checked,
        'current_stage_seconds': sum(x['seconds'] for x in manifest['stages'].values()),
        'metrics': metrics, 'arm_phases': phases, 'controlled_order': order, 'ifeval': policy,
        'arm_dpo': dpo, 'source_dpo': source_dpo, 'iterative_dpo': iterations, 'online_grpo': online,
        'sft_rubric_length_limits': rubric_limits, 'public_benchmark_vote_outcomes': public}
    write_json(SUPP/'combined.json', result)
    with (SUPP/'combined-metrics.csv').open('w', newline='') as f:
        fields = ['experiment', 'n', 'majority_accuracy', 'invalid_rate', 'length_limit_rate',
                  'correct_votes', 'wrong_votes', 'invalid_votes']
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metrics)
    rows = ['| Experiment | Pairs | Voting accuracy | Invalid | Token limit | Judge devices |',
            '|---|---:|---:|---:|---:|---|']
    for row in metrics:
        rows.append(f"| {row['experiment'].replace('m1-cpu-training-20261116/', '')} | {row['n']} | "
                    f"{pct(row['majority_accuracy'])} | {pct(row['invalid_rate'])} | {pct(row['length_limit_rate'])} | "
                    f"{' / '.join(row['actual_devices']) or 'none'} |")
    phase_table = ['| Variant | Round | Actor | Correctness reward | Total reward | Zero-variance groups | Max gradient norm |',
                   '|---|---:|---|---:|---:|---:|---:|']
    for row in phases:
        phase_table.append(f"| {row['variant']} | {row['round']} | {row['phase']} | "
            f"{row['mean_correctness_reward']:.4f} | {row['reward_mean']:.4f} | "
            f"{row['zero_variance_groups']} | {row['max_gradient_norm']:.6f} |")
    policy_table = ['| Policy | Prompts | Strict prompt | Loose prompt | Strict instruction | Loose instruction |',
                    '|---|---:|---:|---:|---:|---:|']
    for name, row in policy.items():
        policy_table.append(f"| {name} | {row['n']} | {pct(row['strict']['prompt_accuracy'])} | "
            f"{pct(row['loose']['prompt_accuracy'])} | {pct(row['strict']['instruction_accuracy'])} | "
            f"{pct(row['loose']['instruction_accuracy'])} |")
    order_table = ['| Actor pair | Tested pairs | Identical rubrics | Valid controlled comparisons | Consistency |',
                   '|---|---:|---:|---:|---:|']
    for name, row in order.items():
        order_table.append(f"| {name} | {row['pairs']} | {row['identical_rubric_pairs']} | "
                           f"{row['valid_comparisons']} | {pct(row['consistency_among_valid'])} |")
    short = read(PRIMARY/'06-sft-evaluation/summary.json')
    long = read(SUPP/'long-decode-sft/summary.json')
    joint = read(SUPP/'long-two-stage-sft/summary.json')
    joint_rubrics = read_jsonl(SUPP/'long-two-stage-sft/rubrics.jsonl')
    joint_rubric_limits = sum(r['hit_token_limit'] for r in joint_rubrics)
    teacher_lengths = read(SUPP/'teacher-target-lengths.json')['summary']['train']
    shared = '\n'.join(rows)
    phase_md = '\n'.join(phase_table)
    policy_md = '\n'.join(policy_table)
    order_md = '\n'.join(order_table)
    duration = result['current_stage_seconds']/60
    english = f'''# Actual local experiment results

[简体中文](RESULTS.zh-CN.md)

The primary Qwen2.5-0.5B / LoRA protocol completed {len(manifest['stages'])} stage receipts on an Apple M1 Pro (16 GB), with CPU training/scoring and MPS generation with recorded CPU retries, including three standard ARM rounds, two ablations, four public benchmark slices and downstream policy paths. All {checked} recorded stage artifact hashes match. The current completed-stage receipts total {duration:.1f} minutes; this excludes retired attempts and supplementary jobs. The separate 135M CPU smoke run completed in about 10 minutes.

An earlier MPS-training run failed a [fixed-weight gradient and finite-difference audit](../docs/GRADIENT_AUDIT.md). Its finite losses did not establish correct backward gradients. This report uses a fresh CPU-autograd run; no trained weights from that attempt were reused. The old artifacts remain explicitly marked as diagnostic only.

During public-benchmark evaluation, one long-input MPS worker exceeded 20 minutes under heavy swapping. A reviewed recovery fix added a five-minute worker deadline and a fresh CPU retry with unchanged weights, prompts, seeds and decoding budgets. The incomplete benchmark stage was archived and fully rerun; completed CPU training was retained. The manifest's code-fix history and source snapshots preserve this boundary. See the [engineering notes](../docs/M1_ENGINEERING.md).

These are real small-scale executions, **not a reproduction of the papers' model scale or published scores**. Some stages finish by correctly skipping an empty optimization set. See the [coverage audit](../docs/PAPER_COVERAGE.md) for unexecuted original-scale work.

## Pairwise results

{shared}

Pairs repeat across variants, and multiple benchmark comparisons share prompts. Counts are not numbers of independent test tasks. Both primary held-out evaluation labels are B; the full four-row validation slice is also B-only. Training has 8 A and 16 B labels. This small, imbalanced sample cannot establish general preference accuracy. Benchmark response order is independently shuffled with a fixed seed.

Across the four public benchmark slices, individual sampled judgments included {public['correct_votes']} correct, {public['wrong_votes']} valid-but-wrong and {public['invalid_votes']} invalid outputs. Of the invalid outputs, {public['invalid_at_limit']} reached the token cap, {public['invalid_before_limit']} stopped before it, and {public['invalid_without_generation']} had no generation record. A token-limit flag and an invalid decision are distinct measurements: a model may emit a decision and continue, or stop without one.

## Decoding budget and order

With unchanged SFT weights, raising the judge cap from 320 to 768 tokens changed invalid output from {pct(short['invalid_rate'])} to {pct(long['invalid_rate'])}, token-limit frequency from {pct(short['length_limit_rate'])} to {pct(long['length_limit_rate'])}, and voting accuracy from {pct(short['majority_accuracy'])} to {pct(long['majority_accuracy'])}. This is a separate diagnostic, not a replacement of the primary result. Inspect raw texts before attributing every failure to truncation. The compact prompt's brevity request and the original long teacher trajectories are not perfectly matched. Its format illustration also uses `Winner: A`; this run does not separately isolate any position effect caused by that example. These are possible prompt confounds, not established explanations for every failure.

The rubric cap remains {rubric_limits['token_cap']} tokens in both settings; {rubric_limits['at_token_limit']}/{rubric_limits['unique_prompts']} unique SFT evaluation rubrics reached that limit. Increasing only the judge budget therefore does not test a fully unconstrained two-stage pipeline. The joint-budget check below addresses this additional limitation.

The primary, judge-only and joint-budget judgments ran on `{short['actual_devices']}`, `{long['actual_devices']}` and `{joint['actual_devices']}`, respectively; {joint['fallback_generations']} joint-budget votes used CPU recovery. CPU and MPS sampling are not promised to produce identical tokens for the same seed. This is therefore not a device-controlled causal estimate of generation-budget effects. The [analysis amendment](supplementary/provenance/analysis-device-note.json) records this reporting correction without changing data or metrics.

The [teacher-length audit](supplementary/teacher-target-lengths.json) found that all 24 training judgments exceed 320 tokens (median {teacher_lengths['judge']['median']:.0f}), while 19/24 rubrics exceed 128 tokens (median {teacher_lengths['rubric']['median']:.1f}). This prompted a separate exploratory run with rubric/judge caps of 320/1024 and context 4096, declared before supplementary execution. With the same SFT weights and pairs, invalid output was {pct(joint['invalid_rate'])}, judge token-limit frequency {pct(joint['length_limit_rate'])}, and voting accuracy {pct(joint['majority_accuracy'])}; {joint_rubric_limits}/{len(joint_rubrics)} rubrics reached their new cap. It changes both budgets, so it cannot attribute an effect to either one alone. Teacher target lengths do not guarantee that the small model will terminate at those lengths.

{order_md}

The supplementary order check matches five votes, temperature 0.7 and seeds; comparisons additionally require identical generated rubrics and valid votes in both directions. The primary single reversed generation uses greedy decoding and cannot isolate position effects. No valid comparable decisions means the consistency rate is undefined, not evidence of zero position bias.

## What the updates actually learned

{phase_md}

The table separates correctness rewards from total rewards, which also contain format shaping for judge phases. A nonzero gradient driven by formatting is not evidence of improved correctness. Zero-variance groups supply no preference advantage; fixed-reference KL and optimizer effects may still change weights. SFT losses compare different training examples and are not validation-loss curves.

Fresh local CRG retained {manifest['stages']['01-curation']['result']['retained']} of {manifest['stages']['01-curation']['result']['attempted']} attempts. SFT used the separately released teacher-curated data, not invented successful local CRG records.

ARM-labeled DPO: `{dpo.get('status', 'completed')}`, {dpo.get('steps', 0)} steps. Separate source-label DPO: {source_dpo['steps']} steps. IterDPO completed {len(iterations['rounds'])} rounds with step counts {[r['steps'] for r in iterations['rounds']]}. Online GRPO had {online['zero_variance_groups']}/{online['groups']} zero-variance groups and mean reward {online['reward_mean']:.4f}. Empty ARM labels were never replaced with source labels.

The online judge awarded {online['reward_diagnostics']['full_wins']} full two-order wins and {online['reward_diagnostics']['half_credit']} half-credit rewards; {online['reward_diagnostics']['half_credit_with_one_valid_order']} half-credit cases had only one valid judgment direction. Such rewards can reflect output-format reliability rather than a demonstrated preference improvement. The maximum pre-clipping CPU gradient norm was {online['reward_diagnostics']['max_pre_clip_gradient_norm']:.6f}, with a clipping threshold of 1. These are judge rewards, not independent instruction-following scores.

## Official IFEval policy checks

{policy_md}

These four seeded prompts use official strict/loose verifiers and a 512-token response cap. Their low sample size and output budget limit interpretation. The source-DPO policy is a baseline; its score is not attributed to ARM-generated preference labels.

ARM-DPO and IterDPO policy checks are included only if those stages actually produced trained adapters. Empty accepted sets do not create substitute policy scores.

![Actual diagnostic measurements](supplementary/diagnostics.png)

## Reproducibility and remaining limits

[Combined JSON](supplementary/combined.json), [CSV](supplementary/combined-metrics.csv), [primary raw artifacts](m1-cpu-training-20261116/), and [supplementary raw artifacts](supplementary/) contain the evidence. Model/data revisions, code snapshots, failed runs, token IDs, rewards and checkpoint hashes are retained. [Engineering measurements](../docs/M1_ENGINEERING.md) distinguish sampled driver allocations from whole-process memory. The final cache-backed continuation used no hosted model APIs.

The experiments do not establish improved alignment or reproduce the theorem's empirical premises. Full 4B/8B training, original-scale data and model sweeps, external reward-model baselines, derived FollowBench/InfoBench pairs, external-judge policy benchmarks and clinical-domain transfer remain unexecuted. No scores were fabricated for those cells.
'''
    chinese = f'''# 本地真实实验结果

[English](RESULTS.md)

主实验在 Apple M1 Pro（16 GB）上以 Qwen2.5-0.5B／LoRA、CPU 训练及概率计算、MPS 生成及有记录的 CPU 重试完成 {len(manifest['stages'])} 个阶段凭据，包括三轮标准 ARM、两项消融、四套公开基准切片及下游策略路径。{checked} 个阶段产物哈希全部核验通过。当前有效阶段的耗时合计 {duration:.1f} 分钟，不含已退役的诊断尝试与补充实验。独立的 135M CPU smoke 流程约耗时 10 分钟。

先前的 MPS 训练未通过[固定权重梯度及有限差分核查](../docs/GRADIENT_AUDIT.md)，损失为有限数并不代表反向梯度正确。本报告使用重新从基础模型开始的 CPU 自动微分实验，不复用旧尝试的训练权重；旧产物明确标记为仅供诊断。

公开基准评测时，一批长输入的 MPS 子进程在高换页压力下运行超过 20 分钟。经过审查的恢复修复加入了五分钟子进程上限，超时后用相同权重、提示词、种子及解码预算在新 CPU 进程重试。未完成的基准阶段已归档并完整重跑，已完成的 CPU 训练保留。清单中的代码修复记录及源码快照保留这一边界，详见[工程说明](../docs/M1_ENGINEERING.md)。

这是真实的小规模执行，**不是论文模型规模与发表分数的复现**。部分阶段因没有有效偏好标签而正确跳过优化。原始规模实验的缺口见[覆盖审计](../docs/PAPER_COVERAGE.md)。

## 回答对评测

{shared}

各变体重复使用留出样本，部分基准比较共享提示词，因此表中条数不是独立任务数。主评测的两条标准偏好均为 B，完整四条验证切片也均为 B；24 条训练数据中 A/B 为 8/16。这种极小且不平衡的切片不能代表总体偏好能力。公开基准的回答顺序使用固定种子独立打乱。

四套公开基准的逐次判定中，正确 {public['correct_votes']} 次，有效但判错 {public['wrong_votes']} 次，无效 {public['invalid_votes']} 次。无效输出中，{public['invalid_at_limit']} 次达到长度上限，{public['invalid_before_limit']} 次提前停止，{public['invalid_without_generation']} 次没有生成记录。达到 token 上限与判定无效是不同指标：模型可能先给出判定再继续生成，也可能提前停止却没有给出判定。

## 输出预算与顺序控制

固定 SFT 权重、样本和五票设置，将裁判预算从 320 提高至 768 tokens 后，无效输出率由 {pct(short['invalid_rate'])} 变为 {pct(long['invalid_rate'])}，达到长度上限的比例由 {pct(short['length_limit_rate'])} 变为 {pct(long['length_limit_rate'])}，投票准确率由 {pct(short['majority_accuracy'])} 变为 {pct(long['majority_accuracy'])}。这是独立诊断，不替换主实验结果。不能把所有错误都归因于截断；紧凑提示词的简短要求与较长教师轨迹不完全匹配，格式示例还使用了 `Winner: A`。本实验未单独识别这些提示词因素及示例可能引入的位置影响，不能将其视为所有失败的已证实原因。

两种设置的 rubric 预算均保持 {rubric_limits['token_cap']} tokens；SFT 评测中 {rubric_limits['at_token_limit']}/{rubric_limits['unique_prompts']} 条独立 rubric 达到了长度上限。因此，仅放宽裁判预算并不等于验证了完整、无限长的两阶段流程；下面的双预算诊断进一步检查这一限制。

主配置、仅放宽裁判预算、双预算配置的实际裁判设备分别为 `{short['actual_devices']}`、`{long['actual_devices']}`、`{joint['actual_devices']}`；双预算配置中有 {joint['fallback_generations']} 次判决通过 CPU 回退完成。CPU 与 MPS 即使使用相同种子，也不保证逐 token 一致，因此这不是严格控制设备后的纯预算因果实验。[分析修订记录](supplementary/provenance/analysis-device-note.json)保存了这一说明的变更，不改变实验数据或指标。

[教师长度核查](supplementary/teacher-target-lengths.json)发现：24 条训练判决全部超过 320 tokens，中位数为 {teacher_lengths['judge']['median']:.0f}；19/24 条 rubric 超过 128 tokens，中位数为 {teacher_lengths['rubric']['median']:.1f}。因此，在补充实验执行前另行声明了探索性的双预算诊断：rubric／判决上限为 320／1024、上下文为 4096，权重与样本保持不变。其无效率为 {pct(joint['invalid_rate'])}，裁判达到上限的比例为 {pct(joint['length_limit_rate'])}，投票准确率为 {pct(joint['majority_accuracy'])}；{joint_rubric_limits}/{len(joint_rubrics)} 条 rubric 达到了新上限。该实验同时改变了两项预算，不能将变化单独归因于其中一项；教师目标长度也不保证小模型会在相同长度结束。

{order_md}

补充顺序检查在正反顺序下保持五票、温度 0.7 和随机种子一致；有效受控比较还要求生成的 rubric 完全一致、正反两侧投票均有效。主配置的单次反序输出采用贪心解码，只能作为混合解码诊断。如果没有符合条件的比较，顺序一致率为未定义，不能解释为不存在位置偏差。

## 实际优化信号

{phase_md}

表格区分正确性奖励与包含格式塑形的总奖励。由格式差异产生的非零梯度不代表判断正确率提升。奖励方差为零的组没有偏好优势信号，但固定参考 KL 与优化器效应仍可能使权重变化。SFT 各步使用不同样本，损失曲线不等于验证集学习曲线。

本地 CRG 的 {manifest['stages']['01-curation']['result']['attempted']} 次尝试保留了 {manifest['stages']['01-curation']['result']['retained']} 条；SFT 使用独立发布的教师筛选数据，没有伪造本地 CRG 成功样本。

ARM 标签 DPO 状态为 `{dpo.get('status', 'completed')}`，执行 {dpo.get('steps', 0)} 步。独立源标签 DPO 执行 {source_dpo['steps']} 步。IterDPO 共 {len(iterations['rounds'])} 轮，各轮更新次数为 {[r['steps'] for r in iterations['rounds']]}。在线 GRPO 的 {online['groups']} 组中有 {online['zero_variance_groups']} 组奖励方差为零，平均奖励为 {online['reward_mean']:.4f}。没有用源标签填补 ARM 的无效标签。

在线裁判给出 {online['reward_diagnostics']['full_wins']} 次双顺序全胜及 {online['reward_diagnostics']['half_credit']} 次半额奖励；其中 {online['reward_diagnostics']['half_credit_with_one_valid_order']} 次半额奖励只有一个顺序的判决有效。这类奖励可能反映输出格式的可靠程度，不能直接证明偏好能力提升。CPU 裁剪前最大梯度范数为 {online['reward_diagnostics']['max_pre_clip_gradient_norm']:.6f}，裁剪上限为 1。这些是裁判奖励，不是独立的指令遵循得分。

## 官方 IFEval 策略检查

{policy_md}

四条固定种子选出的提示词使用官方 strict／loose 验证器，每条输出预算为 512 tokens。样本量及长度预算限制了结论；源标签 DPO 的分数属于独立基线，不能归功于 ARM 偏好标签。

只有 ARM-DPO 或 IterDPO 实际产生训练后的适配器时，才增加相应策略评测；空标签集不会被替换成其他策略的分数。

![实际诊断测量](supplementary/diagnostics.png)

## 证据与未完成范围

[汇总 JSON](supplementary/combined.json)、[CSV](supplementary/combined-metrics.csv)、[主实验原始记录](m1-cpu-training-20261116/)与[补充原始记录](supplementary/)保留实际证据。模型／数据版本、代码快照、失败尝试、token、奖励和权重哈希均已记录。[工程说明](../docs/M1_ENGINEERING.md)区分了 MPS 分配量与整进程物理内存。后续离线运行没有调用托管模型 API。

这些结果不足以证明对齐效果提升，也不是理论定理的实验验证。原始 4B／8B 全量训练、原规模数据及模型比较、外部奖励模型基线、FollowBench／InfoBench 派生回答对、依赖外部裁判的策略基准及医疗领域迁移尚未执行；对应结果没有填入虚构分数。
'''
    (ROOT/'experiments/RESULTS.md').write_text(english)
    (ROOT/'experiments/RESULTS.zh-CN.md').write_text(chinese)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(3)
    budgets = [short, long, joint]
    axes[0].bar(x-.17, [100*r['invalid_rate'] for r in budgets], .34, label='Invalid')
    axes[0].bar(x+.17, [100*r['length_limit_rate'] for r in budgets], .34, label='Judge token limit')
    axes[0].set(xticks=x, xticklabels=['128 / 320', '128 / 768', '320 / 1024'], ylim=(0, 110),
                xlabel='Rubric / judge token caps',
                ylabel='Percent', title='Fixed-weight decoding diagnostic (2 pairs)')
    axes[0].legend()
    x = np.arange(len(policy))
    axes[1].bar(x-.17, [100*r['strict']['prompt_accuracy'] for r in policy.values()], .34, label='Strict')
    axes[1].bar(x+.17, [100*r['loose']['prompt_accuracy'] for r in policy.values()], .34, label='Loose')
    policy_labels = {'base': 'Base', 'source-dpo': 'Source DPO', 'online-grpo': 'Online GRPO',
                     'arm-dpo': 'ARM DPO', 'iterative-dpo': 'IterDPO'}
    axes[1].set(xticks=x, xticklabels=[policy_labels[name] for name in policy], ylim=(0, 110),
                ylabel='Prompt accuracy (%)', title='Official IFEval verifiers (4 prompts)')
    axes[1].legend()
    fig.suptitle('Actual bounded measurements; not paper-scale scores', weight='bold')
    fig.tight_layout()
    fig.savefig(SUPP/'diagnostics.png', dpi=160)
    plt.close(fig)
    plan = read(SUPP/'plan.json')
    plan['status'] = 'experiments_completed'
    write_json(SUPP/'plan.json', plan)
    print('Saved experiments/RESULTS.md and RESULTS.zh-CN.md', flush=True)


if __name__ == '__main__':
    main()
