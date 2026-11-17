"""Export bounded, provenance-preserving HelpSteer3 and PPE-IFEval pair diagnostics.

This is an experiment-specific export, not the papers' unpublished pair selection
or the full PPE Best-of-K aggregation. No model-generated labels are used.
"""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import json
import random

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from openrubrics_lab.config import load_config
from openrubrics_lab.data import prompt_key, validate_pair
from openrubrics_lab.io import digest, file_hash, read_jsonl, write_json, write_jsonl

SOURCES = {
    'helpsteer3': ('nvidia/HelpSteer3', 'f6d145777bcbde96137596340fab89793acd1031',
                  'preference/validation.jsonl.gz'),
    'ppe-ifeval': ('lmarena-ai/PPE-IFEval-Best-of-K', 'a4f7fae2443d96529dee6b65863a816a051b3a4d',
                  'data/train-00000-of-00001.parquet'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/m1.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1 or args.output.exists():
        parser.error('Use a positive sample count and a new output directory')
    config = load_config(args.config)
    tok = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'])
    excluded = {prompt_key(r['instruction']) for f in ['train.jsonl', 'validation.jsonl']
                for r in read_jsonl(Path('data/m1')/f)}
    args.output.mkdir(parents=True)
    receipts = {}
    for name, (repo, revision, filename) in SOURCES.items():
        source = Path(hf_hub_download(repo, filename, repo_type='dataset', revision=revision))
        if name == 'helpsteer3':
            with gzip.open(source, 'rt') as f:
                raw = [json.loads(line) for line in f]
        else:
            raw = pq.read_table(source).to_pylist()
        indexed = list(enumerate(raw))
        random.Random(config['seed']).shuffle(indexed)
        selected, skipped, seen = [], Counter(), set()
        for index, row in indexed:
            evidence = {}
            if name == 'helpsteer3':
                context = row['context']
                # Preserve the whole request; do not drop turns to make it fit.
                if len(context) != 1 or context[0]['role'] != 'user':
                    skipped['multi_turn'] += 1
                    continue
                preference = row['overall_preference']
                if preference == 0:
                    skipped['tie'] += 1
                    continue
                prompt, a, b = context[0]['content'], row['response1'], row['response2']
                label = 'A' if preference < 0 else 'B'
                subset = row['domain']
                evidence = {'overall_preference': preference, 'language': row['language'],
                            'source_row': index}
            else:
                # Only compare an unambiguously passing answer with one failing
                # both official prompt-level strict and loose stored checks.
                scores = row['score_data']
                assert len(scores) == 32
                positive = [i for i, s in enumerate(scores)
                            if s['prompt_level_strict_acc'] and s['prompt_level_loose_acc']]
                negative = [i for i, s in enumerate(scores)
                            if not s['prompt_level_strict_acc'] and not s['prompt_level_loose_acc']]
                if not positive or not negative:
                    skipped['no_unambiguous_conflict'] += 1
                    continue
                rng = random.Random(config['seed']+index)
                ci, ri = rng.choice(positive), rng.choice(negative)
                prompt = row['ifeval_prompt']
                a, b = row[f'response_{ci+1}'], row[f'response_{ri+1}']
                label, subset = 'A', 'ifeval'
                evidence = {'question_id': row['question_id'], 'key': row['key'],
                            'response_model': row['model_name'], 'chosen_index_zero_based': ci,
                            'rejected_index_zero_based': ri, 'chosen_checks': scores[ci],
                            'rejected_checks': scores[ri], 'instruction_id_list': row['instruction_id_list'],
                            'instruction_kwargs': row['kwargs']}
            key = prompt_key(prompt)
            if key in excluded or key in seen:
                skipped['prompt_overlap_or_duplicate'] += 1
                continue
            estimate = len(tok.encode(prompt+a+b))+config['rubric_tokens']+config['judge_tokens']+420
            if estimate > config['max_context']:
                skipped['context_overflow'] += 1
                continue
            sample_id = f'{name}:{index}'
            if int(digest(sample_id+str(config['seed']))[:8], 16) % 2:
                a, b, label = b, a, 'B' if label == 'A' else 'A'
            selected.append(validate_pair({'id': sample_id, 'instruction': prompt,
                'response_a': a, 'response_b': b, 'label': label, 'subset': subset,
                'source': repo, 'source_revision': revision, 'benchmark': name,
                'source_evidence': evidence}))
            seen.add(key)
            if len(selected) == args.samples:
                break
        if len(selected) != args.samples:
            raise ValueError(f'{name}: insufficient eligible pairs')
        target = args.output/f'{name}.jsonl'
        write_jsonl(target, selected)
        receipts[name] = {'repository': repo, 'revision': revision, 'file': filename,
            'source_sha256': file_hash(source), 'download_bytes': source.stat().st_size,
            'source_rows': len(raw), 'pairs': len(selected), 'excluded_before_selection': dict(skipped),
            'output_sha256': file_hash(target),
            'scope': 'Bounded single-turn/context-fitting pair diagnostic; not full paper or Best-of-K scores'}
    write_json(args.output/'manifest.json', {'sources': receipts, 'seed': config['seed'],
        'samples_per_source': args.samples, 'config': config, 'exporter_sha256': file_hash(Path(__file__)),
        'selection': 'Seeded source-row shuffle; no training/validation overlap; no response truncation'})
    print(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    main()
