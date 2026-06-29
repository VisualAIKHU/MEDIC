import os
import sys
import argparse
import json
import contextlib
import io

from collections import defaultdict

from utils.eval_utils_dc import score_generation, \
    score_generation_by_type, \
    coco_gen_format_save

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True)
parser.add_argument('--anno', required=True)
parser.add_argument('--iter', required=True, type=int)
args = parser.parse_args()
results = os.listdir(args.results_dir)
results_path = os.path.join(args.results_dir, 'eval_results.txt')

results = [res for res in results if res != 'eval_results.txt']

total_best_results = defaultdict(lambda : ('iter', -10000))
sc_best_results = defaultdict(lambda : ('iter', -10000))

line_eval_result = None 

f = open(results_path, 'w')
for res in results:
    path = os.path.join(args.results_dir, res)
    sc_path = os.path.join(path, 'sc_results.json')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sc_eval_result = score_generation(args.anno, sc_path)
    sc_captions = json.load(open(sc_path, 'r'))
    message = '===================={} results===================\n'.format(res)
    message += '-------------semantic change captions only----------\n'
    for k, v in sc_eval_result.items():
        iter_name , prev_best = sc_best_results[k]
        if prev_best < v:
            sc_best_results[k] = (res, v)
        message += '{}: {}\n'.format(k, v)

    f.write(message)

    if len(results) == 1 or str(args.iter) == str(res):
        line_eval_result = sc_eval_result

if line_eval_result is not None:
    metric_order = ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
    summary = [f"iter {args.iter:5d}"]
    for m in metric_order:
        if m in line_eval_result:
            summary.append(f"{m}: {line_eval_result[m]:.3f}")
    print("  ".join(summary))
else:
    print(f"iter {args.iter:5d} (no matching res folder to log)")

summary_message = '\n\n\n=========Results Summary==========\n'
summary_message += '------------semantic change best result-------------\n'
for metric, pairs in sc_best_results.items():
    summary_message += '{}: {} ({})\n'.format(metric, pairs[1], pairs[0])

f.write(summary_message)
f.close()
