import os
import argparse
import json
import time
import sys
import numpy as np
import torch
torch.backends.cudnn.enabled  = True
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.config_transformer_dirl import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.model import DIRL, AddSpatialInfo
from models.CCR_expert import CCR
from utils.utils import AverageMeter, accuracy, set_mode, load_checkpoint, \
                        decode_sequence, decode_sequence_transformer, coco_gen_format_save
from utils.vis_utils import visualize_att
from tqdm import tqdm

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--snapshot', type=int, required=True)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--exp_dir', type=str, default='./exp')
parser.add_argument('--run_name', type=str, default='run')
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
# assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

# Device configuration
use_cuda = torch.cuda.is_available()
if args.gpu == -1:
    gpu_ids = cfg.gpu_id
else:
    gpu_ids = [args.gpu]
torch.backends.cudnn.enabled  = True
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

cfg.exp_dir = args.exp_dir
cfg.exp_name = args.run_name

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name

output_dir = os.path.join(exp_dir, exp_name)

test_output_dir = os.path.join(output_dir, f'test_output_{args.snapshot}')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)
caption_output_path = os.path.join(test_output_dir, 'captions', 'test')
if not os.path.exists(caption_output_path):
    os.makedirs(caption_output_path)
att_output_path = os.path.join(test_output_dir, 'attentions', 'test')
if not os.path.exists(att_output_path):
    os.makedirs(att_output_path)

if args.visualize:
    visualize_save_dir = os.path.join(test_output_dir, 'visualizations')
    if not os.path.exists(visualize_save_dir):
        os.makedirs(visualize_save_dir)

snapshot_dir = os.path.join(output_dir, 'snapshots')
snapshot_file = '%s_checkpoint_%d.pt' % (exp_name, args.snapshot)
snapshot_full_path = os.path.join(snapshot_dir, snapshot_file)
checkpoint = load_checkpoint(snapshot_full_path)
change_detector_state = checkpoint['change_detector_state']
speaker_state = checkpoint['speaker_state']


# Load modules
change_detector = DIRL(cfg)
change_detector.load_state_dict(change_detector_state)
change_detector = change_detector.to(device)

speaker = CCR(cfg)
speaker.load_state_dict(speaker_state)
speaker.to(device)

spatial_info = AddSpatialInfo()
spatial_info.to(device)

print(change_detector)
print(speaker)
# print(spatial_info)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
idx_to_word = train_dataset.get_idx_to_word()
test_dataset, test_loader = create_dataset(cfg, 'test')


set_mode('eval', [change_detector, speaker]) 
with torch.no_grad():
    test_iter_start_time = time.time()

    full_type_probs_pos = {}
    full_type_probs_neg = {}

    result_sents_pos_2 = {}
    result_sents_neg_2 = {}

    result_sents_pos = {}
    result_sents_neg = {}
    for i, batch in tqdm(enumerate(test_loader)):

        d_feats, nsc_feats, sc_feats, labels, labels_with_ignore, \
        no_chg_labels, no_chg_labels_with_ignore, masks, no_chg_masks, \
        aux_labels_pos, aux_labels_neg, d_img_paths, nsc_img_paths, sc_img_paths = batch

        val_batch_size = d_feats.size(0)

        d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device)
        d_feats, nsc_feats, sc_feats = spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
        no_chg_labels, no_chg_labels_with_ignore, no_chg_masks = no_chg_labels.to(device), no_chg_labels_with_ignore.to(device), no_chg_masks.to(device)
        aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

        diff_bef_pos, diff_aft_pos, full_logits_pos, weighted_pos = change_detector(d_feats, sc_feats, aux_labels_pos)
        diff_bef_neg, diff_aft_neg, full_logits_neg, weighted_neg = change_detector(d_feats, nsc_feats, aux_labels_neg)
        
        diff_cat_pos = torch.cat([diff_bef_pos, diff_aft_pos], dim=-1)  # (B, N, 2D)
        diff_cat_neg = torch.cat([diff_bef_neg, diff_aft_neg], dim=-1)  # (B, N, 2D)

        speaker_output_pos, pos_att = speaker.sample(diff_cat_pos, weighted_pos, sample_max=1)
        speaker_output_neg, neg_att = speaker.sample(diff_cat_neg, weighted_neg, sample_max=1)
        
        gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:])
        gen_sents_neg = decode_sequence_transformer(idx_to_word, speaker_output_neg[:, 1:])

        for j in range(val_batch_size):
            gts = decode_sequence_transformer(idx_to_word, labels[j][:, 1:])
            gts_neg = decode_sequence_transformer(idx_to_word, no_chg_labels[j][:, 1:])
            sent_pos = gen_sents_pos[j]
            sent_neg = gen_sents_neg[j]
            image_id = d_img_paths[j].split('_')[-1]
            image_num = image_id.split('.')[0]

            result_sents_pos[image_id] = sent_pos
            result_sents_neg[image_id + '_n'] = sent_neg

            att_bef_path = os.path.join(att_output_path, image_num + '_before')
            att_aft_path = os.path.join(att_output_path, image_num + '_after')

            prob_pos = torch.softmax(full_logits_pos[j], dim=-1).detach().cpu().numpy()
            prob_neg = torch.softmax(full_logits_neg[j], dim=-1).detach().cpu().numpy()

            label_pos = aux_labels_pos[j].item()
            label_neg = aux_labels_neg[j].item()

            pred_label_pos = full_logits_pos[j].argmax(dim=-1).item()
            pred_label_neg = full_logits_neg[j].argmax(dim=-1).item()

            prob_str_pos = ' '.join([f'{p:.4f}' for p in prob_pos])
            prob_str_neg = ' '.join([f'{p:.4f}' for p in prob_neg])

            full_type_probs_pos[image_id] = f"{label_pos} | {prob_str_pos}"
            full_type_probs_neg[image_id] = f"{label_neg} | {prob_str_neg}"

            result_sents_pos_2[image_id] = {
                "caption": sent_pos,
                "image_id": image_id,
                "prediction": pred_label_pos,
                "change_type": label_pos
            }
            result_sents_neg_2[image_id + '_n'] = {
                "caption": sent_neg,
                "image_id": image_id + '_n',
                "prediction": pred_label_neg,
                "change_type": label_neg
            }

    change_type_output_path_pos = os.path.join(caption_output_path, 'predicted_6_types_pos.json')
    change_type_output_path_neg = os.path.join(caption_output_path, 'predicted_6_types_neg.json')
    with open(change_type_output_path_pos, 'w') as f:
        json.dump(full_type_probs_pos, f, indent=4)
    with open(change_type_output_path_neg, 'w') as f:
        json.dump(full_type_probs_neg, f, indent=4)

    test_iter_end_time = time.time() - test_iter_start_time

    result_save_path_pos = os.path.join(caption_output_path, 'sc_results.json')
    result_save_path_neg = os.path.join(caption_output_path, 'nsc_results.json')
    coco_gen_format_save(result_sents_pos, result_save_path_pos)
    coco_gen_format_save(result_sents_neg, result_save_path_neg)

    result_save_path_pos_2 = os.path.join(caption_output_path, 'sc_results_data.json')
    result_save_path_neg_2 = os.path.join(caption_output_path, 'nsc_results_data.json')
    with open(result_save_path_pos_2, 'w') as f:
        json.dump(list(result_sents_pos_2.values()), f, indent=4)
    with open(result_save_path_neg_2, 'w') as f:
        json.dump(list(result_sents_neg_2.values()), f, indent=4)
