import os
import sys
import json
import argparse
import time
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import random
import wandb

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs.config_transformer_medic import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.model import DIRL, AddSpatialInfo, init_weights
from models.CCR_expert import CCR
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, load_checkpoint, \
                        LanguageModelCriterion, decode_sequence, decode_sequence_transformer, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode, \
                        EntropyLoss, LabelSmoothingLoss

from utils.vis_utils import visualize_att
from easydict import EasyDict
from torch.cuda.amp import autocast, GradScaler
from scheduler import build_scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
# parser.add_argument('--entropy_weight', type=float, default=0.0)
parser.add_argument('--visualize_every', type=int, default=10)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--exp_dir', type=str, default='./exp')
parser.add_argument('--run_name', type=str, default='debugging')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint for resuming training")
parser.add_argument('--lambda_disentangle', type=float, default=0.1, help='disentangle loss weight')
parser.add_argument('--lambda_consistency', type=float, default=0.1, help='consistency loss weight')
parser.add_argument('--cls_temperature', type=float, default=1.0, help='temperature for change classification')
parser.add_argument('--type_temperature', type=float, default=1.0, help='temperature for change type classification')
parser.add_argument('--lambda_cls', type=float, default=0.1, help='weight for change classification loss')
parser.add_argument('--lambda_type', type=float, default=0.1, help='weight for change type classification loss')

args = parser.parse_args()
merge_cfg_from_file(args.cfg)
# print(os.path.basename(args.cfg).replace('.yaml', ''))
# assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

cfg.exp_dir = args.exp_dir
cfg.exp_name = args.run_name
cfg.train.optim.lr = args.lr
cfg.data.train.batch_size = args.batch_size
cfg.train.optim.weight_decay = args.weight_decay
cfg.train.max_iter = args.max_iter
cfg.lambda_disentangle = args.lambda_disentangle
cfg.lambda_consistency = args.lambda_consistency
cfg.lambda_cls = args.lambda_cls
cfg.lambda_type = args.lambda_type
cfg.cls_temperature = args.cls_temperature
cfg.type_temperature = args.type_temperature

run = wandb.init(
    project='medic_chg',
    name = f'{args.exp_dir}/{args.run_name}',
    config={
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "max_iter": args.max_iter,
        "disentangle": args.lambda_disentangle,
        "consistency": args.lambda_consistency,
        "cls_temp.": args.cls_temperature,
        "type_temp.": args.type_temperature,
        "cls:": cfg.lambda_cls,
        "type:": cfg.lambda_type,
    },
    notes = f'{args.exp_dir, args.run_name}',
    reinit=True
)

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

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_file_save = os.path.join(output_dir, 'cfg.json')
json.dump(cfg, open(cfg_file_save, 'w'))

sample_dir = os.path.join(output_dir, 'eval_gen_samples')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
sample_subdir_format = '%s_samples_%d'

sent_dir = os.path.join(output_dir, 'eval_sents')
if not os.path.exists(sent_dir):
    os.makedirs(sent_dir)
sent_subdir_format = '%s_sents_%d'

snapshot_dir = os.path.join(output_dir, 'snapshots')
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_file_format = '%s_checkpoint_%d.pt'

train_logger = Logger(cfg, output_dir, is_train=True)
val_logger = Logger(cfg, output_dir, is_train=False)

random.seed(1111)
np.random.seed(1111)
torch.manual_seed(1111)

# Load modules
change_detector = DIRL(cfg)
change_detector.apply(init_weights)
for expert in change_detector.expert_modules_change:
    expert.reset_parameters()
change_detector.expert_module_no_change.reset_parameters()
change_detector.to(device)

speaker = CCR(cfg)
speaker.to(device)

spatial_info = AddSpatialInfo()

print(change_detector)
print(speaker)

with open(os.path.join(output_dir, 'model_print'), 'w') as f:
    print(change_detector, file=f)
    print(speaker, file=f)
    print(spatial_info, file=f)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'val')
train_size = len(train_dataset)
val_size = len(val_dataset)

# all_params = list(change_detector.parameters()) + list(speaker.parameters())
# optimizer = build_optimizer(all_params, cfg)

trainable_params = list(filter(lambda p: p.requires_grad, 
                        list(change_detector.parameters()) + list(speaker.parameters())))
optimizer = build_optimizer(trainable_params, cfg)
# optimizer = AdamW(
#     trainable_params,
#     lr=args.lr,
#     weight_decay=args.weight_decay,
#     betas=(0.9, 0.999)
# )

scaler = GradScaler()

# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=cfg.train.optim.step_size,
#     gamma=cfg.train.optim.gamma)

# lr_scheduler = CosineAnnealingLR(
#     optimizer,
#     T_max=args.max_iter,
#     eta_min=1e-6
# )

# Train loop
t = 0
epoch = 0

if args.resume_checkpoint:
    checkpoint_path = args.resume_checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        change_detector.load_state_dict(checkpoint['change_detector_state'], strict=False)
        speaker.load_state_dict(checkpoint['speaker_state'])

        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

        scaler.load_state_dict(checkpoint['scaler_state'])

        t = checkpoint['iteration']
        epoch = checkpoint['epoch']

        scheduler_cfg = EasyDict({
            'type_name': 'PolyLR',
            'keywords': {   
                'gamma': cfg.train.optim.gamma,        
                'n_iteration': args.max_iter      
            }
        })
        lr_scheduler = build_scheduler(scheduler_cfg, optimizer, last_epoch=t-1)
        # lr_scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=args.max_iter,
        #     eta_min=1e-6
        # )
        if 'lr_scheduler_state' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        else:
            lr_scheduler.step()  # fallback
        # lr_scheduler.step()

        # print(f"Restored iteration {t}, lr: {optimizer.param_groups[0]['lr']}")
        # raise RuntimeError

        print(f"Resumed from iteration {t}, epoch {epoch}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Starting fresh.")
        scheduler_cfg = EasyDict({
            'type_name': 'PolyLR',
            'keywords': {
                'gamma': cfg.train.optim.gamma,      
                'n_iteration': args.max_iter 
            }
        })
        lr_scheduler = build_scheduler(scheduler_cfg, optimizer)
        # lr_scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=args.max_iter,
        #     eta_min=1e-6
        # )
else:
    print("No checkpoint provided. Starting fresh.")
    scheduler_cfg = EasyDict({
            'type_name': 'PolyLR',
            'keywords': {
                'gamma': cfg.train.optim.gamma,   
                'n_iteration': args.max_iter     
            }
        })
    lr_scheduler = build_scheduler(scheduler_cfg, optimizer)
    # lr_scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=args.max_iter,
    #     eta_min=1e-6
    # )

set_mode('train', [change_detector, speaker])
# ss_prob = speaker.ss_prob

while t < cfg.train.max_iter:
    epoch += 1
    print('Starting epoch %d' % epoch)
    # lr_scheduler.step()
    speaker_loss_avg = AverageMeter()
    cdcr_loss_avg = AverageMeter()
    sim_loss_avg = AverageMeter()
    change_cls_loss_avg = AverageMeter()    
    change_type_loss_avg = AverageMeter()
    expert_disentangle_loss_avg = AverageMeter()
    expert_consistency_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()

    if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0:
        frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
        ss_prob_prev = ss_prob
        ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                      cfg.train.scheduled_sampling_max_prob)
        speaker.ss_prob = ss_prob
        if ss_prob_prev != ss_prob:
            print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))

    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()
        
        # 5+4+6
        d_feats, nsc_feats, sc_feats, labels, labels_with_ignore, \
        no_chg_labels, no_chg_labels_with_ignore, masks, no_chg_masks, \
        aux_labels_pos, aux_labels_neg, d_img_paths, nsc_img_paths, sc_img_paths = batch

        batch_size = d_feats.size(0)
        labels = labels.squeeze(1)
        labels_with_ignore = labels_with_ignore.squeeze(1)
        masks = masks.squeeze(1).float()
        no_chg_labels = no_chg_labels.squeeze(1)
        no_chg_labels_with_ignore = no_chg_labels_with_ignore.squeeze(1)
        no_chg_masks = no_chg_masks.squeeze(1).float()

        d_feats, nsc_feats, sc_feats = d_feats.to(device),  nsc_feats.to(device), sc_feats.to(device) # torch.Size([128, 1024, 14, 14])
        d_feats, nsc_feats, sc_feats = spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
        no_chg_labels, no_chg_labels_with_ignore, no_chg_masks = no_chg_labels.to(device), no_chg_labels_with_ignore.to(device), no_chg_masks.to(device)
        aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

        optimizer.zero_grad()

        with autocast():
            diff_bef_pos, diff_aft_pos, dirl_loss_pos, losses_pos, weighted_pos = change_detector(d_feats, sc_feats, aux_labels_pos)
            diff_bef_neg, diff_aft_neg, dirl_loss_neg, losses_neg, weighted_neg = change_detector(d_feats, nsc_feats, aux_labels_neg)

            diff_cat_pos = torch.cat([diff_bef_pos, diff_aft_pos], dim=-1)  # (B, N, 2D)
            diff_cat_neg = torch.cat([diff_bef_neg, diff_aft_neg], dim=-1)  # (B, N, 2D)

            loss_pos, att_pos, ccr_loss_pos = speaker._forward(
                diff_cat_pos, weighted_pos, labels, masks, labels_with_ignore=labels_with_ignore)
            loss_neg, att_neg, ccr_loss_neg = speaker._forward(
                diff_cat_neg, weighted_neg, no_chg_labels, no_chg_masks, labels_with_ignore=no_chg_labels_with_ignore)
            
            speaker_loss = 0.5 * loss_pos + 0.5 * loss_neg
            dirl_loss = 0.5 * dirl_loss_pos + 0.5 * dirl_loss_neg
            ccr_loss = 0.5 * ccr_loss_pos + 0.5 * ccr_loss_neg

            change_cls_loss = 0.5 * losses_pos['change_cls_loss'] + 0.5 * losses_neg['change_cls_loss']
            change_type_loss = 0.5 * losses_pos['change_type_cls_loss'] + 0.5 * losses_neg['change_type_cls_loss']
            expert_disentangle_loss = 0.5 * losses_pos['expert_disentangle_loss'] + 0.5 * losses_neg['expert_disentangle_loss']
            expert_consistency_loss = 0.5 * losses_pos['expert_consistency_loss'] + 0.5 * losses_neg['expert_consistency_loss']
            total_loss = (
                speaker_loss +
                0.03 * dirl_loss +
                0.05 * ccr_loss +
                cfg.lambda_cls * change_cls_loss +
                cfg.lambda_type * change_type_loss +
                cfg.lambda_disentangle * expert_disentangle_loss +
                cfg.lambda_consistency * expert_consistency_loss
            )
            
            # log values
            speaker_loss_val = speaker_loss.item()
            dirl_loss_val = dirl_loss.item()
            ccr_loss_val = ccr_loss.item()
            change_cls_loss_val = change_cls_loss.item()
            change_type_loss_val = change_type_loss.item()
            expert_disentangle_loss_val = expert_disentangle_loss.item()
            expert_consistency_loss_val = expert_consistency_loss.item()
            total_loss_val = total_loss.item()

            speaker_loss_avg.update(speaker_loss_val, 2 * batch_size)
            cdcr_loss_avg.update(dirl_loss_val, 2 * batch_size)
            sim_loss_avg.update(ccr_loss_val, 2 * batch_size)
            change_cls_loss_avg.update(change_cls_loss_val, 2 * batch_size)
            change_type_loss_avg.update(change_type_loss_val, 2 * batch_size)
            expert_disentangle_loss_avg.update(expert_disentangle_loss_val, 2 * batch_size)
            expert_consistency_loss_avg.update(expert_consistency_loss_val, 2 * batch_size)
            total_loss_avg.update(total_loss_val, 2 * batch_size)

            stats = {}

            stats['speaker_loss'] = speaker_loss_val
            stats['avg_speaker_loss'] = speaker_loss_avg.avg
            stats['cdcr_loss'] = dirl_loss_val
            stats['avg_cdcr_loss'] = cdcr_loss_avg.avg
            stats['sim_loss'] = ccr_loss_val
            stats['avg_sim_loss'] = sim_loss_avg.avg
            stats['change_cls_loss'] = change_cls_loss
            stats['avg_change_cls_loss'] = change_cls_loss_avg.avg
            stats['change_type_loss'] = change_type_loss_val
            stats['avg_change_type_loss'] = change_type_loss_avg.avg
            stats['expert_disentangle_loss'] = expert_disentangle_loss_val
            stats['avg_expert_disentangle_loss'] = expert_disentangle_loss_avg.avg
            stats['expert_consistency_loss'] = expert_consistency_loss_val
            stats['avg_expert_consistency_loss'] = expert_consistency_loss_avg.avg
            stats['total_loss'] = total_loss_val
            stats['avg_total_loss'] = total_loss_avg.avg

        #results, sample_logprobs = model(d_feats, q_feats, labels, cfg=cfg, mode='sample')
        # total_loss.backward()
        scaler.scale(total_loss).backward()
        if cfg.train.grad_clip != -1.0:  # enable, -1 == disable
            scaler.unscale_(optimizer)  # unscale the gradients before clipping
            nn.utils.clip_grad_norm_(change_detector.parameters(), cfg.train.grad_clip)
            nn.utils.clip_grad_norm_(speaker.parameters(), cfg.train.grad_clip)

        # optimizer.step()

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        iter_end_time = time.time() - iter_start_time

#
        t += 1
        wandb.log({"total_loss": stats['avg_total_loss'], "speaker_loss": stats['avg_speaker_loss'], "cdcr_loss": stats['avg_cdcr_loss'], 
                    "sim_loss": stats['avg_sim_loss'], "change_type_loss": stats['avg_change_type_loss'], 
                    "change_cls_loss": stats['avg_change_cls_loss'], 
                    "expert_disentangle_loss": stats['avg_expert_disentangle_loss'],
                    "expert_consistency_loss": stats['avg_expert_consistency_loss'],
                    "lr": optimizer.param_groups[0]['lr']}, step= t)

        if t % cfg.train.log_interval == 0:
            train_logger.print_current_stats(epoch, i, t, stats, iter_end_time)
            train_logger.plot_current_stats(
                epoch,
                float(i * batch_size) / train_size, stats, 'loss')

        if t % cfg.train.snapshot_interval == 0:
            speaker_state = speaker.state_dict()
            chg_det_state = change_detector.state_dict()
            checkpoint = {
                'change_detector_state': chg_det_state,
                'speaker_state': speaker_state,
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'iteration': t,
                'epoch': epoch,
                'model_cfg': cfg
            }
            save_path = os.path.join(snapshot_dir,
                                     snapshot_file_format % (exp_name, t))
            save_checkpoint(checkpoint, save_path)

            print('Running eval at iter %d' % t)
            set_mode('eval', [change_detector, speaker])
            with torch.no_grad():
                test_iter_start_time = time.time()

                idx_to_word = train_dataset.get_idx_to_word()

                if args.visualize:
                    sample_subdir_path = sample_subdir_format % (exp_name, t)
                    sample_save_dir = os.path.join(sample_dir, sample_subdir_path)
                    if not os.path.exists(sample_save_dir):
                        os.makedirs(sample_save_dir)
                sent_subdir_path = sent_subdir_format % (exp_name, t)
                sent_save_dir = os.path.join(sent_dir, sent_subdir_path)
                if not os.path.exists(sent_save_dir):
                    os.makedirs(sent_save_dir)

                result_sents_pos = {}
                result_sents_neg = {}
                for val_i, val_batch in enumerate(val_loader):
                    d_feats, nsc_feats, sc_feats, labels, labels_with_ignore, \
                    no_chg_labels, no_chg_labels_with_ignore, masks, no_chg_masks, \
                    aux_labels_pos, aux_labels_neg, d_img_paths, nsc_img_paths, sc_img_paths = val_batch

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

                    speaker_output_pos, _ = speaker.sample(diff_cat_pos, weighted_pos)
                    speaker_output_neg, _ = speaker.sample(diff_cat_neg, weighted_neg)

                    gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:]) # no start
                    gen_sents_neg = decode_sequence_transformer(idx_to_word, speaker_output_neg[:, 1:])

                    for val_j in range(speaker_output_pos.size(0)):
                        gts = decode_sequence_transformer(idx_to_word, labels[val_j][:, 1:])
                        gts_neg = decode_sequence_transformer(idx_to_word, no_chg_labels[val_j][:, 1:])

                        sent_pos = gen_sents_pos[val_j]
                        sent_neg = gen_sents_neg[val_j]

                        image_id = d_img_paths[val_j].split('_')[-1]
                        result_sents_pos[image_id] = sent_pos
                        result_sents_neg[image_id + '_n'] = sent_neg

                        message = '%s results:\n' % d_img_paths[val_j]
                        message += '\t' + sent_pos + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts:
                            message += gt + '\n'
                        message += '===================================\n'
                        message += '%s results:\n' % nsc_img_paths[val_j]
                        message += '\t' + sent_neg + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts_neg:
                            message += gt + '\n'
                        message += '===================================\n'
                        # print(message)

                test_iter_end_time = time.time() - test_iter_start_time
                result_save_path_pos = os.path.join(sent_save_dir, 'sc_results.json')
                result_save_path_neg = os.path.join(sent_save_dir, 'nsc_results.json')
                coco_gen_format_save(result_sents_pos, result_save_path_pos)
                coco_gen_format_save(result_sents_neg, result_save_path_neg)

            set_mode('train', [change_detector, speaker])
    # lr_scheduler.step()
