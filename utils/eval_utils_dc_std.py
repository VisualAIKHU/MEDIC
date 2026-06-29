import sys
COCO_PATH = '/home/seanoh/coco-caption/' # i.e. /home/user/code/coco-caption
sys.path.insert(0, COCO_PATH)

import json
import copy
import numpy as np

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
from typing import Optional
from PIL import Image
import torch
import open_clip

def coco_gt_format_save(gt_file, neg=False):
    gt = json.load(open(gt_file, 'r'))
    gt_dict = {}
    info_dict = {
        'contributor': 'dummy',
        'date_created': 'dummy',
        'description': 'dummy',
        'url': 'dummy',
        'version': 'dummy',
        'year': 'dummy'
    }

    gt_dict['info'] = info_dict
    gt_dict['licenses'] = info_dict
    gt_dict['type'] = 'captions'
    gt_dict['images'] = []
    gt_dict['annotations'] = []

    count = 0
    for k, v in gt.items():
        image_id = k + '.png'

        im = {'filename': image_id, 'id': image_id}
        gt_dict['images'].append(im)
        for c in v:
            annotation = {'caption': c, 'id': count, 'image_id': image_id}
            count += 1
            gt_dict['annotations'].append(annotation)

    json.dump(gt_dict, open(gt_file.split('.json')[0] + '_reformat.json', 'w'))

def coco_gen_format(gen_dict):
    results = []
    for k, v in gen_dict.items():
        results.append({'caption': v, 'image_id': k})
    return results

def coco_gen_format_save(gen_dict, save_path):
    results = coco_gen_format(gen_dict)
    json.dump(results, open(save_path, 'w'))

def merge_gt_files(gt_file1, save_path):
    gt1 = json.load(open(gt_file1, 'r'))
    # gt2 = json.load(open(gt_file2, 'r'))
    gt_dict = {}
    info_dict = {
        'contributor': 'dummy',
        'date_created': 'dummy',
        'description': 'dummy',
        'url': 'dummy',
        'version': 'dummy',
        'year': 'dummy'
    }

    gt_dict['info'] = info_dict
    gt_dict['licenses'] = info_dict
    gt_dict['type'] = 'captions'
    gt_dict['images'] = []
    gt_dict['annotations'] = []

    count = 0
    for k, v in gt1.items():
        image_id = k + '.png'
        im = {'filename': image_id, 'id': image_id}
        gt_dict['images'].append(im)
        for c in v:
            annotation = {'caption': c, 'id': count, 'image_id': image_id}
            count += 1
            gt_dict['annotations'].append(annotation)

    json.dump(gt_dict, open(save_path, 'w'))

def score_generation(anno_file, result_file):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)

    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()

    coco_eval.evaluate()
    return copy.deepcopy(coco_eval.eval)

def valid_caption(coco, coco_res, img_id):
    refs = coco.imgToAnns.get(img_id, [])
    hyps = coco_res.imgToAnns.get(img_id, [])

    if not refs or not any(a.get("caption", "").strip() for a in refs):
        return False
    if not hyps or not any(a.get("caption", "").strip() for a in hyps):
        return False
    return True

def score_generation_by_type(anno_file, result_file, type_file):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)
    coco_eval = COCOEvalCap(coco, coco_res)

    type_dict = json.load(open(type_file, 'r'))
    results = {}

    METRIC_NAMES = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
                    'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
    
    for type, image_ids in type_dict.items():
        filtered = set(coco_res.getImgIds()).intersection(set(image_ids))

        filtered = [img_id for img_id in filtered if valid_caption(coco, coco_res, img_id)]

        if not filtered:
            print(f"[SKIP] No valid captions for type: {type}")
            results[type] = {metric: 0.0 for metric in METRIC_NAMES}
            continue

        coco_eval.params['image_id'] = list(filtered)
        coco_eval.evaluate()
        results[type] = copy.deepcopy(coco_eval.eval)

    return results

def score_generation_with_ids(anno_file, result_file, img_ids):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)

    coco_eval = COCOEvalCap(coco, coco_res)
    filtered = set(coco_res.getImgIds()).intersection(set(img_ids))
    coco_eval.params['image_id'] = list(filtered)

    coco_eval.evaluate()
    return copy.deepcopy(coco_eval.eval)

def score_generation_by_type_with_ids(anno_file, result_file, type_file, img_ids):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)
    coco_eval = COCOEvalCap(coco, coco_res)

    type_dict = json.load(open(type_file, 'r'))
    results = {}
    for type, image_ids in type_dict.items():
        filtered = set(coco_res.getImgIds()).intersection(set(image_ids))
        filtered_twice = filtered.intersection(set(img_ids))
        coco_eval.params['image_id'] = list(filtered_twice)
        coco_eval.evaluate()
        results[type] = copy.deepcopy(coco_eval.eval)

    return results

def pointing(gen_mapping, gt_mapping, type_ids=None):
    pointings = []
    count = 0
    if type_ids:
        type_ids = set([str(int(id.split('.')[0])) for id in type_ids])
    for id, (gen_before, gen_after) in gen_mapping.items():
        if type_ids and id not in type_ids:
            continue
        gt_before, gt_after = gt_mapping[id]
        if gt_before is not None:
            gen_before_flat = gen_before.flatten()
            gt_before_flat = gt_before.flatten()
            p_before = gt_before_flat[np.argmax(gen_before_flat)]
            count += 1
        else:
            p_before = 0.0
        if gt_after is not None:
            gen_after_flat = gen_after.flatten()
            gt_after_flat = gt_after.flatten()
            p_after = gt_after_flat[np.argmax(gen_after_flat)]
            count += 1
        else:
            p_after = 0.0
        p = p_before + p_after
        pointings.append(p)
    m_pointing = sum(pointings) / float(count)
    return m_pointing

def coverage(gen_mapping, gt_mapping, type_ids=None):
    coverages = []
    if type_ids:
        type_ids = set([str(int(id.split('.')[0])) for id in type_ids])
    for id, (gen_before, gen_after) in gen_mapping.items():
        # normalize
        gen_before = gen_before / gen_before.sum()
        gen_after = gen_after / gen_after.sum()
        if type_ids and id not in type_ids:
            continue
        gt_before, gt_after = gt_mapping[id]
        if gt_before is not None:
            s_before = (gt_before * gen_before).sum()
        else:
            s_before = 0.0
        if gt_after is not None:
            s_after = (gt_after * gen_after).sum()
        else:
            s_after = 0.0
        score = (s_before + s_after) / 2.0
        coverages.append(score)
    m_coverage = np.mean(coverages)
    return m_coverage

def _ensure_png(name: str):
    return name if name.lower().endswith(".png") else name + ".png"

def _id_to_pair_paths(image_id: str, root_before: str, root_after: str):
    base = os.path.basename(image_id)
    base = _ensure_png(base.split('.')[0])  # '123' or '123.png' -> '123.png'
    p_before = os.path.join(root_before, base)
    p_after  = os.path.join(root_after,  base)
    return p_before, p_after, base  # base는 '123.png'
    
@torch.no_grad()
def clipscore_pair_concat(
    result_file: str,
    images_root_before: str,
    images_root_after: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[str] = None,
    batch_size: int = 32,
    normalize_to_100: bool = True
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    results = json.load(open(result_file, "r"))  # [{'image_id':..., 'caption':...}, ...]
    # 준비
    concat_images = []
    texts = []
    keep_ids = []

    for r in results:
        p_b, p_a, base = _id_to_pair_paths(r["image_id"], images_root_before, images_root_after)
        try:
            im_b = Image.open(p_b).convert("RGB")
            im_a = Image.open(p_a).convert("RGB")
        except Exception:
            concat_images.append(None)
            texts.append(r["caption"])
            keep_ids.append(base)
            continue

        w = im_b.width + im_a.width
        h = max(im_b.height, im_a.height)
        canvas = Image.new("RGB", (w, h))
        canvas.paste(im_b, (0, 0))
        canvas.paste(im_a, (im_b.width, 0))

        concat_images.append(canvas)
        texts.append(r["caption"])
        keep_ids.append(base)

    scores = []
    for i in range(0, len(concat_images), batch_size):
        ims = concat_images[i:i+batch_size]
        caps = texts[i:i+batch_size]
        if len(ims) == 0:
            break

        valid_idx = [j for j, im in enumerate(ims) if im is not None]
        if len(valid_idx) == 0:
            scores.extend([0.0] * len(ims))
            continue

        img_tensor = torch.stack([preprocess(ims[j]) for j in valid_idx]).to(device)
        txt_tensor = tokenizer(caps).to(device)

        img_feat = model.encode_image(img_tensor)
        txt_feat = model.encode_text(txt_tensor)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sims_valid = (img_feat * txt_feat[valid_idx]).sum(-1).tolist()

        sims_full = [0.0] * len(ims)
        for pos, j in enumerate(valid_idx):
            sims_full[j] = sims_valid[pos]

        if normalize_to_100:
            sims_full = [max(0.0, s) * 100.0 for s in sims_full]

        scores.extend(sims_full)

    mean_score = float(np.mean(scores)) if scores else 0.0
    return {"PairCLIPScore": mean_score, "per_item": scores, "image_ids": keep_ids}

@torch.no_grad()
def clipscore_pair_concat_by_type(
    result_file: str,
    type_file: str,
    images_root_before: str,
    images_root_after: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[str] = None,
    batch_size: int = 32,
    normalize_to_100: bool = True
):
    base = clipscore_pair_concat(
        result_file, images_root_before, images_root_after,
        model_name=model_name, pretrained=pretrained,
        device=device, batch_size=batch_size, normalize_to_100=normalize_to_100
    )
    id2score = {img_id: s for img_id, s in zip(base["image_ids"], base["per_item"])}
    type_dict = json.load(open(type_file, "r"))

    out = {}
    for t, ids in type_dict.items():
        ids_norm = [_ensure_png(str(x).split('.')[0]) for x in ids]
        vals = [id2score[i] for i in ids_norm if i in id2score]
        out[t] = {"PairCLIPScore": float(np.mean(vals)) if len(vals) else 0.0}
    return out

if __name__ == '__main__':
    anno_path = '/mnt/disk1/clevr_dc/change_captions_control_6_paraphrased.json'
    coco_gt_format_save(anno_path)

    save_path = '/mnt/disk1/clevr_dc/total_change_captions_reformat_control_6_paraphrased.json'
    merge_gt_files(anno_path, save_path)
