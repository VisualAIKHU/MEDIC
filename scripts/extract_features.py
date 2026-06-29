import argparse, os, json, time
import numpy as np
from imageio import imread
import cv2

import torch
import torchvision

torch.backends.cudnn.benchmark = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_dir', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=4, type=int)
parser.add_argument('--batch_size', default=128, type=int)

parser.add_argument('--no_save', action='store_true',
                    help='Skip saving .npy files to avoid disk I/O.')
parser.add_argument('--time_end_to_end', action='store_true',
                    help='Also print end-to-end time (preproc + H2D + forward).')
parser.add_argument('--warmup_iters', default=10, type=int,
                    help='Number of dummy forwards before timing to stabilize performance.')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def build_model(args):
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if 'resnet' not in args.model:
        raise ValueError('Feature extraction only supports ResNets')

    cnn = getattr(torchvision.models, args.model)(pretrained=True)

    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(args.model_stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))

    model = torch.nn.Sequential(*layers)
    model.to(device)
    model.eval()
    return model


def warmup_model(model, height, width, iters=10):
    x = torch.randn(1, 3, height, width, device=device)
    print(f"[Warmup] Running {iters} dummy forwards for {height}x{width} input...")
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    if use_cuda:
        torch.cuda.synchronize()
    print("[Warmup] Done.\n")


def run_batch(cur_batch, model, time_end_to_end=False, print_prefix=""):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std  = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).to(device)

    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    if use_cuda:
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    with torch.no_grad():
        feats = model(image_batch)
    if use_cuda:
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    batch_size = image_batch.shape[0]
    forward_ms_total = (t3 - t2) * 1000.0
    forward_ms_per_image = forward_ms_total / max(1, batch_size)

    print(f"{print_prefix} forward_ms_per_image={forward_ms_per_image:.3f} ms, "
          f"batch={batch_size}, input_shape={tuple(image_batch.shape)}")

    if time_end_to_end:
        end2end_ms_total = (t3 - t0) * 1000.0
        end2end_ms_per_image = end2end_ms_total / max(1, batch_size)
        print(f"{print_prefix} end2end_ms_per_image={end2end_ms_per_image:.3f} ms "
              f"(includes preproc & host->device)")

    feats = feats.cpu().clone().numpy()
    return feats


def main(args):
    input_paths = []
    idx_set = set()
    for fn in os.listdir(args.input_image_dir):
        if not fn.endswith('.png'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        input_paths.append((os.path.join(args.input_image_dir, fn), idx))
        idx_set.add(idx)
    input_paths.sort(key=lambda x: x[1])
    assert len(idx_set) == len(input_paths)
    if args.max_images is not None:
        input_paths = input_paths[:args.max_images]

    if len(input_paths) == 0:
        raise RuntimeError("No input images found in: %s" % args.input_image_dir)

    print("First:", input_paths[0])
    print("Last :", input_paths[-1])

    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args)

    warmup_model(model, args.image_height, args.image_width, args.warmup_iters)

    img_size = (args.image_height, args.image_width)
    i0 = 0
    cur_batch = []
    cur_path_batch = []

    for i, (path, idx) in enumerate(input_paths):
        img = imread(path, pilmode='RGB')
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
        img = img.transpose(2, 0, 1)[None]
        cur_batch.append(img)
        cur_path_batch.append(path)

        if len(cur_batch) == args.batch_size:
            feats = run_batch(cur_batch, model, time_end_to_end=args.time_end_to_end,
                              print_prefix=f"[batch {i0}-{i0+len(cur_batch)-1}]")
            for img_full_path, feat in zip(cur_path_batch, feats):
                feat = feat.squeeze()
                img_base_path = os.path.basename(img_full_path)
                output_path = os.path.join(args.output_dir, img_base_path)
                if not args.no_save:
                    np.save(output_path, feat)
            i1 = i0 + len(cur_batch)
            i0 = i1
            print(f'Processed {i1} / {len(input_paths)} images')
            cur_batch = []
            cur_path_batch = []

    if len(cur_batch) > 0:
        feats = run_batch(cur_batch, model, time_end_to_end=args.time_end_to_end,
                          print_prefix=f"[batch {i0}-{i0+len(cur_batch)-1}]")
        for img_full_path, feat in zip(cur_path_batch, feats):
            feat = feat.squeeze()
            img_base_path = os.path.basename(img_full_path)
            output_path = os.path.join(args.output_dir, img_base_path)
            if not args.no_save:
                np.save(output_path, feat)
        i1 = i0 + len(cur_batch)
        print(f'Processed {i1} / {len(input_paths)} images')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
