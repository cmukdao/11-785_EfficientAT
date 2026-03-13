import argparse
import torch
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext

import time
import json

from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.ensemble import get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels


def audio_tagging(args):
    """
    Running Inference on an audio clip.
    """
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # load model architecture
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        if model_name.startswith("dymn"):
            model = get_dymn(
                width_mult=NAME_TO_WIDTH(model_name),
                pretrained_name=None if args.checkpoint_path else model_name,
                strides=args.strides,
                num_classes=args.num_classes
            )
        else:
            model = get_mobilenet(
                width_mult=NAME_TO_WIDTH(model_name),
                pretrained_name=None if args.checkpoint_path else model_name,
                strides=args.strides,
                head_type=args.head_type,
                num_classes=args.num_classes
            )

    # load local checkpoint if provided
    if args.checkpoint_path is not None:
        ckpt = torch.load(args.checkpoint_path, map_location=device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # remove "module." prefix if needed
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned_state_dict[k[len("module."):]] = v
            else:
                cleaned_state_dict[k] = v

        model.load_state_dict(cleaned_state_dict, strict=False)

    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # reset vram stats to record memory usuage
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    start_fwd = time.perf_counter()

    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    end_fwd = time.perf_counter()

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    print("************* Acoustic Event Detected: *****************")
    for k in range(10):
        print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
            preds[sorted_indexes[k]]))
    print("********************************************************")

    results = {
        "audio_path": audio_path,
        "device": str(device),
        "autocast": bool(args.cuda),
        "spec_shape": list(spec.shape),
        "forward_ms": (end_fwd - start_fwd) * 1000,
        "peak_gpu_mem_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "top1_label": labels[sorted_indexes[0]],
        "top1_score": float(preds[sorted_indexes[0]]),
        "top10": [
            {"label": labels[i], "score": float(preds[i])}
            for i in sorted_indexes[:10]
        ]
    }

    with open(f"{model_name}_inference_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='mn10_as')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--num_classes', type=int, default=527)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--audio_path', type=str, required=True)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=[])

    args = parser.parse_args()

    audio_tagging(args)
