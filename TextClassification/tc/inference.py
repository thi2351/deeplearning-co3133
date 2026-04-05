"""Load text checkpoints: HuggingFace ``AutoModelForSequenceClassification`` or LSTM/CNN/RCNN (.pth)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from tc.config import (
    DEFAULT_CLASS_NAMES,
    DEMO_SAMPLES_PATH,
    MAX_LEN,
    MODELS_DIR,
)
from tc.custom_models import CUSTOM_TOKENIZER_ID, build_custom_model_from_state_dict, is_custom_checkpoint_stem


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except RuntimeError:
        return torch.load(path, map_location="cpu")


def load_label_names(
    num_labels: int,
    models_dir: Optional[Path] = None,
) -> List[str]:
    """Prefer ``models/label_names.json`` if present; else default list (must match training id order)."""
    root = Path(models_dir or MODELS_DIR)
    path = root / "label_names.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            if len(data) != num_labels:
                raise ValueError(
                    f"label_names.json has {len(data)} entries but checkpoint has num_labels={num_labels}"
                )
            return list(data)
    names = list(DEFAULT_CLASS_NAMES)
    if len(names) != num_labels:
        raise ValueError(
            f"Default label list length {len(names)} != num_labels {num_labels}. "
            f"Add {root / 'label_names.json'} with exactly {num_labels} strings in training label order."
        )
    return names


def infer_num_labels_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    priority_keys = (
        "classifier.out_proj.weight",
        "classifier.weight",
        "score.weight",
    )
    for key in priority_keys:
        t = state_dict.get(key)
        if t is not None and getattr(t, "dim", lambda: 0)() == 2:
            return int(t.shape[0])
    for key, t in state_dict.items():
        if key.endswith("score.weight") and t.dim() == 2:
            return int(t.shape[0])
    raise ValueError("Could not infer num_labels from state_dict (unexpected checkpoint layout).")


def checkpoint_stem_base(stem: str) -> str:
    """``roberta-base_best`` → ``roberta-base``; custom names unchanged."""
    s = stem
    if s.endswith("_best"):
        s = s[: -len("_best")]
    return s


def checkpoint_stem_to_pretrained_id(stem: str) -> str:
    """HF id for loading, or ``bert-base-uncased+LSTM``-style label for custom CNN/LSTM/RCNN."""
    pid = checkpoint_stem_base(stem)
    if is_custom_checkpoint_stem(pid):
        return f"{CUSTOM_TOKENIZER_ID}+{pid.upper()}"
    return pid


def is_likely_hf_pretrained_id(stem: str) -> bool:
    """Reject custom stems (LSTM, CNN, RCNN) and obvious non-HF names."""
    pid = checkpoint_stem_base(stem)
    if is_custom_checkpoint_stem(pid):
        return False
    if not pid or len(pid) > 80:
        return False
    if pid.upper() == pid and "_" not in pid and "-" not in pid:
        return False
    if re.fullmatch(r"[A-Za-z0-9_.\-]+", pid) is None:
        return False
    return True


def is_allowed_text_checkpoint_stem(stem: str) -> bool:
    return is_likely_hf_pretrained_id(stem) or is_custom_checkpoint_stem(checkpoint_stem_base(stem))


def load_model_and_tokenizer(
    checkpoint_path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
    class_names: Optional[Sequence[str]] = None,
    state_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.nn.Module, Any, torch.device, List[str]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    stem = checkpoint_path.stem
    if not is_likely_hf_pretrained_id(stem):
        raise ValueError(
            f"Checkpoint stem {stem!r} does not look like a HuggingFace model id. "
            "Use load_text_checkpoint_bundle() for LSTM/CNN/RCNN .pth files."
        )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pretrained_id = checkpoint_stem_base(stem)
    if state_dict is None:
        state_dict = _load_torch(checkpoint_path)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must be a state_dict (dict of tensors).")

    num_labels = infer_num_labels_from_state_dict(state_dict)
    names = list(class_names) if class_names is not None else load_label_names(num_labels)
    if len(names) != num_labels:
        raise ValueError(f"class_names length {len(names)} != num_labels {num_labels}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_id,
        num_labels=num_labels,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device, names


def load_text_checkpoint_bundle(
    checkpoint_path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[torch.nn.Module, Any, torch.device, List[str], bool]:
    """Load either HF sequence-classification .pth or notebook LSTM/CNN/RCNN weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    base = checkpoint_stem_base(checkpoint_path.stem)
    sd = _load_torch(checkpoint_path)
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint must be a state_dict (dict of tensors).")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_custom_checkpoint_stem(base):
        from transformers import AutoTokenizer

        model = build_custom_model_from_state_dict(base, sd)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_ID)
        num_labels = int(sd["fc.weight"].shape[0])
        names = list(class_names) if class_names is not None else load_label_names(num_labels)
        if len(names) != num_labels:
            raise ValueError(f"class_names length {len(names)} != num_labels {num_labels}")
        return model, tokenizer, device, names, True

    model, tokenizer, device, names = load_model_and_tokenizer(
        checkpoint_path,
        device=device,
        class_names=class_names,
        state_dict=sd,
    )
    return model, tokenizer, device, names, False


@torch.inference_mode()
def predict_text(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    class_names: Sequence[str],
    topk: int = 5,
    max_len: int = MAX_LEN,
    *,
    is_custom: bool = False,
) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text")

    if is_custom:
        enc = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        logits = model(input_ids)
        logits = logits[0] if logits.dim() == 2 else logits
    else:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[0]
    probs = logits.float().softmax(dim=0)
    k = max(1, min(int(topk), len(class_names)))
    topk_prob, topk_idx = probs.topk(k)
    top1_i = int(topk_idx[0].item())

    return {
        "top1_id": top1_i,
        "top1_name": class_names[top1_i],
        "top1_prob": float(topk_prob[0].item()),
        "topk": [
            {"id": int(i), "name": class_names[int(i)], "prob": float(p)}
            for i, p in zip(topk_idx.tolist(), topk_prob.tolist())
        ],
    }


def load_demo_samples(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = Path(path or DEMO_SAMPLES_PATH)
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and x.get("text")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one text through a saved transformer .pth")
    parser.add_argument("checkpoint", type=Path, help=".pth file (stem = HuggingFace model id, e.g. roberta-base.pth)")
    parser.add_argument("text", nargs="?", default="", help="Single line of text (or read stdin if empty)")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    import sys

    line = args.text.strip() or sys.stdin.read().strip()
    if not line:
        parser.error("Provide text as argv or stdin.")

    model, tokenizer, device, names, is_custom = load_text_checkpoint_bundle(args.checkpoint)
    out = predict_text(
        model, tokenizer, line, device, names, topk=args.topk, is_custom=is_custom
    )
    print(f"Top-1: {out['top1_name']} ({out['top1_prob']:.2%})")
    for row in out["topk"]:
        print(f"  {row['prob']:.2%}  [{row['id']}] {row['name']}")


if __name__ == "__main__":
    main()
