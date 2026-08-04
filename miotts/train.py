"""LoRA fine-tuning for Indic-Mio (Qwen3-0.6B speech-LM backbone).

Ground truth for this recipe:
  - Indic-Mio's own HF discussion (SPRINGLab/Indic-Mio, thread #1): "Use the speech
    codec to convert your audio data into speech codes. Map those codes to the token
    format and then you can do simple SFT training just like for text."
  - config.json: architectures=["Qwen3ForCausalLM"], no custom modeling code,
    vocab_size=164480, tie_word_embeddings=true.
  - Speech tokens occupy ids [speech_token_offset, speech_token_offset + speech_vocab_size)
    i.e. 151669..164468 inclusive, one FSQ-quantized token per 25Hz codec frame.
  - chat_template.jinja is plain ChatML: "<|im_start|>{role}\n{content}<|im_end|>\n".
  - Emotion/style tags (<happy>, <whisper>, ...) are plain text appended to the
    sentence, not special tokens -- just include them in the `text` field.

Data format: a JSONL manifest, one example per line:
    {"text": "Hello, how are you today?", "audio_path": "data/wavs/0001.wav"}
    {"text": "नमस्ते, आप कैसे हैं?", "audio_path": "data/wavs/0002.wav"}

`text` should already include any emotion tag the clip expresses (e.g. "... <happy>").
`audio_path` is resolved relative to the manifest file's directory.

The global (speaker) embedding produced by the codec is only consumed by the codec's
decoder at inference time -- it never enters the LM's vocabulary, so it plays no part
in LM fine-tuning and is not computed here.

Two-stage pipeline:
  1. `prepare`: run MioCodec over each manifest entry, cache content_token_indices to
     disk (one .pt per example) so re-running training doesn't re-encode audio.
  2. `train`: build ChatML sequences (prompt = user turn, target = assistant turn's
     mapped speech tokens), mask loss on everything except the assistant turn, and run
     a standard HF Trainer + LoRA SFT loop.

Usage:
    python -m miotts.train prepare --manifest data/train.jsonl --cache-dir data/cache
    python -m miotts.train train --manifest data/train.jsonl --cache-dir data/cache \
        --output-dir runs/lora-v1
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import MioConfig

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Some Qwen3 (QK-norm) + LoRA combinations report shape/CUBLAS errors when q_proj/k_proj
# are adapted; fall back to this set with --no-lora-qk if that happens.
SAFE_LORA_TARGET_MODULES = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Example:
    text: str
    audio_path: Path


def load_manifest(manifest_path: Path) -> list[Example]:
    base_dir = manifest_path.parent
    examples = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = base_dir / audio_path
            examples.append(Example(text=row["text"], audio_path=audio_path))
    return examples


def cache_path_for(cache_dir: Path, example: Example, index: int) -> Path:
    return cache_dir / f"{index:07d}_{example.audio_path.stem}.pt"


def _load_audio_from_bytes(audio_bytes: bytes, sample_rate: int) -> torch.Tensor:
    """In-memory equivalent of miocodec.util.load_audio (path -> waveform).

    Decodes straight from a bytes buffer so streamed HF audio never touches disk.
    """
    import io

    import soundfile as sf
    import torchaudio

    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
        frames = f.read(dtype="float32", always_2d=True)
        waveform = torch.from_numpy(frames.T)
        sr = f.samplerate

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    max_val = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / max_val
    return waveform.squeeze(0)


def cmd_prepare_hf(args):
    """Stream an HF audio dataset, encode each clip with the codec, and cache only
    the resulting content_token_indices tensor (a few KB) -- raw audio bytes are
    decoded in-memory and discarded, never written to disk.

    For a gated dataset repo (e.g. Shubhangi7/marathi-tts-elevenlabs), use --dataset
    with the repo id and set HF_TOKEN in the environment before running -- the token
    is only ever read from the environment, never hardcoded or logged here.
    """
    import os

    from datasets import Audio, load_dataset
    from miocodec import MioCodecModel

    if bool(args.dataset) == bool(args.data_files):
        raise ValueError("Pass exactly one of --dataset or --data-files")

    cfg = MioConfig()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    codec = MioCodecModel.from_pretrained(cfg.codec_name).eval().to(device)

    if args.dataset:
        token = os.environ.get("HF_TOKEN")
        ds = load_dataset(args.dataset, streaming=True, split="train", token=token)
    else:
        ds = load_dataset(
            args.format,
            data_files=args.data_files,
            streaming=True,
            split="train",
        )
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))
    if args.skip:
        ds = ds.skip(args.skip)
    if args.limit:
        ds = ds.take(args.limit)

    id_prefix = args.id_prefix
    seen = 0
    encoded = 0
    skipped = 0
    for row in ds:
        seen += 1
        if "audio_id" in row:
            audio_id = row["audio_id"]
        elif "utterance_id" in row:
            audio_id = row["utterance_id"]
        else:
            audio_path = Path(row["audio"]["path"]).stem if row["audio"].get("path") else f"row{seen:07d}"
            audio_id = f"{id_prefix}_{audio_path}" if id_prefix else audio_path
        out_path = cache_dir / f"{audio_id}.pt"
        if out_path.exists() and not args.overwrite:
            continue

        audio_bytes = row["audio"]["bytes"]
        if not audio_bytes:
            print(f"[{audio_id}] no audio bytes, skipping")
            skipped += 1
            continue

        try:
            waveform = _load_audio_from_bytes(audio_bytes, sample_rate=codec.config.sample_rate)
        except Exception as e:
            print(f"[{audio_id}] failed to decode audio, skipping: {e}")
            skipped += 1
            continue

        waveform = waveform.to(device)
        with torch.no_grad():
            features = codec.encode(waveform, return_content=True, return_global=False)

        content_token_indices = features.content_token_indices.detach().cpu()
        if content_token_indices.numel() == 0:
            print(f"[{audio_id}] codec produced zero tokens, skipping")
            skipped += 1
            continue

        torch.save(
            {"text": row["text"], "content_token_indices": content_token_indices},
            out_path,
        )
        encoded += 1
        if encoded % 50 == 0:
            print(f"  encoded {encoded} (seen {seen}, skipped {skipped}) -> {cache_dir}")

    print(f"Done. {encoded}/{seen} cached to {cache_dir} ({skipped} skipped)")


def cmd_prepare(args):
    """Encode every manifest audio file to codec content tokens and cache to disk."""
    from miocodec import MioCodecModel
    from miocodec.util import load_audio

    cfg = MioConfig()
    manifest_path = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    examples = load_manifest(manifest_path)
    print(f"Loaded {len(examples)} examples from {manifest_path}")

    device = args.device
    codec = MioCodecModel.from_pretrained(cfg.codec_name).eval().to(device)

    skipped = 0
    for i, example in enumerate(examples):
        out_path = cache_path_for(cache_dir, example, i)
        if out_path.exists() and not args.overwrite:
            continue
        if not example.audio_path.exists():
            print(f"[{i}] missing audio, skipping: {example.audio_path}")
            skipped += 1
            continue

        waveform = load_audio(str(example.audio_path), sample_rate=codec.config.sample_rate)
        waveform = waveform.to(device)
        with torch.no_grad():
            features = codec.encode(waveform, return_content=True, return_global=False)

        content_token_indices = features.content_token_indices.detach().cpu()
        if content_token_indices.numel() == 0:
            print(f"[{i}] codec produced zero tokens, skipping: {example.audio_path}")
            skipped += 1
            continue

        torch.save(
            {"text": example.text, "content_token_indices": content_token_indices},
            out_path,
        )
        if (i + 1) % 50 == 0:
            print(f"  encoded {i + 1}/{len(examples)}")

    print(f"Done. {len(examples) - skipped}/{len(examples)} cached to {cache_dir}")


class SpeechSFTDataset(Dataset):
    """Builds ChatML input_ids/labels from cached (text, content_token_indices) pairs.

    Loss is masked (-100) over the prompt (system template + user turn + assistant
    header) and computed only over the assistant turn's speech tokens + closing
    <|im_end|>, matching standard instruction-tuning SFT masking.
    """

    def __init__(self, cache_dirs: Path | list[Path], tokenizer, config: MioConfig, max_length: int = 1024):
        if isinstance(cache_dirs, (str, Path)):
            cache_dirs = [cache_dirs]
        self.paths = sorted(p for d in cache_dirs for p in Path(d).glob("*.pt"))
        if not self.paths:
            raise ValueError(f"No cached examples found in {cache_dirs}; run `prepare` first.")
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length

    def __len__(self):
        return len(self.paths)

    def _speech_token_ids(self, content_token_indices: torch.Tensor) -> list[int]:
        cfg = self.config
        codes = content_token_indices.tolist()
        for code in codes:
            if not (0 <= code < cfg.speech_vocab_size):
                raise ValueError(f"codec code {code} out of range [0, {cfg.speech_vocab_size})")
        return [cfg.speech_token_offset + code for code in codes]

    def __getitem__(self, idx):
        record = torch.load(self.paths[idx], map_location="cpu", weights_only=True)
        tok = self.tokenizer

        prompt = tok.apply_chat_template(
            [{"role": "user", "content": record["text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

        speech_ids = self._speech_token_ids(record["content_token_indices"])
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        target_ids = speech_ids + [im_end_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


def collate(batch, pad_token_id: int):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attention_mask = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
        attention_mask.append(ex["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class PushCheckpointToHubCallback:
    """Pushes each saved trainer checkpoint to a HF model repo, then deletes the
    local copy so a long run doesn't accumulate checkpoints on disk.
    """

    def __init__(self, repo_id: str, tokenizer, sweep_out_dir: str | None = None):
        from transformers import TrainerCallback

        self.repo_id = repo_id
        self.tokenizer = tokenizer
        self.sweep_out_dir = sweep_out_dir
        self._callback_cls = TrainerCallback

    def as_callback(self):
        import shutil
        import subprocess
        import sys

        from huggingface_hub import HfApi

        repo_id = self.repo_id
        tokenizer = self.tokenizer
        sweep_out_dir = self.sweep_out_dir
        pending_sweeps = []  # [(ckpt_dir, Popen, log_path, step, attempt), ...]

        def _reap_finished_sweeps(block=False):
            still_pending = []
            for ckpt_dir, proc, log_path, step, attempt in pending_sweeps:
                if block:
                    proc.wait()
                returncode = proc.poll()
                if returncode is None:
                    still_pending.append((ckpt_dir, proc, log_path, step, attempt))
                    continue
                if returncode == 0:
                    print(f"simran sweep done for step {step}")
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                elif attempt < 1:
                    print(f"simran_sweep attempt {attempt + 1} failed for step {step}, retrying")
                    still_pending.append((ckpt_dir, _launch_sweep(ckpt_dir, step), log_path, step, attempt + 1))
                else:
                    stderr = Path(log_path).read_text()[-2000:] if Path(log_path).exists() else ""
                    print(f"simran_sweep gave up for step {step} after 2 attempts:\n{stderr}")
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
            pending_sweeps[:] = still_pending

        def _launch_sweep(ckpt_dir, step):
            out_dir = f"{sweep_out_dir}_step{step}"
            log_path = f"{ckpt_dir}.sweep.log"
            log_file = open(log_path, "w")
            print(f"Launching simran sweep for step {step} -> {out_dir} (background) ...")
            return subprocess.Popen(
                [sys.executable, "-m", "miotts.simran_sweep", "--model-path", str(ckpt_dir), "--out-dir", out_dir],
                stdout=log_file,
                stderr=log_file,
            )

        class _Callback(self._callback_cls):
            def on_save(self, args, state, control, **kwargs):
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if not ckpt_dir.exists():
                    return control
                tokenizer.save_pretrained(ckpt_dir)

                api = HfApi()
                print(f"Pushing step {state.global_step} checkpoint to {repo_id} ...")
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="model",
                    folder_path=str(ckpt_dir),
                    path_in_repo=f"checkpoint-{state.global_step}",
                    commit_message=f"checkpoint at step {state.global_step}",
                )
                print(f"Pushed {ckpt_dir} to hub")

                if sweep_out_dir is not None:
                    proc = _launch_sweep(ckpt_dir, state.global_step)
                    pending_sweeps.append((ckpt_dir, proc, f"{ckpt_dir}.sweep.log", state.global_step, 0))
                else:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)

                _reap_finished_sweeps(block=False)
                return control

            def on_train_end(self, args, state, control, **kwargs):
                print("Waiting for any in-flight simran sweeps to finish...")
                _reap_finished_sweeps(block=True)
                return control

        return _Callback()


def cmd_train(args):
    from functools import partial

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    cfg = MioConfig()
    init_model = args.init_model or cfg.model_name
    init_kwargs = {"subfolder": args.init_subfolder} if args.init_subfolder else {}
    tokenizer = AutoTokenizer.from_pretrained(init_model, **init_kwargs)
    model = AutoModelForCausalLM.from_pretrained(init_model, torch_dtype=torch.bfloat16, **init_kwargs)
    model.config.use_cache = False

    if args.full_finetune:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tune: {n_trainable:,} trainable params")
    else:
        target_modules = SAFE_LORA_TARGET_MODULES if args.no_lora_qk else DEFAULT_LORA_TARGET_MODULES
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = SpeechSFTDataset([Path(d) for d in args.cache_dir], tokenizer, cfg, max_length=args.max_length)
    print(f"Training on {len(dataset)} cached examples")

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2e-5 if args.full_finetune else 2e-4

    if args.push_every_steps:
        save_strategy_kwargs = {"save_strategy": "steps", "save_steps": args.push_every_steps}
    else:
        save_strategy_kwargs = {"save_strategy": "epoch"}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_total_limit=1,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        **save_strategy_kwargs,
    )

    callbacks = []
    if args.push_to_hub_repo:
        callbacks.append(
            PushCheckpointToHubCallback(args.push_to_hub_repo, tokenizer, args.sweep_out_dir).as_callback()
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=partial(collate, pad_token_id=tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    kind = "full fine-tuned model" if args.full_finetune else "LoRA adapter"
    print(f"Saved {kind} to {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune Indic-Mio with LoRA")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Encode audio to cached codec tokens")
    prep.add_argument("--manifest", required=True, help="Path to JSONL manifest")
    prep.add_argument("--cache-dir", required=True, help="Directory to write cached .pt files")
    prep.add_argument("--device", default="cuda")
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(func=cmd_prepare)

    prep_hf = subparsers.add_parser(
        "prepare-hf",
        help="Stream an HF parquet audio dataset and cache codec tokens (no raw audio to disk)",
    )
    prep_hf.add_argument(
        "--dataset",
        default=None,
        help="HF dataset repo id, e.g. 'Shubhangi7/marathi-tts-elevenlabs' -- loads via "
        "load_dataset(repo_id, token=$HF_TOKEN). Use this for gated repos instead of "
        "--data-files. Set HF_TOKEN in the environment first if the repo is gated.",
    )
    prep_hf.add_argument(
        "--data-files",
        default=None,
        help="HF data_files glob, e.g. "
        "'hf://datasets/<org>/<name>/data/data_train_00000_of_00870-*.parquet' "
        "or a list of shard globs. Mutually exclusive with --dataset.",
    )
    prep_hf.add_argument("--cache-dir", required=True, help="Directory to write cached .pt files")
    prep_hf.add_argument("--device", default="cuda")
    prep_hf.add_argument("--overwrite", action="store_true")
    prep_hf.add_argument("--skip", type=int, default=0, help="Skip the first N examples in the stream")
    prep_hf.add_argument("--limit", type=int, default=0, help="Stop after N examples (0 = no limit)")
    prep_hf.add_argument(
        "--format",
        default="parquet",
        help="load_dataset builder name for --data-files, e.g. 'parquet' or 'arrow'",
    )
    prep_hf.add_argument(
        "--id-prefix",
        default=None,
        help="Prefix for synthesized cache ids when the dataset has no audio_id/utterance_id "
        "column (e.g. a per-language tag, to avoid filename collisions across datasets sharing "
        "the same cache dir)",
    )
    prep_hf.set_defaults(func=cmd_prepare_hf)

    train = subparsers.add_parser("train", help="Run LoRA SFT over cached examples")
    train.add_argument("--manifest", required=True, help="Path to JSONL manifest (for reference)")
    train.add_argument(
        "--cache-dir", required=True, nargs="+",
        help="One or more directories of cached .pt files (merged into a single dataset)",
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=float, default=3.0)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Defaults to 2e-4 for LoRA, 2e-5 for --full-finetune",
    )
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--lora-r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument(
        "--no-lora-qk",
        action="store_true",
        help="Drop q_proj/k_proj from LoRA targets (workaround for Qwen3 QK-norm + LoRA issues)",
    )
    train.add_argument(
        "--full-finetune",
        action="store_true",
        help="Update all model weights instead of a LoRA adapter",
    )
    train.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a trainer checkpoint dir to resume from",
    )
    train.add_argument(
        "--init-model",
        default=None,
        help="HF repo id or local path to initialize weights from, instead of MioConfig's base model "
        "(e.g. MeghanaKap/miomio_cp1_public to continue fine-tuning from a prior run)",
    )
    train.add_argument(
        "--init-subfolder",
        default=None,
        help="Subfolder within --init-model's repo to load (e.g. checkpoint-2740)",
    )
    train.add_argument(
        "--push-to-hub-repo",
        default=None,
        help="HF model repo id to push each checkpoint to (e.g. MeghanaKap/miomio_cp1_public)",
    )
    train.add_argument(
        "--push-every-steps",
        type=int,
        default=0,
        help="Save+push a checkpoint every N steps instead of once per epoch (requires --push-to-hub-repo)",
    )
    train.add_argument(
        "--sweep-out-dir",
        default=None,
        help="If set, run the simran sweep against each checkpoint before pushing, writing to <this>_step<N>",
    )
    train.set_defaults(func=cmd_train)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
