import math
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch

from litgpt import Tokenizer
from litgpt.config import Config
from litgpt.model import GPT

ORC_REL_MARKERS = {"that", "who"}


def load_checkpoint(ckpt_file: Path, device: torch.device, max_seq_length: int) -> GPT:
    model = GPT(Config.from_checkpoint(ckpt_file.parent))
    raw = torch.load(ckpt_file, map_location=device)
    state = raw.get("model") if isinstance(raw, dict) and isinstance(raw.get("model"), dict) else raw
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_file}")
    model.load_state_dict(state, strict=True)
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    return model.to(device).eval()


def resolve_checkpoint_file(checkpoint_input: Path) -> Path:
    if checkpoint_input.is_file():
        if checkpoint_input.name != "lit_model.pth":
            raise ValueError(f"Expected 'lit_model.pth', got {checkpoint_input.name}")
        return checkpoint_input
    if checkpoint_input.is_dir():
        ckpt = checkpoint_input / "lit_model.pth"
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"No lit_model.pth found in {checkpoint_input}")
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_input}")


def step_from_checkpoint(ckpt_file: Path) -> int:
    name = ckpt_file.parent.name if ckpt_file.name == "lit_model.pth" else ckpt_file.name
    return int(name.split("-")[1])


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_word(word: str) -> str:
    return word.strip(".,?!;:\"'()[]{}").lower()


def _word_token_spans(sentence: str, tokenizer: Tokenizer) -> Tuple[List[int], List[Tuple[str, int, int]]]:
    """Tokenize sentence without BOS and return (token_ids, [(word, start, end), ...])."""
    tok, pos = [], 0
    spans = []
    for i, word in enumerate(sentence.split()):
        prefix = "" if i == 0 else " "
        ids = tokenizer.encode(prefix + word, bos=False, eos=False).tolist()
        spans.append((word, pos, pos + len(ids)))
        tok.extend(ids)
        pos += len(ids)
    return tok, spans


def lexical_mass(probs: torch.Tensor, token_lists: List[List[int]]) -> float:
    """Sum the mean probability over each token-id list."""
    total = 0.0
    for ids in token_lists:
        if ids:
            total += probs[torch.tensor(ids, device=probs.device)].mean().item()
    return total


def extract_orc_moved_np(sentence: str, noun_forms_orc: Set[str]) -> Tuple[List[str], Optional[str]]:
    """Extract the moved NP and its head noun from an ORC sentence."""
    noun_candidates = {
        normalize_word(w) for w in noun_forms_orc if normalize_word(w) not in {"the", "a", "an"}
    }
    words = sentence.split()
    if not words or normalize_word(words[0]) != "the":
        return [], None

    head_idx = None
    for i in range(1, len(words)):
        w = normalize_word(words[i])
        if w in ORC_REL_MARKERS or w == "the":
            break
        if w in noun_candidates:
            head_idx = i
            break

    if head_idx is None:
        return [], None

    np_words = [normalize_word(w) for w in words[: head_idx + 1] if normalize_word(w)]
    return np_words, (np_words[-1] if np_words else None)


def np_chain_logprob(
    model: GPT,
    x_prefix: torch.Tensor,
    token_ids: List[int],
    device: torch.device,
    first_step_lsp: Optional[torch.Tensor] = None,
) -> float:
    """
    Sum of autoregressive log-probs: sum_i log P(token_ids[i] | x_prefix + token_ids[:i]).
    If first_step_lsp is provided, reuses it for the first token to save a forward pass.
    """
    if not token_ids:
        return 0.0
    log_prob = 0.0
    x = x_prefix.clone()
    for i, tok_id in enumerate(token_ids):
        if i == 0 and first_step_lsp is not None:
            log_prob += first_step_lsp[tok_id].item()
        else:
            with torch.no_grad():
                logits = model(x)[0, -1, :]
            log_prob += torch.log_softmax(logits, dim=-1)[tok_id].item()
        x = torch.cat([x, torch.tensor([[tok_id]], device=device)], dim=1)
    return log_prob
