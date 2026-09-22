"""Check that the logged GPT-5 nano calls used the prompt text now in jevbench.

The final runs were made from an uncommitted tree, so the git SHA in their
manifests does not identify the code. This recomputes each call's prompt token
count from the current prompt text and the frozen post, and compares it with
the count the provider billed. Every call should differ by the same small
chat-format overhead; any other difference means the prompt changed.

Needs `pip install tiktoken` (o200k_base, the GPT-5 tokenizer) and the frozen
sample at data/final-ucb-2025.jsonl.

    python scripts/check_prompt_tokens.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jevbench.client import production_verdict_prompt, standard_choice_prompt  # noqa: E402
from jevbench.data import load  # noqa: E402
from jevbench.questions import monolithic  # noqa: E402

LOGS = {"runs/final-nano-minimal-2025.jsonl": production_verdict_prompt,
        "runs/final-nano-low-2025.jsonl": production_verdict_prompt,
        "runs/diagnostic/standard-choice-api-2025.jsonl": standard_choice_prompt,
        "runs/diagnostic/standard-choice-api-tolerant-2025.jsonl": standard_choice_prompt}

enc = tiktoken.get_encoding("o200k_base")
posts = {i.id: i for i in load(Path("data/final-ucb-2025.jsonl"))}
verdict = monolithic()["verdict"]
ok = True
for path, prompt in LOGS.items():
    system = len(enc.encode(prompt(verdict)))
    gaps = Counter(r["prompt_tokens"] - system
                   - len(enc.encode("Title and post:\n" + posts[r["item_id"]].body))
                   for r in map(json.loads, Path(path).read_text().splitlines()))
    ok &= len(gaps) == 1
    print(f"{path}: overhead {dict(gaps)}")
print("prompts match the logs" if ok else "MISMATCH: a prompt differs from what was sent")
sys.exit(0 if ok else 1)
