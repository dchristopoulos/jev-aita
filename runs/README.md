# Run logs

Each `.jsonl` file has one record per API request: post ID, flair, parsed
answer, latency, tokens and billed cost. No post text. Later logs also keep
the raw typed answer in `raw_output`.
A `.meta.json` manifest sits next to each complete run.

## Original final runs

| Log | Models and arms | Posts | Code SHA-256 |
|---|---|---:|---|
| `final-jev-2025` | Jev: direct and two questions | 770 | `6bb631e4…` |
| `final-sonnet-2025` | Sonnet 5 | 770 | `6bb631e4…` |
| `final-nano-minimal-2025` | GPT-5 nano, minimal effort | 770 | `6bb631e4…` |
| `final-nano-low-2025` | GPT-5 nano, low effort | 770 | `28bcf4f4…` |
| `final-local-2025` | Qwen 3.6 and Gemma 4, local | 770 each | `28bcf4f4…` |

The last two were restarted after a parser change; see the deviations in
[`docs/FINAL_PROTOCOL.md`](../docs/FINAL_PROTOCOL.md). The manifests of
these runs need three caveats, because the runner that wrote them had gaps
since fixed:

- `started` is the time the run *finished*. The first record's `ts` is the
  real start.
- `git_sha` is `e473a91`, but the tree had uncommitted changes, so it does not
  identify the code. `code_sha256` does, but that code is not published;
  [`scripts/check_prompt_tokens.py`](../scripts/check_prompt_tokens.py) shows
  that the chat prompt in the current code is the one these runs sent.
- They have no `prompts`, `git_dirty` or `finished` fields. The runner adds
  those now.

`final-nano-low-2025` and `final-local-2025` were named
`*-normalized-2025` when they ran; they were renamed once the interrupted
logs moved to `diagnostic/`. Contents are unchanged.

## Later five-question run

`followup-yesno5-2025.jsonl` has one five-yes/no Jev call for each of the
same 770 posts. Its manifest records a clean run at `4958c82`, 770 successes,
zero failures, and $0.03362 billed. The fitted comparison in
[`docs/FIVE_QUESTION_FOLLOWUP.md`](../docs/FIVE_QUESTION_FOLLOWUP.md) uses this
log and the 300-post development logs below. It is separate from the original
final runs.

## `diagnostic/`: development and checks

| Log | What it is |
|---|---|
| `aborted-local-2025.jsonl` | Qwen, stopped after 184 of 770 posts, 6 parse failures under the old strict parser |
| `aborted-nano-low-2025.jsonl` | GPT-5 nano low, stopped after 104 of 770 posts, 1 parse failure |
| `standard-choice-api-2025.jsonl`, `-tolerant-` | Label-only prompt, GPT-5 nano minimal only, first 55 and 83 posts; stopped when I lowered the spending limit |
| `pilot-*` | 40-post pipeline check on posts from the old 2023 dataset (not the 200-post development sample), and an 8-post check of the two-question arm on the development sample |
| `steps-dev-*.jsonl`, `steps-summary.md` | Five Jev setups (direct, 5 yes/no, 2 severity scores, 5 mixed steps, 5 steps + verdict) on 300 unused 2023 posts (100 + 200); used in Part 1 and to fit the later five-question comparison |

## `dev-2023/`: development history

Runs on the earlier 2023 sample (`OsamaBsher/AITA-Reddit-Dataset`) and the
separate Civil Comments calibration test. Their chat prompt omitted the label
definitions Jev received, so they are not a fair comparison. The code that
produced them is at git tag `dev-2023`.
