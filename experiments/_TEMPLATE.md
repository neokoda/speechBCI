# Experiment: <name>

**Purpose:** <one line — what question does this run answer?>

**Date:** YYYY-MM-DD
**Git SHA:** <commit hash when run>

## Command

```bash
<exact command used, including `ulimit -s unlimited` prefix if applicable>
```

## Configuration

- **Checkpoint:** `<path>`
- **LM dir:** `<path>`
- **Rescore LM:** `<none | gpt2 | gemma3-270m | llama2-7b | ...>`
- **Hyperparameters:** `beam=`, `nbest=`, `acoustic_scale=`, `alpha=`, `beta=`, `gamma=`, `blank_penalty=`
- **Eval split:** `<24sess test | 19sess test | subset N>`

## Results

| Metric | Value |
|---|---|
| Greedy PER | |
| WFST-only WER / CER | |
| Rescore WER / CER | |
| Oracle WER (n-best) | |

## Notes

<anything surprising, follow-ups, known caveats>
