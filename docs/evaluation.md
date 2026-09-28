# Generation, scoring, and reporting

Prepare benchmark prompts with `prepare --inference`. This retains answers for scoring metadata but uses only the native question prefix to generate on the full 1024-token canvas. A training-length cap is never applied to inference prompts.

```bash
lfrm generate --config "$recipe" --data "$artifacts/test" \
  --checkpoint "$artifacts/joint-weights.pt" --embedding "$teacher" \
  --tokenizer "$teacher" --elf-selector .9999 --prompt-selector .9999 \
  --powers 2.5 2 1.5 --steps 32 --sccfg 2 --seed 42 \
  --output "$artifacts/evaluation/seed42"
lfrm score --config "$recipe" --data "$artifacts/test" \
  --predictions "$artifacts/evaluation/seed42/predictions.jsonl" \
  --workers 8 --output "$artifacts/evaluation/seed42/scoring"
```

Use a distinct directory per seed/setting. Generation checkpoints the generator state and completed predictions after each batch. Repeating the same command resumes; changing the checkpoint, selectors, seed, batch size, clocks, or guidance rejects reuse. Results record the artifact hashes, independent selectors, full coverage, per-batch time, denoiser-call count, and one terminal decoder call. Decoding is greedy. A requested ODE step corresponds to one backbone call at CFG=1; terminal decoding adds a separate backbone call.

Scoring also records an input contract and rejects reuse after predictions, dataset, or supplied benchmark cache change. Official EvalPlus reports retain the complete base/plus result file and separate base/plus accuracy with exact problem coverage.

Generation uses FP32 ELF/prompt computation and FP64 averaging of local time embeddings. Live teacher extraction uses BF16 Qwen and FP32 projection to BF16 targets. These are separate arithmetic choices. SCCFG is an input to the learned guidance embedder, not an extra unconditional forward; SCCFG=1 still uses recurrent self-conditioning. Optional CFG ≠1 adds unconditional model calls. All supplied configurations use CFG=1.

Pre-NFT defaults are ELF/prompt .9999/.9999. NFT math defaults are .99/.99; OCI defaults are .9/.9 at 128 steps and SCCFG3. Select ELF and prompt EMAs independently. A selected-weight export loads as `raw/raw`.

Flow checkpoints can use original teacher prompts with `--prompt-source teacher --features live --teacher ... --representation ...`; they do not require a learned prompt encoder. The feature provider still extracts question-only activations at evaluation.

## Scorers

- GSM8K: the canonical numeric extractor, including boxed answers, answer markers, commas, and fractions.
- MATH: Math-Verify 0.9.0 with boxed-first LaTeX and expression extraction, strict equivalence, 6-digit rounding, and a 5-second timeout. Put the reference final answer in `metadata.gold_answer`. Reward/scoring parse failures in a gold answer are errors.
- OCI training: independently sandboxed native tests after pinned EvalPlus sanitization. Reference validation is mandatory before NFT. Tests run in fresh processes with time/memory limits; infrastructure errors propagate.
- HumanEval/MBPP: the pinned official EvalPlus implementation, retaining base and plus results. This uses official benchmark data, not the OCI training tests.

For official code benchmarks, first populate an EvalPlus cache in a trusted network-enabled environment (using its official data download functions). The scorer mounts that cache read-only and runs with networking disabled:

```bash
lfrm score --config configs/oci_l.yaml --data "$artifacts/humaneval" \
  --predictions "$artifacts/evaluation/seed42/predictions.jsonl" \
  --benchmark humaneval --evalplus-cache /path/to/evalplus-cache \
  --workers 16 --output "$artifacts/evaluation/seed42/evalplus"
```

Use `--benchmark mbpp` for MBPP+. The cache directory must be the contents expected at `$HOME/.cache/evalplus`. No entry-point aliases or repair heuristics are added. Native-test correctness and official EvalPlus base/plus accuracy are distinct measurements.

Generation records seed-level predictions. Use matched problem IDs when computing pass@k, majority vote, or paired bootstrap intervals; do not pool rows across unmatched protocols.
