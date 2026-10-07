# Optional generation and KV-cache check

Run `01_generate.ipynb` in the existing ML environment. It loads the full FP32 model,
uses 32 prompt tokens, generates 20 tokens, and checks prefill/decode call lengths.
Each execution writes a unique record under `results/notebook_runs/`.
This is a host-side reference experiment; it is not quantized bare-metal inference.
No newly executed generation result is bundled with the cleanup.
