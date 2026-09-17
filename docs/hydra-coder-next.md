# One H200: Qwen3-Coder-Next Q5_K_M

Official quantization: https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF
Downloaded revision: b82fb7382639d97b38fa7672e526c760c2fb358e.
Four shards total 56,710,372,768 bytes (52.82 GiB). Both presets share these files.

- `qwen3-coder-next-q5km`: native 262144-token context.
- `qwen3-coder-next-q5km-1m`: experimental 1000000-token context with upstream YaRN settings. Long-context quality and runtime compatibility are not verified here.

Request one H200, one server slot, preferably 128G host memory. Estimated FP16 attention KV cache is 6 GiB at 262144 tokens, 22.9 GiB at 1M (12 attention layers, 2 KV heads, head size 256). Weights plus KV leave substantial room within an H200's 141GB, but recurrent state, compute buffers and runtime overhead also consume memory. No inference test was run.

The local gateway overrides the preset context: set the selected host's `llamacpp.context_size` to the intended value. Keep the client's context budget lower to leave generation room. Downloading/selecting this preset does not silently increase those budgets.

Context management should happen between requests: retain instructions and recent turns, summarize older messages, then send the shorter history. Existing gateway character truncation is lossy deletion, not a token-aware summary. These hybrid-model presets disable runtime context shifting; do not assume shifting the live KV cache is supported. Summarization requires another inference request but does not require restarting the server.


## MTP

The published Qwen3-Coder-Next checkpoint has no MTP/NextN tensors in its
[weight index](https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/model.safetensors.index.json).
No compatible draft artifact was found in the Qwen, ggml-org or Unsloth GGUF
repositories (checked 2026-09-17), so these presets deliberately leave MTP
unconfigured. Qwen3-Next architecture support in llama.cpp alone does not supply
missing trained weights. Do not substitute an unrelated model's MTP head.

For supported presets, the default maximum draft length is now 8 tokens, including
Qwen3.8-27B. This is a maximum proposal length, not a guarantee of eight accepted
tokens or improved speed. Restart an existing server to apply it.
