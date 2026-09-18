# External model artifact contract

Large-language-model weights and generated prediction tensors are published
separately on Hugging Face: **`https://huggingface.co/<organization>/<model-name>`**.
This code release does not embed checkpoints or private machine paths. Replace this
placeholder with the final model URL before publication.

The public entry points accept a local prediction tensor:

```bash
python -m chembomas.wet.fd_wqp.run --round round_7 \
  --predictions /path/to/LLM_predict.pt
python -m chembomas.dry.edbo.run \
  --pseudo-predictions /path/to/LLM_predict.pt
```

The tensor must be readable by `torch.load` and contain one prediction per row of
the corresponding inference space. The released data tables document row order;
the code does not silently reorder candidates. Replace the placeholder Hugging Face
URL in this file before publication and record the exact model revision/checksum.
