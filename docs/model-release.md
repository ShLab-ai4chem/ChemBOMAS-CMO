# External model artifact contract

Large-language-model weights are published separately on Hugging Face:
**`https://huggingface.co/<organization>/<model-name>`**. The prediction tensors
used by the paper are included in this repository alongside their input tables, so
the BO experiments do not require downloading model weights. Replace this
placeholder with the final model URL before publication.

The public entry points accept a local prediction tensor:

```bash
python -m chembomas.wet.fd_wqp.run --round round_7
python -m chembomas.dry.edbo.run \
  --iteration 20
```

The bundled tensors are readable by `torch.load` and contain one prediction per row
of the corresponding inference space. External tensors may be passed through the
command-line overrides; the code does not silently reorder candidates. Record the
exact model revision/checksum when the model URL is finalized.
