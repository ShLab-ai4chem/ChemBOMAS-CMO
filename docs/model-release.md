# External model artifact contract

Large-language-model weights are published separately on Hugging Face:

**`https://huggingface.co/AI4Chem/ChemBOMAS-basemodel`**

The prediction tensors used by the paper are included in this repository alongside
their input tables, so the BO experiments do not require downloading model weights.

Fetch the checkpoint when regenerating the published predictions from scratch:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download AI4Chem/ChemBOMAS-basemodel \
  --local-dir ./ChemBOMAS-basemodel
```

The regression entry points accept that local directory through
`--pretrained_model_path`; optional LoRA adapters and regression heads are passed
with `--lora_adapter_path` and `--yield_predictor_path`.

The public entry points accept a local prediction tensor:

```bash
python -m chembomas.wet.fd_wqp.run --round round_7
python -m chembomas.dry.edbo.run \
  --iteration 5
```

The bundled tensors are readable by `torch.load` and contain one prediction per row
of the corresponding inference space. External tensors may be passed through the
command-line overrides; the code does not silently reorder candidates. Record the
exact model revision/checksum when regenerating predictions, and pin that revision
when reporting results.
