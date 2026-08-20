##  Instructions


1. `args.py` is the configuration file.
2. Set `--is_embed_watermark=1` and run `python main.py` to generate the PoLO proof.
3. Set `--is_fine_tue=1` to trigger the OFA attack configuration.

## Minimal Usage

Go to the code directory first:

```bash
cd code
```

Generate a PoLO proof:

```bash
python main.py --is_embed_watermark=1
```

Trigger the OFA / fine-tuning attack configuration:

```bash
python fine_tune_attack.py --is_fine_tue=1
```

## Configuration

The main configuration file is:

```text
code/args.py
```

You can modify experiment settings there, including:

- dataset
- model/network type
- watermark length
- training epochs
- differential privacy settings
- fine-tuning attack settings
- pruning attack settings

## Main Files

```text
code/main.py                # Main script for PoLO proof generation
code/args.py                # Configuration file
code/fine_tune_attack.py    # Fine-tuning / OFA-style attack script
code/pruning_attack.py      # Pruning attack script
code/hash_based_verify.py   # Hash-based verification script
```

## Notes

- The project uses relative paths, so running scripts from `code` is the safest choice.
- If you want to change the training or attack setup, edit `args.py` first.

## Output directory

Scripts write checkpoints to `../model_files/<dataset>/<network>/` relative to
`code/`. The directory tree ships empty; it is created for you and must stay in
place. Trained checkpoints are not included in this artifact because of their
size (see the top-level README).
