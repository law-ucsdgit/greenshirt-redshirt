# greenshirt-redshirt

This project has been split out from the notebook into separate files so training and evaluation can happen in different scripts.

## Structure

- `train.py` – runs the training loop
- `evaluate.py` – loads a saved checkpoint and evaluates it
- `connect4/` – reusable game logic, model code, and config

## Run training

```bash
python3 train.py
```

## Run evaluation

```bash
python3 evaluate.py --checkpoint checkpoints/policy_net1_ep300.pt --html
```

The checkpoints are saved under `checkpoints/` by default.