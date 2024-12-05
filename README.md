# This is the code release of the following paper: Unique Views, Unique Moves: Next-POI Recommendation with Dynamic Semantics and Asymmetrical Relations.

# Quick Start

## Dependencies

```python
python==3.8
torch==1.10.0
torchvision==0.11.1
dgl-cu113==0.9.1
tqdm
torch-scatter>=2.0.8
pyg==2.0.4
```

## Train models

1.Switch to src/ folder

```python
cd src/
```

2.Run scripts

```python
python main.py  --gpu 0 --dataset PHO
```
