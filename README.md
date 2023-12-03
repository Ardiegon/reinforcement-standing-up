# USD-2023Z
Reinforcement learning with PyBullet simulator

## Instalation
run
```bash
conda create --name usd python==3.9
pip install -r requirements.txt
```
1. If using windows, it may be that you have to install Visuall C++ Tools lib from here https://visualstudio.microsoft.com/pl/visual-cpp-build-tools/
2. If Your CUDA driver version is different than >12.0, install torch by hand from https://pytorch.org/get-started/locally/

## Train
to Humanoid model with SAC agent 
```bash
python /scripts/train.py --cuda
```

## Tests
#TODO
