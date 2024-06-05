# Waymax Motion Planning

This repository contains code for motion planning using the Waymax simulator. The code includes vehicle state updates, reward calculations, and visualization of simulation states. The primary goal is to implement and test motion planning algorithms in a simulated environment.

## Features

- State updates for vehicles in a simulated environment
- Reward calculations for different actions
- Visualization of simulation states

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Besttenson/5AEIP0.git
cd 5AEIP0
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. If you want to train with local filtered scenario, download the .zip file from (link). Unzip it and put the folder under the root repository.

4. To train the model, run training.py
```bash
python training.py
```
5. To visualize the mode, run visualization.py
```bash
python visualization.py
```
6. Config.py shows all hyperparameters that can be tuned
