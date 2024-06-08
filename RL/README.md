# Waymax Motion Planning

This repository contains code for motion planning using the Waymax simulator. The code includes vehicle state updates, reward calculations, and visualization of simulation states. The primary goal is to implement and test motion planning algorithms in a simulated environment.

## Features

- State updates for vehicles in a simulated environment
- Reward calculations for different actions
- Visualization of simulation states

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Besttenson/5ARIP10.git
cd 5ARIP10/RL
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. If you want to train with local filtered scenario, refer to [Scenario Extraction folder](https://github.com/Besttenson/5ARIP10/tree/main/waymo_motion_scenario_mining) to generate your own scenarios.
   Otherwise, you could use the scenarios in Waymo Open Dataset for training.

5. To train the model, run training.py
```bash
python training.py
```
5. To visualize the mode, run visualization.py
```bash
python visualization.py
```
6. Config.py shows all hyperparameters that can be tuned
