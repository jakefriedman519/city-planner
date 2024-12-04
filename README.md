# City Planning Simulation with Deep Q-Learning

This project implements a city planning simulation using Deep Q-Learning (DQL) to optimize building placement in a grid-based environment.

## Files

The project consists of four main Python files:

- `model.py`: Contains the main training and evaluation logic for the Deep Q-Learning agent.
- `gui.py`: Implements the graphical user interface for visualizing the city planning process.
- `dql.py`: Defines the Deep Q-Learning NN architecture.
- `city_planner_gym.py`: Implements the gym environment for the city planning simulation.

## Usage

1. Install dependencies in a virtual environment

```bash
python -m venv .venv
source .venv
pip install -r requirements.txt
```

2. Run the model

> [!NOTE]  
> This runs both the training and evaluation loop

```bash
python model.py
```

To visualize the trained agent's performance, ensure the GUI flag is set to `True` in `model.py`
