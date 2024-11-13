import gym
from gym import spaces
import numpy as np


class CityPlanningEnv(gym.Env):
    def __init__(self):
        super(CityPlanningEnv, self).__init__()

        self.grid_size = 10
        self.budget = 1000

        self.building_map = {
            0: 'school',
            1: 'hospital',
            2: 'factory',
            3: 'residential',
            4: 'park',
            5: 'restaurant'
        }

        # Building costs
        self.building_costs = {
            "school": 100,
            "hospital": 100,
            "factory": 75,
            "residential": 10,
            "park": 20,
            "restaurant": 5
        }

        # Reward parameters
        self.rewards = {
            "school_residential": 100,
            "hospital_residential": 50,
            "park_existence": 10,
            "hospital_school": 20,
            "residential_restaurant": 5,
            "factory_existence": 50,
            "school_existence": 50,
            "hospital_existence": 50,
            "residential_existence": 100,
            "restaurant_existence": 25
        }

        # Penalties
        self.penalties = {
            "school_factory": -250,
            "residential_factory": -150,
            "same_building_adjacent": -25
        }

        # Define action and observation spaces
        # Action space: 0 to 5 (representing building types) for each of the 100 squares
        self.action_space = spaces.MultiDiscrete([10, 10, 6])

        # Observation space: a grid of building types and remaining budget
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.int32),
            "budget": spaces.Box(low=0, high=self.budget, shape=(), dtype=np.float32)
        })

        # Initialize environment state
        self.reset()

    def reset(self):
        # Reset the environment
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)  # All empty initially
        self.remaining_budget = self.budget
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        # Return current grid and remaining budget
        return {
            "grid": self.grid.copy(),
            "budget": self.remaining_budget
        }

    def step(self, action):
        # Action: (row, col, building_type)
        row, col, building_type = action
        building_type = int(building_type)
        building_name = self.building_map[building_type]
        # Check if the position is empty and within budget
        if self.grid[row, col] == 0 and self.remaining_budget >= self.building_costs[building_name]:
            self.grid[row, col] = building_type + 1  # +1 to represent the building type in the grid
            self.remaining_budget -= self.building_costs[building_name]
            reward = self._calculate_reward(row, col, building_type)
        else:
            reward = -10  # Penalty for an invalid move

        # Check if budget is exhausted or grid is full
        self.done = self.remaining_budget <= 0 or np.all(self.grid)
        return self._get_observation(), reward, self.done, {}

    def _calculate_reward(self, row, col, building_type):
        reward = 0

        # Reward and penalty calculations based on the grid and position
        if building_type == 0:  # School
            reward += self.rewards["school_existence"]
            reward += self._check_nearby(row, col, 2, 1) * self.rewards["school_residential"]
            reward += self._check_nearby(row, col, 3, 4) * self.penalties["school_factory"]

        elif building_type == 1:  # Hospital
            reward += self.rewards["hospital_existence"]
            reward += self._check_nearby(row, col, 4, 1) * self.rewards["hospital_residential"]
            reward += self._check_nearby(row, col, 3, 0) * self.rewards["hospital_school"]

        elif building_type == 2:  # Factory
            reward += self.rewards["factory_existence"]
            reward += self._check_nearby(row, col, 2, 1) * self.penalties["residential_factory"]

        elif building_type == 3:  # Residential
            reward += self.rewards["residential_existence"]
            reward += self._check_nearby(row, col, 2, 5) * self.rewards["residential_restaurant"]

        elif building_type == 4:  # Park
            reward += self.rewards["park_existence"]

        elif building_type == 5:  # Restaurant
            reward += self.rewards["restaurant_existence"]

        # Check for adjacent same-type building penalty (excluding Residential and Restaurant)
        if building_type not in {3, 5}:
            reward += self._check_nearby(row, col, 1, building_type) * self.penalties["same_building_adjacent"]

        return reward

    def _check_nearby(self, row, col, radius, target_building):
        count = 0
        for i in range(max(0, row - radius), min(self.grid_size, row + radius + 1)):
            for j in range(max(0, col - radius), min(self.grid_size, col + radius + 1)):
                if (i != row or j != col) and self.grid[i, j] == target_building + 1:
                    count += 1
        return count

    def render(self, mode="human"):
        print(f"Grid:\n{self.grid}")
        print(f"Remaining Budget: {self.remaining_budget}")

    def close(self):
        pass
