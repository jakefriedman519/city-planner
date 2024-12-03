import numpy as np
import tensorflow as tf
import gym
class DQL(tf.keras.Model):
    def __init__(self, num_actions, grid_shape, budget_shape, **kwargs): 
        super(DQL, self).__init__(
            **kwargs
        )  # Pass any additional kwargs to the parent constructor
        self.num_actions = num_actions
        self.grid_input = tf.keras.Input(shape=grid_shape, name="grid_input")
        self.budget_input = tf.keras.Input(shape=budget_shape, name="budget_input")

        # Initialization
        initializer = tf.keras.initializers.HeNormal()

        # Convolutional layers for grid processing
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer=initializer)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer=initializer)
        self.flatten = tf.keras.layers.Flatten()

        # Dense layers for combining grid and budget inputs
        self.dense1 = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.dense2 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=initializer)
        self.output_layer = tf.keras.layers.Dense(num_actions, activation="linear")

    def call(self, inputs):
        # Inputs: a dictionary with keys "grid_input" and "budget_input"
        grid_input = inputs["grid_input"]
        budget_input = inputs["budget_input"]

        # Process grid input through convolutional layers
        x = self.conv1(grid_input)
        x = self.conv2(x)
        x = self.flatten(x)

        # Combine grid features and budget input
        combined = tf.concat([x, budget_input], axis=-1)

        # Pass through dense layers to get the output
        x = self.dense1(combined)
        x = self.dropout(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output

    def get_config(self):
        # Get the configuration of the model to save it
        config = super(DQL, self).get_config()
        # Add any additional configuration parameters that might be used for deserialization
        config.update(
            {
                "num_actions": self.num_actions,
                "grid_shape": self.grid_input.shape[
                    1:
                ],  # Exclude the batch size dimension
                "budget_shape": self.budget_input.shape[
                    1:
                ],  # Exclude the batch size dimension
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Reinitialize the model from the saved config
        return cls(
            num_actions=config["num_actions"],
            grid_shape=config["grid_shape"],
            budget_shape=config["budget_shape"],
        )
