from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
import random
from tensorflow import keras 
import numpy as np
import tensorflow as tf
from tf_agents.utils.common import function
from tf_agents.policies import policy_saver
import os
from enum import Enum

OrganismAction = Enum('OrganismAction', ['UP', 'RIGHT', 'DOWN', 'LEFT'], start=0)


class Food:
    state = None

    def __init__(self, position):
        self.position = position

class Organism:
    neural_net = None
    position = None

    def __init__(self, position, neural_net):
        self.position = position
        self.neural_net = neural_net
    
    def _action_policy(self, obs):
        return random.choice(list(OrganismAction)).value

    def act(self):
        action = self._action_policy(self.position)
        return action
    
food_positions = {}
organism_positions = {}

class Driver:
    dimensions = (10,10)
    initial_food_count = 20
    initial_organism_count = 5
    food_spawn_rate = 0.2

    def __init__(self, dimensions, initial_food_count, initial_organism_count):
        self.organisms = []
        self.food = []
        self.dimensions = dimensions

        self._fill_cell_contents(initial_food_count, initial_organism_count)
    
    def _fill_cell_contents(self, initial_food_count, initial_organism_count):
        food_tracker = initial_food_count

        while food_tracker > 0:
            x_rand_cell = random.randint(0, self.dimensions[0] - 1)
            y_rand_cell = random.randint(0, self.dimensions[1] - 1)
            
            if (x_rand_cell, y_rand_cell) in food_positions:
                continue
            else:
                food_positions[(x_rand_cell, y_rand_cell)] = True
                food_tracker -= 1
                piece_of_food = Food((x_rand_cell, y_rand_cell))

                self.food.append(piece_of_food)
        
        organism_tracker = initial_organism_count

        while organism_tracker > 0:
            x_rand_cell = random.randint(0, self.dimensions[0] - 1)
            y_rand_cell = random.randint(0, self.dimensions[1] - 1)

            if (x_rand_cell, y_rand_cell) in organism_positions or (x_rand_cell, y_rand_cell) in food_positions:
                continue
            else:
                organism_positions[(x_rand_cell, y_rand_cell)] = True
                organism_tracker -= 1
                organism = Organism((x_rand_cell, y_rand_cell), (4,))
                
                self.organisms.append(organism)
    
    def _step(self):

        organism_actions = {}
        for organism in self.organisms:
            organism_action = organism.act()
            print(f"organism {organism} taking action {organism_action}")

            organism_actions[organism] = organism_action
          
          
        

    def run(self, max_steps=100):         

        for step in range(max_steps):
            print(f"Running step {step}")
            result = self._step()


driver = Driver((10, 10), 20, 5)

print([x.position for x in driver.organisms])
print([x.position for x in driver.food])

# driver.run(max_steps=10)