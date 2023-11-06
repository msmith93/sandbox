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
CellContents = Enum('CellContents', ['EMPTY', 'ORGANISM', 'FOOD', 'OUTOFBOUNDS'], start=0)


food_positions = {}
organism_positions = {}
DIMENSIONS = (10, 10)

class Food:
    state = None

    def __init__(self, position):
        self.position = position

class Organism:
    neural_net = None
    position = None
    vision_distance = 3
    max_energy = 20

    def __init__(self, position, neural_net):
        self.position = position
        self.neural_net = neural_net
        self.energy = self.max_energy
    
    def _action_policy(self, obs):
        return random.choice(list(OrganismAction)).value

    def get_desired_action(self):
        obs = [self.position, self._look_around()]
        action = self._action_policy(obs)
        
        return action

    def eat_food(self):
        self.energy = self.max_energy

    def _look_around(self):
        # Returns an N by M array of organisms and food within X cells

        observable_space = np.zeros(shape=(self.vision_distance * 2 + 1, self.vision_distance * 2 + 1), dtype=np.int32)
        observable_space.fill(CellContents.EMPTY.value)

        for x_pos in range(-self.vision_distance, self.vision_distance + 1):
            abs_x_pos = x_pos + self.position[0]

            for y_pos in range(-self.vision_distance, self.vision_distance + 1):
                abs_y_pos = y_pos + self.position[1]
                
                # Calculations were done relative to organism (-5 to 5) but array must be indexed 0 to 10
                array_x_pos = x_pos + self.vision_distance
                array_y_pos = y_pos + self.vision_distance

                if (x_pos, y_pos) == (0, 0):
                    observable_space[array_x_pos][array_y_pos] = -1
                elif abs_x_pos >= DIMENSIONS[0] or abs_y_pos >= DIMENSIONS[1] or abs_x_pos < 0 or abs_y_pos < 0:
                    observable_space[array_x_pos][array_y_pos] = CellContents.OUTOFBOUNDS.value
                elif (abs_x_pos, abs_y_pos) in food_positions:
                    observable_space[array_x_pos][array_y_pos] = CellContents.FOOD.value
                elif (abs_x_pos, abs_y_pos) in organism_positions:
                    observable_space[array_x_pos][array_y_pos] = CellContents.ORGANISM.value
            
        # print(observable_space)
        return observable_space

class Driver:
    food_spawn_rate = 0.2

    def __init__(self, dimensions, initial_food_count, initial_organism_count):
        self.organisms = {}
        self.food = {}
        self.dimensions = dimensions
        self.initial_food_count = initial_food_count
        self.initial_organism_count = initial_organism_count

        self._fill_cell_contents()
    
    # Just for troubleshooting
    def print_board(self):
        board_contents = np.zeros(shape=self.dimensions, dtype=np.int32)
        board_contents.fill(CellContents.EMPTY.value)

        for x_pos in range(self.dimensions[0]):
            for y_pos in range(self.dimensions[1]):
                if (x_pos, y_pos) in food_positions:
                    board_contents[x_pos][y_pos] = CellContents.FOOD.value
                # Fill this one second, because if there's an organism on that cell, it's about to eat the food
                if (x_pos, y_pos) in organism_positions:
                    board_contents[x_pos][y_pos] = CellContents.ORGANISM.value
        
        print(board_contents)

    def _fill_cell_contents(self):
        food_tracker = self.initial_food_count

        while food_tracker > 0:
            x_rand_cell = random.randint(0, self.dimensions[0] - 1)
            y_rand_cell = random.randint(0, self.dimensions[1] - 1)
            
            if (x_rand_cell, y_rand_cell) in food_positions:
                continue
            else:
                food_positions[(x_rand_cell, y_rand_cell)] = True
                food_tracker -= 1
                piece_of_food = Food((x_rand_cell, y_rand_cell))

                self.food[(x_rand_cell, y_rand_cell)] = piece_of_food
        
        organism_tracker = self.initial_organism_count

        while organism_tracker > 0:
            x_rand_cell = random.randint(0, self.dimensions[0] - 1)
            y_rand_cell = random.randint(0, self.dimensions[1] - 1)

            if (x_rand_cell, y_rand_cell) in organism_positions or (x_rand_cell, y_rand_cell) in food_positions:
                continue
            else:
                organism_positions[(x_rand_cell, y_rand_cell)] = True
                organism_tracker -= 1
                organism = Organism((x_rand_cell, y_rand_cell), (4,))
                
                self.organisms[(x_rand_cell, y_rand_cell)] = organism
    
    def _get_target_pos(self, position, action):
        target_position = list(position)
        if action == OrganismAction.UP.value and position[1] < DIMENSIONS[1]:
            target_position[1] += 1
        elif action == OrganismAction.RIGHT.value and position[0] < DIMENSIONS[0]:
            target_position[0] += 1
        elif action == OrganismAction.DOWN.value and position[1] > 0:
            target_position[1] -= 1
        elif action == OrganismAction.LEFT.value and position[0] > 0:
            target_position[0] -= 1
        
        return target_position[0], target_position[1]

    def _check_organism_action_results(self, organism_actions):
        food_targets = {}
        final_results = {}
        for organism, action in organism_actions.items():
            target_pos = self._get_target_pos(organism.position, action)

            final_results[organism] = {"position": target_pos}

            if target_pos in food_positions:
                if target_pos in food_targets:
                    food_targets[target_pos].append(organism)
                else:
                    food_targets[target_pos] = [organism]

        for target_pos, organisms in food_targets.items():
            winner_organism = random.choice(organisms)

            final_results[winner_organism]["reward"] = True

            # O(n)? Seems like it shouldn't be
            food_positions.pop(target_pos)
            self.food.pop(target_pos)
        
        return final_results
    
    def _step(self):

        organism_actions = {}
        for organism in self.organisms.values():
            organism_action = organism.get_desired_action()

            organism_actions[organism] = organism_action

        results = self._check_organism_action_results(organism_actions)

        organism_positions.clear()

        for organism, result in results.items():
            
            organism.position = result.get("position")
            organism_positions[organism.position] = True
            if "reward" in result:
                organism.eat_food()
            
            organism.energy -= 1
        
        print(organism_positions)

    def run(self, max_steps=100):         

        for step in range(max_steps):
            print(f"Running step {step}")
            self.print_board()
            result = self._step()


driver = Driver((DIMENSIONS[0], DIMENSIONS[1]), 50, 20)

driver.run(max_steps=5)