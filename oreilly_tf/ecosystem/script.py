import random
import numpy as np
import sys
from enum import Enum

from evolutionary_keras.models import EvolModel
from evolutionary_keras.optimizers import NGA
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

OrganismAction = Enum('OrganismAction', ['UP', 'RIGHT', 'DOWN', 'LEFT'], start=0)
CellContents = Enum('CellContents', ['EMPTY', 'ORGANISM', 'FOOD', 'OUTOFBOUNDS'], start=0)


food_positions = {}
organism_positions = {}

MAX_ENERGY = 10
OLD_AGE = 30
VISION_DISTANCE = 3

GAME_STEPS = 25
DIMENSIONS = (10, 10)
INITIAL_FOOD = 50
INITIAL_ORG = 10
FOOD_SPAWN_RATE = 1
FOOD_SPAWN_PERC = 80
REPRODUCE_PROB = 80
MARKER_SIZE = 32

# GAME_STEPS = 200
# DIMENSIONS = (100, 100)
# INITIAL_FOOD = 4000
# INITIAL_ORG = 500
# MARKER_SIZE = 8
# FOOD_SPAWN_RATE = 100
# FOOD_SPAWN_PERC = 80
# REPRODUCE_PROB = 80

# GAME_STEPS = 500
# DIMENSIONS = (500, 500)
# INITIAL_FOOD = 6000
# INITIAL_ORG = 1000
# MARKER_SIZE = 0.05
# FOOD_SPAWN_RATE = 1000
# FOOD_SPAWN_PERC = 80
# REPRODUCE_PROB = 80

game_history = []
food_y_stack = []
org_y_stack = []

class Food:
    state = None

    def __init__(self, position):
        self.position = position

class Organism:
    neural_net = None
    position = None
    vision_distance = VISION_DISTANCE
    max_energy = MAX_ENERGY
    old_age = OLD_AGE

    def __init__(self, position, model, nga_optimizer):
        self.position = position
        self.model = model
        self.nga_optimizer = nga_optimizer
        self.energy = self.max_energy
        self.age = 0
    
    def _action_policy(self, obs):
        action = list(self.model(obs[np.newaxis])[0])
        return action.index(max(action))

    def get_desired_action(self):
        vision = self._look_around().flatten()

        action = self._action_policy(vision)
        
        return action

    def eat_food(self):
        self.energy = self.max_energy

    # TODO: This needs to be debugged. Seems to be inaccurate 
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
                    # The organism doesn't care about its current location (should it? There could be another organism there)
                    observable_space[array_x_pos][array_y_pos] = CellContents.EMPTY.value
                elif abs_x_pos >= DIMENSIONS[0] or abs_y_pos >= DIMENSIONS[1] or abs_x_pos < 0 or abs_y_pos < 0:
                    observable_space[array_x_pos][array_y_pos] = CellContents.OUTOFBOUNDS.value
                elif (abs_x_pos, abs_y_pos) in food_positions:
                    observable_space[array_x_pos][array_y_pos] = CellContents.FOOD.value
                elif (abs_x_pos, abs_y_pos) in organism_positions:
                    observable_space[array_x_pos][array_y_pos] = CellContents.ORGANISM.value
            
        # print(observable_space)
        return observable_space

class Driver:
    food_spawn_rate = FOOD_SPAWN_RATE
    food_spawn_perc = FOOD_SPAWN_PERC

    def __init__(self, dimensions, initial_food_count, initial_organism_count):
        self.organisms = []
        self.dimensions = dimensions
        self.initial_food_count = initial_food_count
        self.initial_organism_count = initial_organism_count
        self.game_over = False

        self._fill_cell_contents()
    
    def track_board(self):
        org_pos = [x.position for x in self.organisms]
        org_energy = [x.energy for x in self.organisms]
        step_data = {
            "food_positions": list(food_positions.keys()).copy(),
            "organism_positions": org_pos,
            "organism_energies": org_energy,
            # Testing (this can probably be retrieved differently at some point)
            "org0_view": self.organisms[0]._look_around() if len(self.organisms) > 0 else [[]],
            "org0_id": id(self.organisms[0]) if len(self.organisms) > 0 else "NONE",
        }
        game_history.append(step_data)

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
    
    def _add_new_food(self, food_count):
        food_tracker = food_count

        # Make sure we don't try to spawn food if too many tiles already have food
        if len(food_positions) >= (DIMENSIONS[0] * DIMENSIONS[1]) - self.food_spawn_rate:
            return

        while food_tracker > 0:
            x_rand_cell = random.randint(0, self.dimensions[0] - 1)
            y_rand_cell = random.randint(0, self.dimensions[1] - 1)
            
            if (x_rand_cell, y_rand_cell) in food_positions:
                continue
            else:
                food_positions[(x_rand_cell, y_rand_cell)] = True
                food_tracker -= 1


    def _fill_cell_contents(self):
        self._add_new_food(self.initial_food_count)
        
        organism_tracker = self.initial_organism_count

        while organism_tracker > 0:
            x_rand_cell = random.randint(0, self.dimensions[0] - 1)
            y_rand_cell = random.randint(0, self.dimensions[1] - 1)

            if (x_rand_cell, y_rand_cell) in organism_positions or (x_rand_cell, y_rand_cell) in food_positions:
                continue
            else:
                organism_positions[(x_rand_cell, y_rand_cell)] = True
                organism_tracker -= 1

                inputs = tf.keras.layers.Input(shape=((VISION_DISTANCE * 2 + 1) ** 2,), name='my_input')
                hidden = tf.keras.layers.Dense(16,)(inputs)
                outputs = tf.keras.layers.Dense(len(OrganismAction),)(hidden)

                new_nga = NGA(population_size = 1, mutation_rate = 0.2)
                new_model = EvolModel(inputs, outputs)
                new_model.compile(new_nga)

                organism = Organism((x_rand_cell, y_rand_cell), new_model, new_nga)
                
                self.organisms.append(organism)
    
    def _get_target_pos(self, position, action):
        """Get the position that is targetted based on current position and action

        Args:
            position (tuple): (x,y) of current position
            action (int): enum representing a movement action (up, down, left, right)

        Returns:
            tuple: Target position
        """
        target_position = list(position)
        if action == OrganismAction.UP.value and position[1] < DIMENSIONS[1] - 1:
            target_position[1] += 1
        elif action == OrganismAction.RIGHT.value and position[0] < DIMENSIONS[0] - 1:
            target_position[0] += 1
        elif action == OrganismAction.DOWN.value and position[1] > 0:
            target_position[1] -= 1
        elif action == OrganismAction.LEFT.value and position[0] > 0:
            target_position[0] -= 1
        
        return target_position[0], target_position[1]

    def _check_organism_action_results(self, organism_actions):
        """Determine what the results of the actions will be before updating the organism's states,
        in case there are conflicts to resolve (two organisms trying to eat the same food)

        Args:
            organism_actions (dict): dict with key of organism and value of action

        Returns:
            dict: dict with key organism and value containing final state of organism after all actions
        """
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
        
        return final_results
    
    def _update_organism(self, organism, result):
        organism.energy -= 1
        organism.age += 1

        if organism.energy == 0 or organism.age == OLD_AGE:
            self.organisms.remove(organism)
        
        organism.position = result.get("position")
        organism_positions[organism.position] = True
        if "reward" in result:
            organism.eat_food()
            if random.randint(0, 100) < REPRODUCE_PROB:
                organism_optimizer = organism.nga_optimizer
                mutants = organism_optimizer.create_mutants()
                
                weights = mutants[-1]

                inputs = tf.keras.layers.Input(shape=((VISION_DISTANCE * 2 + 1) ** 2,), name='my_input')
                hidden = tf.keras.layers.Dense(16,)(inputs)
                outputs = tf.keras.layers.Dense(len(OrganismAction),)(hidden)

                new_nga = NGA(population_size = 1, mutation_rate = 0.05)
                new_model = EvolModel(inputs, outputs)
                new_model.set_weights(weights)
                new_model.compile(new_nga)


                x_rand_cell = random.randint(0, self.dimensions[0] - 1)
                y_rand_cell = random.randint(0, self.dimensions[1] - 1)

                organism_offspring = Organism((x_rand_cell, y_rand_cell), new_model, new_nga)
                self.organisms.append(organism_offspring)


    def _step(self):

        organism_actions = {}
        for organism in self.organisms:
            organism_action = organism.get_desired_action()

            organism_actions[organism] = organism_action

        results = self._check_organism_action_results(organism_actions)

        organism_positions.clear()

        for organism, result in results.items():
            self._update_organism(organism, result)

        if random.randint(0, 100) < self.food_spawn_perc:
            self._add_new_food(self.food_spawn_rate)
        
        if len(self.organisms) == 0:
            self.game_over = True
            
        
    def run(self, max_steps=100):         

        for step in range(max_steps):
            print(f"Running step {step}")
            
            # Only enable when you want to animate
            self.track_board()

            result = self._step()

            if self.game_over:
                print("No organisms left!")
                break


def input_validation():
    if INITIAL_FOOD > DIMENSIONS[0] * DIMENSIONS[1]:
        print("Initial food count exceeds space")
        sys.exit(-1)
    
    if INITIAL_ORG > DIMENSIONS[0] * DIMENSIONS[1]:
        print("Initial organism count exceeds space")
        sys.exit(-1)

input_validation()

driver = Driver((DIMENSIONS[0], DIMENSIONS[1]), INITIAL_FOOD, INITIAL_ORG)

driver.run(max_steps=GAME_STEPS)




def animation_update(frame):
    global food_y_stack
    global org_y_stack

    ax.clear()
    ax2.clear()

    game_step = game_history[frame]

    food_len = len(game_step.get("food_positions"))
    org_len = len(game_step.get("organism_positions"))

    org_x = [x for x,y in game_step.get("organism_positions")]
    org_y = [y for x,y in game_step.get("organism_positions")]
    org_alphas = [1.0 for x in range(org_len)]
    # org_alphas = [x / (MAX_ENERGY * 2.0) + 0.5 for x in game_step.get("organism_energies")]

    # org_marker_size = [MARKER_SIZE * (x / (MAX_ENERGY * 2.0) + 0.5) for x in game_step.get("organism_energies")]
    org_marker_size = [MARKER_SIZE for x in range(org_len)]

    food_x = [x for x,y in game_step.get("food_positions")]
    food_y = [y for x,y in game_step.get("food_positions")]
    food_marker_size = [MARKER_SIZE for x in range(food_len)]

    boundaries_x = [-1, -1, DIMENSIONS[0], DIMENSIONS[0]]
    boundaries_y = [-1, DIMENSIONS[1], -1, DIMENSIONS[1]]


    if org_len > 0:
        ax.scatter(org_x, org_y, c="r", s=org_marker_size, alpha=org_alphas)
    if food_len > 0:
        ax.scatter(food_x, food_y, c="g", s=food_marker_size)
    ax.scatter(boundaries_x, boundaries_y, c="white")

    time_x = [x for x in range(frame)]

    # Sometimes animation_update gets called multiple times for the same frame
    # so we should only update when it's a new frame
    if frame == 0:
        food_y_stack = []
        org_y_stack = []
        ax1.clear()
    if len(time_x) > len(food_y_stack):
        food_y_stack = food_y_stack + [food_len]
        org_y_stack = org_y_stack + [org_len]

    color_map = ["green", "red"]
    ax1.stackplot(time_x, food_y_stack, org_y_stack, colors=color_map)

    org0_view = game_step.get("org0_view")

    # Does organism[0] even exist? has the population gone extinct?
    if len(org0_view[0]) > 0:
        
        org0_food_x = []
        org0_food_y = []
        org0_orgs_x = []
        org0_orgs_y = []
        org0_oob_x = []
        org0_oob_y = []
        for x in range(VISION_DISTANCE * 2 + 1):
            for y in range(VISION_DISTANCE * 2 + 1):
                if org0_view[x][y] == CellContents.FOOD.value:
                    org0_food_x.append(x)
                    org0_food_y.append(y)
                elif org0_view[x][y] == CellContents.ORGANISM.value:
                    org0_orgs_x.append(x)
                    org0_orgs_y.append(y)
                elif org0_view[x][y] == CellContents.OUTOFBOUNDS.value:
                    org0_oob_x.append(x)
                    org0_oob_y.append(y)
        ax2.scatter(VISION_DISTANCE, VISION_DISTANCE, c="purple", marker='x')
        if len(org0_food_x) > 0:
            ax2.scatter(org0_food_x, org0_food_y, c="g")
        if len(org0_orgs_x) > 0:
            ax2.scatter(org0_orgs_x, org0_orgs_y, c="r")
        if len(org0_oob_x) > 0:
            ax2.scatter(org0_oob_x, org0_oob_y, c="black")

    org_boundaries_x = [-1, -1, VISION_DISTANCE * 2 + 1, VISION_DISTANCE * 2 + 1]
    org_boundaries_y = [-1, VISION_DISTANCE * 2 + 1, -1, VISION_DISTANCE * 2 + 1]

    ax2.scatter(org_boundaries_x, org_boundaries_y, c="white")
    
    ax.set_title("Ecosystem")
    ax1.set_title("Food vs Organism Count")
    ax2.set_title(f"Organism {game_step.get('org0_id')} view")
    
    return

fig, ax = plt.subplots(figsize=(5,10))
# plt.axis('off')

divider = make_axes_locatable(ax)
ax1 = divider.append_axes("bottom", size=0.8, pad=0.5)
ax2 = divider.append_axes("bottom", size=2, pad=0.5,)

scat = ax.scatter(0, 0, c="b", s=5, )
ax.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')

ani = animation.FuncAnimation(fig=fig, func=animation_update, frames=len(game_history), interval=100, repeat=False)

plt.show()

# ani.save(filename="./pillow_example.gif", writer="pillow")
