# Ecosystem Simulator

## Medium Term Goal
* Simulated environment with different species
* Each species has unique characteristics that could include
  * Strength
  * Speed
  * Friendliness (to join a pack?)
    * When they're a part of a pack, they move together and any food eaten is shared evenly within the pack
* Each individual within a species has slight variations? Or does the simulated environnment start with one individual of each species?
* Could this be interactive and allow the user to place a custom species while the simulation is running

# Short Term Goal
* 100 by 100 array
* Individuals with unique neural nets placed randonly within the grid
* Individual can move up, down, left, right each step
* Neural Net Inputs:
  * Field of vision (Food particle positions in 10 by 10 grid around them, each cell is either out of bounds, food, or empty)
* When an individual eats, they replenish energy to survive longer
* If an individual survives for X steps, they can reproduce
* There is a population cap
* Thoughts
  * If this runs long enough, will they all just be moving along the same exact path at the same time? Do we need a way to ensure they don't occupy the same cell? 