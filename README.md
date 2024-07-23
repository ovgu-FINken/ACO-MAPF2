# ACOMultiAgentPathfinder Algorithm

This document describes the key principles of the Multi-Agent Ant-Colony Optimization (ACO) for Multi-Agent Pathfindfing (MAPF).

## Overview

The ACOMultiAgentPathfinder is a solver for the Multi-Agent Pathfinding Problem (MAPF) using Ant Colony Optimization (ACO). The key idea is to evolve the trajectories of multiple agents simultaneously, where each agent is repelled by the pheromones of other agents to avoid collisions. The algorithm searches a time-expanded graph G_t and uses the true distance in the original graph G (found by A*) as a heuristic.

## Problem

In the Graph $G$ find a set of paths $\PI = \pi_1, \pi_2, ..., \pi_N$ for $N$ agents, such that each path goes from start to goal, and no conflicts between the paths exist.
A conflict occurs, if and only if two agents occupy the same state at the same time step, or adjecent time-steps $\pi_i(t) = \pi_j(t)$ or $\pi_i(t) = \pi_j(t+1)$.

## Graph Representation

- **Original Graph $G$**: Represents the physical environment where agents move.
- **Time-Expanded Graph $G_t$**: A graph that represents both spatial and temporal dimensions, allowing for conflict-free path planning.

## Agent

Each agent in the system is represented by the `Agent` class with the following key features:

- Unique ID, start position, and goal position
- Pheromone matrix $\Tau^i$: $\tau_{u(t),v(t+1)}$ where $(u(t),v(t+1))$ is an edge in $G_t$ 
- Node-based pheromone matrix for calculating repelling effect of other agents

The policy of an agent is defined by a decision function which selects the next action in $G_t$ based on its own pheromone values and the pheromone values of other agents.

### Decision Function

An ant calculates a path, taking a sequence of actions based on a decsion function:

The algorithm uses an epsilon-greedy approach for balancing exploration and exploitation:

- With probability ε, choose a random neighbor (explore)
- With probability 1-ε, choose the best neighbor based on pheromone levels and heuristic information (exploit)
- ε decreases over time, favoring exploitation in later iterations

The exploitation strategy works as follows:
$p_{ij} = \tau_{ij}^\alpha * \eta_{ij}^\beta * (\frac{1}{\hat \tau_{ij} + 1})^\gamma$
Where:

$\tau_{ij}$ is the agent's own pheromone on the edge $(i,j)$
$\eta_{ij}$ is the heuristic information (inverse of the distance to the goal)
$\hat \tau_{ij}$ is the average pheromone of other agents at the destination node $j$
### ACO Process

The main ACO process involves the following steps:

1. **Initialization**: Create time-expanded graph and initialize agents with pheromone trails.
2. **Ant Tours**: For each iteration and each ant:
   - Generate paths for all agents using epsilon-greedy strategy
   - (Check for conflicts between agent paths)
3. **Pheromone Update**: Update pheromone trails based on the best solution found in the iteration
4. **Inter-Agent Communication**: Agents share node-based pheromone information at specified intervals
5. **Solution Generation**: After all iterations, generate a final solution, based on a greedy strategy

### Pheromone Update

The pheromone update works as follows:
After calculating a set of paths for a given pheromone matrix, we modify the pheromone values in the following way:
- Evaporation $\tau_ij = (1 - \rho) \tau_{ij}$
- Dispersion (not implemented yet)
- Deposition of new pheromones based on path quality

### Communication

Agents periodically share their pheromone information, allowing for indirect communication and conflict avoidance.
The model for this communication strategy are island models in cooperative coevolution in Evolutionary Algorithms.

The information that an agent receives during communication are:
1) The average values for the pheromone of all other agents: $\sum_{j \neq i} \frac{\Tau^j}{N-1}
2) Information on the estimated state occupancy of other agents

## Execution

Execution is governed by the ACOMultiAgentPathfinder class, which contains information on the environment and makes sure that each agent executes their algorithm and information is communicated between agents.