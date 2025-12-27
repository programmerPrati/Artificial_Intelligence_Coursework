# AI methodologies
This repository contains 5 Artificial Intelligence projects from my class.

We bagan with considering planned and unplanned systems, then focused on the planned ones. Search problems were introduced with states, actions and costs. This leads to Markov Decision Processes.
If we can map out these states and transitions, we can search the state space. This is like planning with a map.

However, in the world this state space is unknown or impossible to fully compute. This is where we have to explore using Q-learning (value and policy iteration) and other reinforcement learning techniques.
This is formed by not knowing the transitionand reward function in MPDs.

The projects in this class were done with a Pacman setup initially provided by UC Berkeley

## Search Based Planning (Project 1)
This project deals with searching the state space to find our goal state using graph searching algorithms: DFS, BFS, UCS and A*. Also implemented heuristic for A*.

## Multi-agent Search
Transitioned from single-agent planning to adversarial environments. I implemented Minimax and Expectimax algorithms to model decision-making against intelligent ghosts, 
utilizing evaluation functions to quantify state-utility under competition.

## Supervised Learning
Investigated pattern recognition and function approximation. Key implementations included Perceptrons, Linear Regression, and Recurrent Neural Networks (RNNs) 
to classify data and predict sequences, forming the bridge between classical AI and modern Deep Learning.

## Reinforcement Learning
Explored agents that learn through environmental feedback rather than hard-coded logic. I implemented Value Iteration for known MDPs and Q-Learning
for model-free environments, allowing the agent to optimize its policy through trial and error.

## Inference and estimation
Addressed "Hidden" state problems where ghost locations are unknown. I utilized probabilistic inference and sensor filtering to estimate positions based on noisy data, 
moving the agent from perfect information to reasoning under uncertainty.
