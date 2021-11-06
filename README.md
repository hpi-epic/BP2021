# Online Marketplace Simulation: A Testbed for Self-Learning Agents
Working repository in context of the bachelorproject "Online Marketplace Simulation: A Testbed for Self-Learning Agents" at the research group Enterprise Platform and Integration Concepts.

The goal of the project is to develop a universal simulation platform for markets with varying numbers of merchants. Being able to run various market simulations is highly relevant for many firms such as SAP and its partners. As the platform is designed as a tool to support evaluation and research, aspects like configurability and ease of use are crucial. While the technology stack is left open for now, high compatibility to common simulation APIs (such as Gym, TF-Agents) is required.
For more complex setups, communication protocols between different agents might have to be implemented as well.

The simulation should cover the interaction between customers and particularly competing merchants, including self-learning agents and their rule-based opponents. While the focus can be put on several different aspects, an adjustable customer behavior model (which determines each participant!s sales) has to be developed. The platform should generate sales and interaction data for each of the merchants, which can then in turn be fed to the self-learning agents. Monitoring tools are required to analyze each agent!s policy and their effects on the overall market. With the help of such simulations, we seek to study the competitiveness of self-adapting pricing tools and their long-term impact on market competitors and customers.

# First Protoype for Marketplace Simulation and Deep Q-Learning
The four Python files in this repository belong to a simple protoype for marketplace simulation. It is build to simulate a simple market with two vendors trying to maximize their profit. One vendor is part of the environment as a rule based competitor, the other one in a simulated agent. The customer behaviour depends on the price and the quality of the product. Furthermore, some random events make the customers less predictable.

If you have not yet done so, install Anaconda and run the following command to create an environment and install the required packages:
```console
conda env create -f scripts/environment.yml -n your_venv_name
```
To activate your created environment use:
```console
conda activate your_venv_name
```
To update an existing environment with the needed packages run the following command:
```console
conda env update --name your_venv_name --file scripts/environment.yml
```
