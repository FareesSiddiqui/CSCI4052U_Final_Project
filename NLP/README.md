# Chess Analysis Tool

## Introduction
For this final checkpoint of our semester long project I have succesfully implemented the NLP (Natural Language Processing) portion of the project. This portion of the tool is still a work in progress and all bugs will be worked out for the final report. The NLP portion of this project allows for users to ask questions about their position to a tailored chatbot about their game in real time. The Chatbot is based off of GPT-2 and has been finetuned with chess commentary data. I did not train this model, rather I was able to find model weights on huggingface of GPT-2 that were fine tuned for the game of chess. Using this chatbot the user can ask questions about the current positions, potential mating nets, general strategies and anything else they may think of. Currently I have just implemented the backend of the tool which is the machine learning portion, over the next coming weeks I will be working on implementing the frontend and tieing all the pieces together in a cohesive manner. This pretrained model is also able to predict optimal moves in the position, but for the final version of this project I will be creating my own engine, I will be trying to mimic Google Deepminds strategy by using a combination of Monte Carlo Tree Search (MCTS) and Reinforcement learning.

## Reinforcement Learning (RL)

Although I have not yet implemented the Reinforcement Learning portion of my project I have a few avenues to explore where I will try to implement this technique which are listed below:

#### Move Optimization

Reinforcement Learning could help optimize move suggestions by learning patterns from an engine like stockfish, and leveraging those decisions to learn the game at a deeper level.
- Train a reinforcement learning agent to predict moves that align with both the chess engine's recommendations and the user's play style
- Incorporate feedback from the outcomes of games to refine the strategy and generate better recommendations.

RL can enable the chatbot to acct as a virtual coach, suggesting moves that not only align with engine evaluations but also help users learn better strategies. The reward system for the agent could be based on the improvement of user performance over time.


## Future Work

1. Incorporate Reinforcement Learning: Implement the RL features I mentioned above, mainly the game engine. Hopefully this engine will be good enough to compete with beginner players at the very least. 

2. Implement the frontend interface for the application so the users can interact with all portions of the application.