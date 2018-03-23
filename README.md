# Atari-RL
play atari game with reinforcement learning

## Pong game with policy gradient (PG)
![pong](https://github.com/mashoujiang/Atari-RL/blob/master/Pong/figures/Pong.gif)
I implemented a pong game agent with policy gradient, using only one hidden layer with 200 units. The model is trained on OpenAI gym environment. 
![reward](https://github.com/mashoujiang/Atari-RL/blob/master/Pong/figures/reward.png)
The model in ./Pong/results/ is trained with learning rate=1e-3, it can get >0 reward with only 2000 epoch.

## Requirements
* python3
* numpy
* pandas
* gym
* tensorflow
* moviepy
