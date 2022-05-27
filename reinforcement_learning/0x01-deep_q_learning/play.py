#!/usr/bin/env python3
"""play atari play"""
import gym
from rl.agents.dqn import DQNAgent
import keras as K
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
train = __import__('train').train

env = gym.make('Breakout-v0')
train(env)
state = env.reset()
DQN_agent = DQNAgent(
    model=K.models.load_model('policy.h5'),
    nb_actions=env.action_space.n,
    memory=SequentialMemory(
        limit=1000000,
        window_length=4
    ),
    policy=GreedyQPolicy()
)
DQN_agent.compile(
    optimizer=Adam(
        lr=.00025,
        clipnorm=1.0
    ),
    metrics=['mae']
)
DQN_agent.test(
    env,
    nb_episodes=11,
    visualize=True
)
