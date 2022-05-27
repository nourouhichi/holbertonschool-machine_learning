#!/usr/bin/env python3
"""reiforcement learning on atari game"""
import gym
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Input
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def create_model(height, width, channels, actions):
    """model architecture function"""
    inputs = Input(shape=(3, height, width, channels))
    lay1 = Convolution2D(32, 8, strides=4, activation='relu')(inputs)
    lay2 = Convolution2D(64, 4, strides=2, activation='relu')(lay1)
    lay3 = Convolution2D(64, 3, strides=1, activation='relu')(lay2)
    lay4 = Flatten()(lay3)
    lay5 = Dense(512, activation='relu')(lay4)
    action = Dense(actions, activation="linear")(lay5)
    return Model(inputs=inputs, outputs=action)


def agent(model, actions):
    """build agent"""
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1.,
                                  value_test=.2,
                                  value_min=.1,
                                  nb_steps=10000)
    memory = SequentialMemory(limit=1000000, window_length=3)
    dqn_agent = DQNAgent(model=model, memory=memory, policy=policy,
                         enable_dueling_network=True,
                         dueling_type='avg',
                         nb_actions=actions,
                         nb_steps_warmup=1000)
    return dqn_agent


def train(env):
    env.reset()
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n
    conv_model = create_model(height=height, width=width,
                              channels=channels, actions=actions)
    print(conv_model.summary())
    dqn = agent(conv_model, actions)
    dqn.compile(optimizer=Adam(lr=0.00025), metrics=['mae', 'accuracy'])
    dqn.fit(env, nb_steps=10000, verbose=2, visualize=True)
    dqn.save_weights('policy.h5', overwrite=True)


env = gym.make("Breakout-v0", render_mode='human')
train(env)
