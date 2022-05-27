#!/usr/bin/env python3
"""train atari play"""
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def model(height, width, channels, actions):
    """model architechture function"""
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4),
              activation='relu',
              input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2),
              activation='relu',))
    model.add(Convolution2D(64, (3, 3),
              activation='relu',))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


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


env = gym.make("Breakout-v0", render_mode='human')


def train(env):
    env.reset()
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n
    conv_model = model(height=height, width=width,
                       channels=channels, actions=actions)
    print(conv_model.summary())
    dqn = agent(conv_model, actions)
    dqn.compile(optimizer=Adam(lr=0.00025), metrics=['mae', 'accuracy'])
    dqn.fit(env, nb_steps=10000, verbose=2, visualize=True)
    dqn.save_weights('policy.h5', overwrite=True)


train(env)
