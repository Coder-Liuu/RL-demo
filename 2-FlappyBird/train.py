import os
import gym
import numpy as np
import parl
from game_env import FlappyBird_Env
from parl.utils import logger
from collections import deque
from replay_memory import ReplayMemory
from game_model import GameModel
from game_agent import GameAgent
from parl.algorithms import DQN
import time

LEARN_FREQ = 5
MEMORY_SIZE = 10000
MEMORY_WARMUP_SIZE = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.99
IMAGE_SIZE = (80,80)

CONTEXT_LEN = 4         # 上下文长度
FRAME_SKIP = 4          # 连续多少帧使用一样的动作

max_episode = 10000     # 最大训练回合数量
init_episode = 1        # 初始回合训练数

def env_reset(env, action=0):
    global recent_obs
    recent_obs = deque(maxlen=CONTEXT_LEN)
    recent_obs.append(env.reset())
    for i in range(CONTEXT_LEN - 1):
        next_obs, _, _, _ = env.step(action)
        recent_obs.append(next_obs)
    obs = list(recent_obs)
    return np.concatenate(obs)

def env_steps(env, action=0):
    global recent_obs
    total_reward = 0
    for i in range(FRAME_SKIP):
        next_obs, reward, isOver, info = env.step(action)
        total_reward += reward
        recent_obs.append(next_obs)
        if isOver:
            break
    obs = list(recent_obs)
    return np.concatenate(obs), total_reward, isOver, info


# 训练一个回合
def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env_reset(env, env.sample())
    step = 0
    while True:
        step += 1
        action = agent.sample([obs])
        next_obs, reward, done, _ = env_steps(env, action)

        # 花费时间不多
        rpm.append([obs, action, reward, next_obs, done])

        # 训练模型
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)
        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# evaluate 5 episodes
def run_evaluate_episodes(agent, env, eval_episodes=5, render=False):
    eval_reward = []
    for i in range(eval_episodes):
        obs = env_reset(env, env.sample())
        episode_reward = 0
        while True:
            action = agent.predict([obs])
            obs, reward, done, _ = env_steps(env, action)
            episode_reward += reward
            if done:
                break
        print(episode_reward)
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def save_all(agent, episode, reward):
    # 保存模型
    logger.info("模型保存成功!")
    save_path = f'save/model3-{episode}_{round(reward,2)}.ckpt'
    agent.save(save_path)


def load_all(agent):
    logger.info("加载模型成功!")
    save_path = f'save/model3-4201_159.82.ckpt'
    agent.restore(save_path)


def main():
    env = FlappyBird_Env(IMAGE_SIZE=IMAGE_SIZE)
    obs_dim = IMAGE_SIZE
    act_dim = 2
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 设置记忆库
    rpm = ReplayMemory(MEMORY_SIZE)

    # 建立AI
    model = GameModel(obs_dim=obs_dim, act_dim=act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = GameAgent(alg, act_dim=act_dim,update_target_steps=200, e_greed=0.1, e_greed_decrement=1e-6)

    load_all(agent)
    eval_reward = run_evaluate_episodes(agent, env, render=False)

    # 记忆库预存
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)


    # 开始训练
    episode = init_episode
    while episode < max_episode:
        # 训练部分
        for i in range(50):
            total_reward = run_train_episode(agent, env, rpm)
            episode += 1
            print(f"episode = {episode},reward = {round(total_reward,2)}")

        # 测试部分
        eval_reward = run_evaluate_episodes(agent, env, render=False)
        logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(episode, agent.e_greed, round(eval_reward,2)))
        save_all(agent, episode, eval_reward)


if __name__ == '__main__':
    main()