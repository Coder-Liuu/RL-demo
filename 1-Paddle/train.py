import os
import gym
import numpy as np
import parl
from game_env import Paddle
from parl.utils import logger, ReplayMemory
from game_model import GameModel
from game_agent import GameAgent
from parl.algorithms import DQN

LEARN_FREQ = 5
MEMORY_SIZE = 20000
MEMORY_WARMUP_SIZE = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
GAMMA = 0.999
max_episode = 1000


# 训练一个回合
def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)

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
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def save_all(agent, episode, reward):
    # 保存模型
    logger.info("模型保存成功!")
    save_path = f'save/model3-{episode}_{round(reward,2)}.ckpt'
    agent.save(save_path)


def load_all(agent):
    logger.info("加载模型成功!")
    save_path = f'save\model3-1000_106.92.ckpt'
    agent.restore(save_path)


def main():
    env = Paddle()
    obs_dim = 5
    act_dim = 3
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 设置记忆库
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)

    # 建立AI
    model = GameModel(obs_dim=obs_dim, act_dim=act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = GameAgent(alg, act_dim=act_dim,update_target_steps=200, e_greed=0.1, e_greed_decrement=1e-6)

    load_all(agent)
    eval_reward = run_evaluate_episodes(agent, env, render=False)
    # print(eval_reward)

    # 记忆库预存
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)


    # 开始训练
    episode = 0
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
