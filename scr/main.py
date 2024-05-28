# 主要作用：创建环境，创建智能体，实现智能体与环境的交互，具体智能体的算法放在专门的智能体的类中
#Description:Main file for running the DDPG algorithm
import torch
import gym
from ddpg import DDPGAgent
import numpy as np

#initialize environment
env = gym.make(id='Pendulum-v1')#'Pendulum-v1' 是一个特定的环境标识符，代表了倒立摆（Inverted Pendulum）环境的一个版本（v1）。倒立摆环境通常被用来测试和训练强化学习算法。在这个环境中，智能体（Agent）需要学会控制一个倒立摆系统，使其能够保持竖直位置。这是一个经典的强化学习问题，旨在让智能体学会通过施加力或扭矩来保持物体的平衡。

#获取状态量和动作量的维度
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

#initialize agent
agent = DDPGAgent(STATE_DIM, ACTION_DIM)

#hyperparameters
MAX_EPISODES = 200#最大轮数
MAX_STEPS = 200#最大步数
BATCH_SIZE = 64

#training loop
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(MAX_STEPS):
        action = agent.get_action(state) + np.random.normal(0, 0.1, size=ACTION_DIM)  #直接输出一个高斯分布的变量，[-1,1]
        next_state, reward, done, _, _= env.step(2*action) #与action范围不同，[-2,2]
        agent.memory.push(state, action, reward, next_state, done)    #把对环境交互得到的状态、动作、奖励、下一个状态、是否结束，保存到经验池中
        episode_reward += reward       
        state = next_state
        
        if len(agent.memory) > BATCH_SIZE:  
            agent.update(BATCH_SIZE)
        if done:
            break
    print(f'Episode: {episode}, Reward: {episode_reward}')


env.close()    #训练结束之后首先要关闭环境

#save model
torch.save(agent.actor.state_dict(), 'actor.pth')       
torch.save(agent.critic.state_dict(), 'critic.pth') 
        