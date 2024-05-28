import gym
import numpy as np
import torch
from ddpg import Actor
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize environment
env = gym.make('Pendulum-v1',render_mode='rgb_array')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]    #获取状态和动作的维度

#load actor model  导入训练好的模型
actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load('actor.pth'))

#initialize pygame
pygame.init()
screen = pygame.display.set_mode((500, 500))  #设置屏幕大小
pygame.display.set_caption('Pendulum-v1')
clock = pygame.time.Clock()

#run environment
EPISODES = 5
MAX_STEPS = 200

for episode in range(EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(MAX_STEPS):
        frame = env.render() #输出这一帧的图像
        frame = np.transpose(frame, (1, 0, 2)) #转换图像的格式   #第一次报错时，点击frame发现是frame为0，说明环境渲染没有输出
        frame = pygame.surfarray.make_surface(frame) #转换为pygame的surface格式,渲染到pygame上去
        pygame.transform.scale(frame, (500, 500), screen) #缩放图像
        screen.blit(frame, (0, 0)) #渲染到屏幕上
        pygame.display.flip()   #刷新屏幕
        
        clock.tick(60)  #控制每秒刷新帧率
        
        state = torch.FloatTensor(state).to(device)  #转换为tensor格式
        action = actor(state).detach().cpu().numpy()  #获取动作
        next_state, reward, done, _, _= env.step(2*action)  #执行动作
        
   
                
        episode_reward += reward  #累计奖励
        state = next_state  #更新状态
        
        if done:

            break    #如果结束了就跳出循环
        
                
    
    print("Episode: {}, Reward: {}".format(episode, episode_reward))

