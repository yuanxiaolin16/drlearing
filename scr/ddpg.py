import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#actor network
#2 linear layers with relu activation ending with tanh activation
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
        
#critic network
#2 linear layers with relu activation ending with no activation   视频作者多次实验发现，输出层没有激活函数时收敛情况较好
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)       
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        X = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(X))
        x = torch.relu(self.fc2(x))
        X = self.fc3(x)
        return x
        
        
#replay buffer
#deque with 15000 max length
#sample, push, and __len__ methods
class ReplayBuffer:
    def __init__(self, max_size=15000):
        self.buffer = deque(maxlen=max_size)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.uint8)       
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
        
    def __len__(self):
        return len(self.buffer)

   

#DDPG agent
#get_action, learn, and __init__ methods
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_size=64, actor_lr=0.0001, critic_lr=0.001, gamma=0.99, tau=0.05, batch_size=128):  #阅读代码后，希望分开设置actor和critic的学习率;tau为更新网络时滤波器的参数
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_size).to(device)  
        self.critic_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.memory = ReplayBuffer()
        
        self.gamma = gamma
        self.tau = tau
        

        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy()
        return action
    
    def update(self, BATCH_SIZE=64):    #把数据转成张量
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)               #rewards是标量，需要升维      
        next_states = torch.FloatTensor(next_states).to(device)
        # dones = torch.FloatTensor(dones).to(device)                   #dones是标量，需要升维
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

       # update critic
        next_actions = self.actor_target(next_states)
        target_values = rewards + self.gamma * self.critic_target(next_states, next_actions)
        values = self.critic(states, actions)    #计算当前的critic网络输出
        critic_loss = nn.MSELoss()(values, target_values.detach())   #计算与目标网络之间的loss
    
        self.critic_optimizer.zero_grad()  #清空梯度
        critic_loss.backward()  #反向传播
        self.critic_optimizer.step()  #更新网络参数
        
        #update actor
        actions = self.actor(states)
        actor_loss = -self.critic(states, actions).mean()   #计算actor网络的loss
        
        self.actor_optimizer.zero_grad()  #清空梯度
        actor_loss.backward()  #反向传播
        self.actor_optimizer.step()  #更新网络参数
        
        #update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    
    
    
        

