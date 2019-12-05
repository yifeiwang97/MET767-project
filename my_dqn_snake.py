import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
from torch.autograd import Variable
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# parameters
gama = 0.999
memory_size = 50000
memory_observe_size = 5000
mini_batch_size = 32
delay_copy = 10
batch_size = 32
epsilon_max = 0.95
epsilon_decay = 0.9999
epsilon_min = 0.005


# policy network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2)
        self.pool1 = nn.MaxPool2d((2,2))

        self.fc1 = nn.Linear(2496*39, out_features=1000)
        self.fc2 = nn.Linear(1000, out_features=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.view(-1, 2496 * 39)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class DQN:
    def __init__(self):
        self.state = None
        #self.action = np.zeros((4,), dtype=int)
        #self.action = torch.IntTensor(1, 4).zero_()
        self.action = None
        self.reward = None
        self.next_state = None
        self.loss = []
        self.memory = deque()
        self.save_dir = os.path.join(os.getcwd(), 'saved_DQN')
        self.policy_name = 'Q.pth'
        self.target_name = 'Target_Q.pth'
        self.policy = Net().to(device)
        self.target = Net().to(device)
        self.copy_network()
        self.load_network()
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=0.001)
        self.critition = nn.SmoothL1Loss()

        self.epsilon = epsilon_max
        self.timestep = 0

    def load_network(self):
        if os.path.isdir(self.save_dir):
            policy_path, target_path = os.path.join(self.save_dir, self.policy_name), os.path.join(self.save_dir, self.target_name)
            self.policy.load_state_dict(torch.load(policy_path))
            self.target.load_state_dict(torch.load(target_path))
            print('Load policy model at %s ' % policy_path)
            print('Load target model at %s ' % target_path)

    def copy_network(self):
        self.target.load_state_dict(self.policy.state_dict())

    def save_network(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        policy_path, target_path = os.path.join(self.save_dir, self.policy_name), os.path.join(self.save_dir, self.target_name)
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.target.state_dict(), target_path)
        print('Saved policy model at %s ' % policy_path)
        print('Saved target model at %s ' % target_path)

    def get_action(self):
        output = self.policy.forward(self.state)
        if random.random() > self.epsilon:
            index = torch.max(output, dim=1)[1].data.cpu().numpy()[0]
        else:
            index = random.randint(0,3)
        self.action = index
        return index

    def interface(self, next_state, reward, terminal):
        action = self.action
        state = self.state.clone()
        state, next_state = Variable(state, requires_grad = True), Variable(next_state, requires_grad = True)
        if len(self.memory) > memory_size:
            self.memory.popleft()
        else:
            self.memory.append((state, action, reward, next_state, terminal))
        if len(self.memory) > memory_observe_size:
            self.memory_replay()
            if self.epsilon > epsilon_min:
                self.epsilon = self.epsilon*epsilon_decay
        if self.timestep % delay_copy == 0:
            self.copy_network()

        self.state = next_state.cuda()
        self.timestep = self.timestep + 1

    def memory_replay(self):
        minibatch = random.sample(self.memory, mini_batch_size)
        state = [data[0] for data in minibatch]
        action = [data[1] for data in minibatch]
        reward = [data[2] for data in minibatch]
        next_state = [data[3] for data in minibatch]
        terminal = [data[4] for data in minibatch]
        y_train = torch.FloatTensor(mini_batch_size, 4).zero_()
        y_label = torch.FloatTensor(mini_batch_size, 4).zero_()
        for i in range(mini_batch_size):
            y_eval = self.policy.forward(state[i])
            label = y_eval.clone()
            y_train[i] = y_eval.clone()
            if terminal[i]:
                label[0][action[i]] = reward[i]
            else:
                y_next_eval = self.target.forward(next_state[i])
                label[0][action[i]] = reward[i] + torch.max(y_next_eval, dim=1)[0].data.cpu().numpy()[0]
            #y_label.append(label.data.cpu().numpy())
            y_label[i] = label.clone()
        self.optimizer.zero_grad()
            #y_train, y_label = torch.from_numpy(np.array(y_train)).cuda(), torch.from_numpy(np.array(y_label)).cuda()
        loss = self.critition(y_eval, label)
        loss.backward()
        print("step:", self.timestep)
        self.optimizer.step()


