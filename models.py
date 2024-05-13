import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#hidden_init函数的作用：根据https://arxiv.org/abs/1509.02971，初始化一个隐藏层（除了输出层），
def hidden_init(layer):
    """
    Description
    -------------
    According to https://arxiv.org/abs/1509.02971
    Initialize the hidden layers (except the output layer) with Uniform(-1/sqrt(fan_in),
    1/sqrt(fan_in)) Where fan_in is the number of units in the layer
   
   #算法：Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))，其中fan_in是层中的单元数,
   
    Parameters
    -------------
    layer : torch.nn.modules.linear.Linear object, the fully connected layer to
            initialize.

    Returns
    -------------
    -1/sqrt(fan_in), 1/sqrt(fan_in)
    """
    #平方根下【 -1/sqrt(fan_in), 1/sqrt(fan_in)】

    fan_in = layer.weight.data.size()[1]  # Not sure if it should be [1] instead.
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

#Actor是一个神经网络模型的 PyTorch 类，继承于神经网络 nn.Module 类。所以Actor具备nn.Nodule的所有属性和方法。
#这一点非常重要，因为当使用一个Actor的实例的时候，其本身具有nn.Module的所有属性和方法。
class Actor(nn.Module):
    def __init__(self, state_size, action_size=1, seed=0, fc1_units=16, fc2_units=8):
        """
        Description
        -------------
        Actor constructor.

        Parameters
        -------------
        state_size  : Int, Dimension of each state
        action_size : Int, Dimension of each action
        seed        : Int, Random seed
        fc1_units   : Int, Number of nodes in first hidden layer
        fc2_units   : Int, Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()#继承父类的属性
        self.fc1 = nn.Linear(state_size, fc1_units)#隐藏层初始化
        self.fc2 = nn.Linear(fc1_units, fc2_units)#隐藏层初始化
        self.fc3 = nn.Linear(fc2_units, action_size)#输出层初始化
        self.reset_parameters() #初始额外参数

    def reset_parameters(self):
        """
        Description
        -------------
        Reinitialize the layers
        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))#初始化隐藏层的权重数据
        self.fc1.bias.data.fill_(0)#初始化隐藏层，偏置数据
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))#初始化隐藏层的权重数据
        self.fc2.bias.data.fill_(0)#初始化隐藏层，偏置数据
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)#初始化输出层的权重数据
        self.fc3.bias.data.fill_(0)#初始化输出层，偏置数据

    #forward函数的作用：将输入的state数据，通过神经网络模型，得到输出的action数据。
    def forward(self, state):
        """
        Description
        -------------
        Apply a forward pass on a state with Actor network.
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)#返回输出层的数据


#Critic是一个神经网络模型的 PyTorch 类，继承于神经网络 nn.Module 类。所以Critic具备nn.Nodule的所有属性和方法。
class Critic(nn.Module):
    def __init__(self, state_size, action_size=1, seed=0, fcs1_units=64, fc2_units=32):
        """
        Description
        -------------
        Critic constructor

        Parameters
        -------------
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fcs1_units (int): Number of nodes in the first hidden layer
        fc2_units (int): Number of nodes in the second hidden layer
        """

        super(Critic, self).__init__()#继承父类的属性
        self.fcs1 = nn.Linear(state_size + action_size, fcs1_units)#隐藏层初始化
        self.fc2 = nn.Linear(fcs1_units, fc2_units)#第二个 隐藏层初始化
        self.fc3 = nn.Linear(fc2_units, 1)#输出成初始化
        self.reset_parameters() #参数初始化

    def reset_parameters(self):
        """
        Description
        -------------
        Reinitialize the layers
        """

        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))#隐藏层的权重数据初始化
        self.fcs1.bias.data.fill_(0)#隐藏层的偏置数据初始化
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))#隐藏层的权重数据初始化
        self.fc2.bias.data.fill_(0)#隐藏层的偏置数据初始化
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)#输出层的权重数据初始化
        self.fc3.bias.data.fill_(0)#输出层的偏置数据初始化

    #forward函数的作用：将输入的state和action数据，通过神经网络模型，得到输出的Q值。
    def forward(self, state, action):
        """
        Description
        -------------
        Apply a forward pass on a state with Critic network.
        """

        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fcs1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)# 返回输出层的数据
