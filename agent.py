import os
from time import sleep #多线程阻塞
from collections import deque #双向队列
from collections import namedtuple #命名元组

import numpy as np #
import torch.optim as optim #优化器
import torch.optim.lr_scheduler as lr_scheduler #学习率调度器
import torch #pytorch
import torch.nn.functional as F #torch.nn.functional 模块提供了一些实用的函数，这些函数是在 torch.nn 模块的基础上构建的
from tensorboardX import SummaryWriter #绘制图表
import matplotlib.pyplot as plt #画图工具
import seaborn as sns #作图工具
from IPython import display

# local imports
from memory import Memory, PrioritizedMemory, Node
from models import Actor, Critic

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "dones")
)

GAMMA = 0.99  # discount factor #折扣因子，用于衡量未来奖励的重要性。在计算累积奖励时，未来奖励会按照折扣因子的大小进行衰减。
TAU_ACTOR = 1e-1  # 软更新的时间常数，用于更新 Actor 和 Critic 网络的目标参数。
# 软更新是一种渐进地将目标网络的参数更新为当前网络参数的方法。
TAU_CRITIC = 1e-3  # soft update of critic target parameters
LR_ACTOR = 1e-4  # learning rate of the actor #Actor 和 Critic 网络的学习率。学习率控制了参数在每次更新时的调整大小。
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY_actor = 0  # L2 weight decay of the actor #Actor 和 Critic 网络的权重衰减系数。权重衰减用于惩罚模型参数的大小，有助于防止过拟合。
WEIGHT_DECAY_critic = 1e-2  # L2 weight decay of the critic
BATCH_SIZE = 64  # minibatch size #每个训练步骤中使用的样本批量大小。在每次参数更新时，会从回放缓冲区中随机抽取一批样本进行训练
BUFFER_SIZE = int(1e6)  # replay buffer size 放缓冲区的大小，用于存储交互过程中的经验数据。回放缓冲区是用于实现经验回放的一种数据结构，用于增强学习中的样本重用和样本随机性。
PRETRAIN = 64  # number of pretraining steps (must be greater than BATCH_SIZE)  #noqa #预训练步数，指在开始正式训练之前的预先训练步数。这些步骤用于填充回放缓冲区，并确保足够的数据可供训练。
MAX_STEP = 100  # number of steps in an episode #每个回合的最大步数
WEIGHTS = "weights/"  # path to the repository where to save the models' weights #保存模型权重的路径
FC1_UNITS_ACTOR = 16  # Number of nodes in first hidden layer #隐藏层节点数 Actor 网络中的隐藏层节点数量。
FC2_UNITS_ACTOR = 8  # Number of nodes in second hidden layer #隐藏层节点数 Actor 网络中的隐藏层节点数量。
FC1_UNITS_CRITIC = 64  # Number of nodes in first hidden layer of the critic network 
FC2_UNITS_CRITIC = 32  # Number of nodes in second hidden layer of the critic network #
DECAY_RATE = 0  # Decay rate of the exploration noise ##探索噪声的衰减率。探索噪声是用于增加智能体探索环境的随机性的一种技术，在训练过程中逐渐减小，以便智能体可以更多地依赖其学到的策略。
EXPLORE_STOP = 1e-3  # Final exploration probability #探索噪声的最终值。当探索噪声衰减到该值以下时，停止衰减过程。


#optimal_f 函数的作用：计算最优解的函数，用于计算交易成本为0、l2和l1的成本模型。
def optimal_f(p, pi, lambd=0.5, psi=0.3, cost="trade_l2"):
    """
    Description
    --------------
    Function with the shape of the optimal solution for cost models with 0, l2 and l1
    trading costs.

    Parameters
    --------------
    p     : Float, the next signal value. 表示下一个信号值，即下一时刻的价格或者状态
    pi    : Float, the current position. 表示当前的持仓量。
    lambd : Float > 0, Parameter of the cost model. 分别是成本模型和交易成本模型的参数
    psi   : Float > 0, Parameter of our model defining the trading cost.#成本模型 交易成本
    cost  : String in ['none', 'trade_l1', 'trade_l2'], cost model. #成本模型 交易成本

    Returns
    --------------
    Float, The function evaluation (which is the next trade).

    如果成本模型是 'trade_0'，则返回的决策是 $p / (2 * \lambda) - \pi$，这是一个基本的交易策略。
    如果成本模型是 'trade_l2'，则返回的决策是 $p / (2 * (\lambda + \psi)) + \psi * \pi / (\lambda + \psi) - \pi$，这是一个考虑了 L2 正则化的交易策略。
    如果成本模型是 'trade_l1'，则根据当前信号值 $p$ 的范围，分为三种情况进行处理：
    如果 $p <= -\psi + 2 * \lambda * \pi$，则返回的决策是 $(p + \psi) / (2 * \lambda) - \pi$。
    如果 $-psi + 2 * \lambda * \pi < p < psi + 2 * \lambda * \pi$，则返回的决策是 $0$，即不进行交易。
    如果 $p >= psi + 2 * \lambda * \pi$，则返回的决策是 $(p - \psi) / (2 * \lambda) - \pi$。
    """

    if cost == "trade_0":#trade_0 类型模型
        return p / (2 * lambd) - pi

    elif cost == "trade_l2":
        return p / (2 * (lambd + psi)) + psi * pi / (lambd + psi) - pi #价格/2*参数+成本/参数+成本-持仓

    elif cost == "trade_l1":
        if p <= -psi + 2 * lambd * pi: #如果p小于
            return (p + psi) / (2 * lambd) - pi #价格+成本/2*参数-持仓

        elif -psi + 2 * lambd * pi < p < psi + 2 * lambd * pi: #如果p在这个区间
            return 0

        elif p >= psi + 2 * lambd * pi:
            return (p - psi) / (2 * lambd) - pi #价格-成本/2*参数-持仓


#optimal_max_pos 函数的作用：计算最优解的函数，用于计算交易成本为0、l2和l1的成本模型。
def optimal_max_pos(p, pi, thresh, max_pos):
    """
    Description
    --------------
    Function with the shape of the optimal solution for MaxPos cost model with l1 trading
    cost.

    Parameters
    --------------
    p       : Float, the next signal value. #参数p表示下一个信号值，即下一时刻的价格或者状态
    pi      : Float, the current position. #参数pi表示当前的持仓量。
    thresh  : Float > 0, threshold of the solution in the infinite horizon case. #参数thresh表示阈值
    max_pos : Float > 0, maximum allowed position. #参数max_pos表示最大持仓量

    Returns
    --------------
    Float, The function evaluation (which is the next trade).
    """

    if abs(p) < thresh:#如果p的绝对值小于阈值
        return 0
    elif p >= thresh: #如果p大于阈值
        return max_pos - pi #最大持仓-持仓
    elif p <= -thresh: #如果p小于阈值
        return -max_pos - pi #-最大持仓-持仓


# Vectorizing.
#optimal_f_vec 函数的作用：将 optimal_f 函数向量化，以便在 PyTorch 中使用。
optimal_f_vec = np.vectorize(optimal_f, excluded=set(["pi", "lambd", "psi", "cost"]))
optimal_max_pos_vec = np.vectorize(#将 optimal_max_pos 函数向量化，以便在 PyTorch 中使用。
    optimal_max_pos, excluded=set(["pi", "thresh", "max_pos"])
)


#OU 过程的的噪声函数
#主要作用是在OU过程中，探索效率，以提高在惯性系统（即环境）中的控制任务的探索效率。
#OU 噪声在探索策略中的作用类似于探索率。它通过为动作添加随机扰动来引入探索性，从而帮助智能体在环境中探索不同的动作空间，学习到更优的策略。
class OUNoise:
    """
    Class of the OU exploration noise.
    """

    def __init__(self, mu=0.0, theta=0.1, sigma=0.1):
        self.mu = mu#OU过程的均值
        self.theta = theta #OU过程的参数 theta 的作用是控制 OU 过程的回归速度，即 OU 过程的均值回归到均值 mu 的速度。
        self.sigma = sigma #OU过程的参数 sigma 的作用是控制 OU 过程的扰动强度，即 OU 过程的噪声大小。
        self.reset() #重置且初始化

    #重置标记位
    def reset(self):
        self.state = self.mu

    #OU过程表达为离散形式
    #sample 函数的作用：生成一个 OU 过程的样本。原理是通过随机微分方程来模拟 OU 过程。
    def sample(self, truncate=False, max_pos=2, position=0, action=0):
        #truncate 为 False 时，OU 过程是连续的，即每次采样都会得到一个新的状态值。
        #position 为当前的持仓量，action 为当前的交易量。
        x = self.state
        if truncate:#如果截断，间断OU过程
            from scipy.stats import truncnorm #导入截断正态分布（这个地方非常重要，决定是否能够正常的得到分布值）

            #m 为最小值，M为最大值
            m = -max_pos - position - action - (1 - self.theta) * x
            M = max_pos - position - action - (1 - self.theta) * x
            #x_a为最小值，x_b为最大值
            x_a, x_b = m / self.sigma, M / self.sigma
            #truncnorm函数，生成一个截断正态分布
            X = truncnorm(x_a, x_b, scale=self.sigma)
            #dx为OU过程的噪声
            dx = self.theta * (self.mu - x) + X.rvs()
            self.state = x + dx#更新状态
            return self.state #返回状态

        else:#如果不截断，也就是连续OU过程
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn() #随机OU过程的噪声
            self.state = x + dx #更新状态
            return self.state


#Agent class 的主要作用是实现 DDPG 算法，包括经验回放、优先级经验回放、OU 噪声、Actor 和 Critic 网络的初始化、训练等功能。
class Agent:
    def __init__(
        self,
        gamma=GAMMA,#参数gamma 的作用是控制未来奖励的重要性。在计算累积奖励时，未来奖励会按照折扣因子的大小进行衰减。
        max_size=BUFFER_SIZE,
        max_step=MAX_STEP,
        memory_type="uniform",#memory_type 为 'uniform' 时，使用均匀经验回放。
        alpha=0.6,#alpha 是优先级采样的超参数，用于控制优先级采样的程度。
        beta0=0.4,#beta0 是优先级采样的超参数，用于控制优先级采样的程度。
        epsilon=1e-8, #epsilon 是优先级采样的超参数，用于确保没有样本的优先级为 0。
        sliding="oldest",#sliding 为 'oldest' 时，当树被填满且新的经验到来时，最旧的叶子将首先被更改。
        batch_size=BATCH_SIZE,#batch_size 为每次训练步骤中使用的样本批量大小。在每次参数更新时，会从回放缓冲区中随机抽取一批样本进行训练。
        theta=1.0, #theta 是 OU 噪声的参数，用于控制 OU 噪声的回归速度。
        sigma=1.0, #sigma 是 OU 噪声的参数，用于控制 OU 噪声的扰动强度。
    ):
        """
        Description
        -------------
        Constructor of class Agent

        Parameters & Arguments
        -------------
        gamma       : Float < 1 (typically 0.99), the discount factor.
        max_size    : Int, the maximum size of the memory buffer.
        max_step    : Int, number of steps in an episode.
        memory_type : String in ['uniform', 'prioritized'] type of experience replay to
                      use.
        alpha       : Float in [0, 1], power of prioritization to use (used only with
                      prioritized experience replay).
        beta0       : Float in [0, 1] that gets annealed to 1 during training because of
                      the bias introduced by priority sampling (used only with prioritized
                      experience replay).
        epsilon     : Float > 0 very small, introduced in priority estimation to ensure no
                      transition has 0 priority.
        sliding     : String in ['oldest', 'random'], when the tree gets saturated and a
                      new experience comes up.
                            - 'oldest' : Oldest leaves are the first to be changed.
                            - 'random' : Random leaves are changed.
        batch_size  : Int, the training batch size.
        theta       : Float, Noise parameter.
        sigma       : Float, Noise parameter.
        memory      : Memory object, the memory buffer.
        noise       : OUNoise object, the exploration noise which is an Ornstein-Uhlenbeck
                      process.

        """

        #断言，当gamma不在[0,1]范围内时，抛出异常
        assert 0 <= gamma <= 1, "Discount factor gamma must be in [0, 1]"
        #断言，当memory_type不在['uniform', 'prioritized']范围内时，抛出异常
        assert memory_type in [
            "uniform",
            "prioritized",
            "per_intervals",
        ], "memory must be in ['uniform', 'prioritized']"
        self.gamma = gamma
        self.max_size = max_size
        self.memory_type = memory_type
        self.epsilon = epsilon


       #memory 的作用是存储智能体与环境交互的经验数据。
       #memory_type 为 'uniform' 时，使用均匀经验回放。当 memory_type 为 'prioritized' 时，使用优先级经验回放。
        if memory_type == "uniform":
            self.memory = Memory(max_size=max_size)#初始化Memory

        elif memory_type == "prioritized": #初始化PrioritizedMemory
            self.memory = PrioritizedMemory(max_size=max_size, sliding=sliding)

        self.max_step = max_step
        self.alpha = alpha
        self.beta0 = beta0
        self.batch_size = batch_size
        self.noise = OUNoise(theta=theta, sigma=sigma) #初始化OU噪声，主要用于探索数据的特征

        # Actor Networks initialized to None
        self.actor_local = None #初始化actor_local 为None
        self.actor_target = None #初始化actor_target 为None

        # Critic Networks initialized to None
        self.critic_local = None #初始化critic_local 为None
        self.critic_target = None #初始化critic_target 为None

    def reset(self):# 重置
        """
        Description
        -------------
        Reset the exploration noise.

        Parameters
        -------------

        Returns
        -------------
        """

        self.noise.reset()

    #step函数的作用：将经验（状态、动作、奖励、下一个状态、是否结束）保存到回放缓冲区中。
    def step(self, state, action, reward, next_state, done, pretrain=False):
        """
        Description
        -------------
        Save the experience (state, action, reward, next_state, not done) in the replay
        buffer. #将经验（状态、动作、奖励、下一个状态、是否结束）保存到回放缓冲区中。

        Parameters
        -------------
        state      : 2-tuple of Floats: - state[0]: pi, the current position.
                                        - state[1]: p, the next value of the signal.
                                        #state 为当前状态，包含当前的持仓量和下一个信号值。
        action     : Float, the action taken (which is the next position).#action 为动作，即下一个持仓量。
        reward     : Float, the computed reward.#reward 为奖励，即当前步骤的奖励。
        next_state : 2-tuple of Floats representing the next state.#next_state 为下一个状态，包含下一个持仓量和下一个信号值。
        done       : Boolean, whether the episode is over or not (I'm not sure if we
                     should only consider time limit as the finishing condition).#done 为是否结束标志，表示当前回合是否结束。
        pretrain   : Boolean, whethen we are in a pretraining phase or not.#pretrain 为是否处于预训练阶段。

        Returns
        -------------
        """

        # We use Pytorch tensors for further use in the pipeline.
        ##张量是 PyTorch 中的基本数据结构
        state_mb = torch.tensor([state], dtype=torch.float)#将state转换为张量
        action_mb = torch.tensor([[action]], dtype=torch.float)#将action转换为张量
        reward_mb = torch.tensor([[reward]], dtype=torch.float)#将reward转换为张量
        next_state_mb = torch.tensor([next_state], dtype=torch.float)#将next_state转换为张量
        not_done_mb = torch.tensor([[not done]], dtype=torch.float) #将done转换为张量

        if self.memory_type == "uniform":#如果memory_type为uniform
            self.memory.add(#添加经验到缓冲区
                (state_mb, action_mb, reward_mb, next_state_mb, not_done_mb)
            )

        #这个地方是优先级经验回放的实现，
        # During pretraining, the just initialized critic network is likely to output
        # near 0 values, so we will assume the TD error to be equal to the reward.
        elif self.memory_type == "prioritized":#如果memory_type为prioritized
            priority = (#计算优先级
                (abs(reward) + self.epsilon) ** self.alpha#reward的绝对值+epsilon的alpha次方
                if pretrain
                else self.memory.highest_priority() #否则，返回最高优先级
            )
            # Add (transition, leaf) to the buffer. #添加（转换，叶子）到缓冲区
            self.memory.add(
                (state_mb, action_mb, reward_mb, next_state_mb, not_done_mb), priority
            )

    #act 函数的作用：在探索性的情况下，通过添加噪声来执行动作。主要原理是在动作中添加噪声，
    # 以便智能体可以在环境中探索不同的动作空间，学习到更优的策略。
    def act(self, state, noise=True, explore_probability=1, truncate=False, max_pos=2):
        #函数参数state 为当前状态，包含当前的持仓量和下一个信号值。
        #noise 为是否添加噪声的标志。
        #explore_probability 为探索概率，用于控制噪声的大小。
        #truncate 为是否截断噪声的标志。
        #max_pos 为最大持仓量。
        """
        Description
        -------------
        Act in an exploratory fashion by adding the noise.

        Parameters
        -------------
        state               : 2-tuple of Floats:
                              - state[0]: pi, the current position.
                              - state[1]: p, the next value of the signal.
        noise               : Boolean, whether to add exploratory noise or not.
        explore_probability : Float, decaying parameter that controls the noise magnitude.
        truncate            : Boolean, truncate the noise sample such that the position
                              remains between -MaxPos and MaxPos.
        max_pos             : Float > 0, truncate the positions between -MaxPos and
                              MaxPos.

        Returns
        -------------
        Float, the clipped action (trade) to be taken. #返回一个，OU信号量
        """

        position = state[1]#当前持仓量
        state = torch.tensor([state], dtype=torch.float)#将state转换为张量
        self.actor_local.eval()#评估模式
        with torch.no_grad():#不进行梯度计算
            action = self.actor_local(state).data.numpy()#获取动作

        self.actor_local.train()#进行一次训练
        if noise: #如果添加噪声
            #进行一次ou过程的采样
            noise_sample = self.noise.sample(
                truncate=truncate,
                max_pos=max_pos,
                position=position,
                action=float(action),
            )
            action += explore_probability * noise_sample #动作+探索概率*噪声样本

        #action 的意义是下一个持仓量，因此需要进行截断
        
        return float(action) #返回动作


     #软更新的时间常数，用于更新 Actor 和 Critic 网络的目标参数。
    def soft_update(self, local_model, target_model, tau):
        """
        Description
        -------------
        According to https://arxiv.org/abs/1509.02971
        Perform a soft target update of weights theta of the target_network using those
        theta_prime of the local network: theta_prime = tau*theta + (1 - tau)*theta_prime

        Parameters
        -------------
        local_model  : Actor or Critic local network.
        target_model : Actor or Critic target network.
        tau          : 0 < tau < 1 #软更新的时间常数，用于更新 Actor 和 Critic 网络的目标参数。

        Returns
        -------------
        Float, the clipped action to be taken (a.k.a the new position).
        """

        for target_param, local_param in zip(#更新目标网络的参数
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data#tau*本地参数+(1-tau)*目标参数
            )

    #初始化Actor和Critic网络
    #prepare_networks 函数的作用：初始化 Actor 和 Critic 网络。
    #函数pretrain 主要作用是在开始正式训练之前的预先训练步骤中，用于填充回放缓冲区，并确保足够的数据可供训练。
    #原理主要是在环境中执行动作，获取奖励，并将经验保存到回放缓冲区中。
    def pretrain(self, env, total_steps=PRETRAIN):
        """
        Description
        -------------
        Pretrain the agent to partially fill the replay buffer.

        Parameters
        -------------
        env : Environment object, it serves as the environment of training for the agent.
        # evn 为环境对象，用作智能体的训练环境。
        total_steps : Int, number of pretraining steps (must be greater than BATCH_SIZE).
        #total_steps 为预训练步数，指在开始正式训练之前的预先训练步数。这些步骤用于填充回放缓冲区，并确保足够的数据可供训练。

        Returns
        -------------
        """

        env.reset()
        with torch.no_grad():
            for i in range(total_steps):
                state = env.get_state()#从环境变量参数对象种，获取当前训练获取状态
                action = self.act( #获取动作
                    state, truncate=(not env.squared_risk), max_pos=env.max_pos
                )
                reward = env.step(action)#执行动作
                next_state = env.get_state()#获取下一个状态
                done = env.done#判断是否完成
        
                self.step(state, action, reward, next_state, done, pretrain=True)#将经验保存到回放缓冲区中
                if done:
                    #如通知完成，重置环境
                    env.reset()

#train函数，本函数是训练智能体的主要函数，包括训练Actor和Critic网络、更新目标网络、保存模型权重等功能。
    #主要参数包括环境对象、总训练次数、Actor和Critic的超参数、预训练步数、模型权重保存路径、训练频率、是否绘制图表等。
    def train(
        self,
        env,
        total_episodes=100,
        tau_actor=TAU_ACTOR,
        tau_critic=TAU_CRITIC,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        weight_decay_actor=WEIGHT_DECAY_actor,
        weight_decay_critic=WEIGHT_DECAY_critic,
        total_steps=PRETRAIN,
        weights=WEIGHTS,
        freq=50,
        fc1_units_actor=FC1_UNITS_ACTOR,
        fc2_units_actor=FC2_UNITS_ACTOR,
        fc1_units_critic=FC1_UNITS_CRITIC,
        fc2_units_critic=FC2_UNITS_CRITIC,
        decay_rate=DECAY_RATE,
        explore_stop=EXPLORE_STOP,
        tensordir="runs/",
        learn_freq=50,
        plots=False,
        pi=0.5,
        lambd=None,
        psi=None,
        phi=None,
        thresh=3,
        mile=50,
        progress="tqdm_notebook",
    ):
        """
        Description
        -------------
        Train the agent for a total number of episodes.

        Parameters
        -------------
        env                 : Environment object, it serves as the training environment
                              for the agent.
        total_episodes      : Int, total number of training episodes.
        tau_actor           : 0 < Float < 1, soft update parameter of the actor.
        tau_critic          : 0 < Float < 1, soft update parameter of the critic.
        lr_actor            : Float, learning rate of the actor network.
        lr_critic           : Float, learning rate of the critic network.
        weight_decay_actor  : Float, L2 weight decay of the actor network.
        weight_decay_critic : Float, L2 weight decay of the critic network.
        total_steps         : Int, number of pretraining steps (must be greater than
                              BATCH_SIZE).
        weights             : String, path to the repository where to save the models'
                              weights.
        freq                : Int, number of episodes between two saved models.
        fc1_units_actor     : Int, number of nodes in the first hidden layer of the actor
                              network.
        fc2_units_actor     : Int, number of nodes in the second hidden layer of the actor
                              network.
        fc1_units_critic    : Int, number of nodes in the first hidden layer of the critic
                              network.
        fc2_units_critic    : Int, number of nodes in the second hidden layer of the
                              critic network.
        decay_rate          : Float, the decay rate of exploration noise.
        explore_stop        : Float, the final exploration noise magnitude.
        tensordir           : String, path to write tensorboard scalars.
        learn_freq          : Int, each time (number_steps%learn_freq == 0), we make a
                              training step.
        plots               : Boolean, whether to plot the shape of the function at the
                              end of each episode or not.
        pi                  : Float, only used when plots is True. The plot is done by
                              fixing pi and moving p between -4 and 4.
        lambd               : Float or None, only used when plots is True. The lambda
                              parameter of the function to plot against the models.
                              If None, lambd will be the lambd parameter of the
                              environment env.
        psi                 : Float or None, only used when plots is True. The psi
                              parameter of the function to plot against the models.
                              If None, lambd will be the lambd parameter of the
                              environment env.
        thresh              : Float > 0, threshold of the solution in the infinite horizon
                              case.

        Returns
        -------------
        """

        # Creare folder where to store the Actor weights.
        if not os.path.isdir(weights):
            os.mkdir(weights)

        # Set the summary writer of tensorboard
        writer = SummaryWriter(log_dir=tensordir)

        if plots:
            _ = plt.figure(figsize=(15, 10))
            range_values = np.arange(-4, 4, 0.01)
            signal_zeros = torch.tensor(
                np.vstack((range_values, np.zeros(len(range_values)))).T,
                dtype=torch.float,
            )
            signal_ones_pos = torch.tensor(
                np.vstack((range_values, 0.5 * np.ones(len(range_values)))).T,
                dtype=torch.float,
            )
            signal_ones_neg = torch.tensor(
                np.vstack((range_values, -0.5 * np.ones(len(range_values)))).T,
                dtype=torch.float,
            )
            if psi is None:
                psi = env.psi

            if lambd is None:
                lambd = env.lambd

            if env.squared_risk:
                result1 = optimal_f_vec(
                    signal_ones_neg[:, 0].numpy(),
                    -pi,
                    lambd=lambd,
                    psi=psi,
                    cost=env.cost,
                )
                result2 = optimal_f_vec(
                    signal_zeros[:, 0].numpy(), 0, lambd=lambd, psi=psi, cost=env.cost
                )
                result3 = optimal_f_vec(
                    signal_ones_pos[:, 0].numpy(),
                    pi,
                    lambd=lambd,
                    psi=psi,
                    cost=env.cost,
                )

            else:
                result1 = optimal_max_pos_vec(
                    signal_ones_neg[:, 0].numpy(), -pi, thresh, env.max_pos
                )
                result2 = optimal_max_pos_vec(
                    signal_zeros[:, 0].numpy(), 0, thresh, env.max_pos
                )
                result3 = optimal_max_pos_vec(
                    signal_ones_pos[:, 0].numpy(), pi, thresh, env.max_pos
                )

        # Define Actor local and target networks
        self.actor_local = Actor(
            env.state_size, fc1_units=fc1_units_actor, fc2_units=fc2_units_actor
        )
        self.actor_target = Actor(
            env.state_size, fc1_units=fc1_units_actor, fc2_units=fc2_units_actor
        )

        # Define the optimizer and its learning rate scheduler for the Actor networks
        actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=lr_actor, weight_decay=weight_decay_actor
        )
        actor_lr_scheduler = lr_scheduler.StepLR(
            actor_optimizer, step_size=mile * 100, gamma=0.5
        )

        # Define Actor local and target networks
        self.critic_local = Critic(
            env.state_size, fcs1_units=fc1_units_critic, fc2_units=fc2_units_critic
        )
        self.critic_target = Critic(
            env.state_size, fcs1_units=fc1_units_critic, fc2_units=fc2_units_critic
        )

        # Define the optimizer and its learning rate scheduler for the Critic networks
        critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=lr_critic,
            weight_decay=weight_decay_critic,
        )
        critic_lr_scheduler = lr_scheduler.StepLR(
            critic_optimizer, step_size=mile * 100, gamma=0.5
        )

        # Save the initialized model
        model_file = weights + "ddpg_1" + ".pth"
        torch.save(self.actor_local.state_dict(), model_file)
        # print('\nSaved model to ' + model_file + '\n')

        # Initialize containers to add some useful information about training (useful to
        # visualize with tensorboard)
        mean_rewards = deque(maxlen=10)
        cum_rewards = []
        actor_losses = deque(maxlen=10)
        critic_losses = deque(maxlen=10)

        # Reset counting the nodes of the SumTree when using Prioritized Experience
        # Replay.
        Node.reset_count()
        # Pretraining to partially fill the replay buffer.
        self.pretrain(env, total_steps=total_steps)
        i = 0
        # exploration_probability = 1
        N_train = total_episodes * env.T // learn_freq
        beta = self.beta0
        self.reset()
        n_train = 0

        range_total_episodes = range(total_episodes)
        # setup progress bar
        if progress == "tqdm_notebook":
            from tqdm import tqdm_notebook

            range_total_episodes = tqdm_notebook(list(range_total_episodes))
            progress_bar = range_total_episodes
        elif progress == "tqdm":
            from tqdm import tqdm

            range_total_episodes = tqdm(list(range_total_episodes))
            progress_bar = range_total_episodes
        else:
            progress_bar = None

        for episode in range_total_episodes:
            # start_time = time()
            episode_rewards = []
            env.reset()
            state = env.get_state()
            done = env.done
            train_iter = 0
            # Environment Exploration phase
            while not done:
                explore_probability = explore_stop + (1 - explore_stop) * np.exp(
                    -decay_rate * i
                )
                action = self.act(
                    state,
                    truncate=(not env.squared_risk),
                    max_pos=env.max_pos,
                    explore_probability=explore_probability,
                )
                reward = env.step(action)
                writer.add_scalar("State/signal", state[0], i)
                writer.add_scalar("Signal/position", state[1], i)
                writer.add_scalar("Signal/action", action, i)
                next_state = env.get_state()
                done = env.done
                self.step(state, action, reward, next_state, done)
                state = next_state
                episode_rewards.append(reward)
                i += 1
                train_iter += 1
                if done:
                    self.reset()
                    total_reward = np.sum(episode_rewards)
                    mean_rewards.append(total_reward)
                    if (episode > 0) and (episode % 5 == 0):
                        mean_r = np.mean(mean_rewards)
                        cum_rewards.append(mean_r)
                        writer.add_scalar("Reward & Loss/reward", mean_r, episode)
                        writer.add_scalar(
                            "Reward & Loss/actor_loss", np.mean(actor_losses), episode
                        )
                        writer.add_scalar(
                            "Reward & Loss/critic_loss", np.mean(critic_losses), episode
                        )

                # Learning phase
                if train_iter % learn_freq == 0:
                    n_train += 1
                    if self.memory_type == "uniform":
                        # Sample a batch of experiences :
                        # (state, action, reward, next_state, done)
                        transitions = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)

                    elif self.memory_type == "prioritized":
                        # Sample a batch of experiences :
                        # (state, action, reward, next_state, done)
                        transitions, indices = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)

                    # Update local Critic network
                    # Use target Actor to compute the next actions to take at the sampled
                    # next states
                    actions_next = self.actor_target(next_states_mb)
                    # Use target Critic to compute the Q values of the sampled
                    # (next_states, actions)
                    Q_targets_next = self.critic_target(next_states_mb, actions_next)
                    Q_targets = rewards_mb + (
                        self.gamma * Q_targets_next * dones_mb
                    )  # Compute target Q values
                    # Compute expected Q values with the local Critic network
                    Q_expected = self.critic_local(states_mb, actions_mb)
                    # Compute the TD errors (needed to update priorities when using
                    # Prioritized replay, and also to compute the loss)
                    td_errors = F.l1_loss(Q_expected, Q_targets, reduction="none")
                    # Update the priorities of experiences in the sampled batch when
                    # Prioritized Experience Replay is used
                    if self.memory_type == "prioritized":
                        # Sum of all priorities.
                        sum_priorities = self.memory.sum_priorities()
                        # Sampling probabilities.
                        probabilities = (
                            self.memory.retrieve_priorities(indices) / sum_priorities
                        ).reshape((-1, 1))
                        # Importance sampling weights.
                        is_weights = torch.tensor(
                            1 / ((self.max_size * probabilities) ** beta),
                            dtype=torch.float,
                        )
                        # Normalize the importance sampling weights.
                        is_weights /= is_weights.max()
                        # Update parameter beta.
                        beta = (1 - self.beta0) * (n_train / N_train) + self.beta0
                        for i_enum, index in enumerate(indices):
                            # Update the priorities of the sampled experiences.
                            self.memory.update(
                                index,
                                (abs(float(td_errors[i_enum].data)) + self.epsilon)
                                ** self.alpha,
                            )

                        # Compute Critic loss function with bias correction.
                        critic_loss = (is_weights * (td_errors ** 2)).mean() / 2

                    elif self.memory_type == "uniform":
                        # Compute Critic loss function.
                        critic_loss = (td_errors ** 2).mean() / 2

                    # Store the current Critic loss value.
                    critic_losses.append(critic_loss.data.item())

                    # Minimize the Critic loss
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    # Clip the gradient to avoid taking huge steps in the gradient update
                    torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 0.1)
                    critic_optimizer.step()
                    critic_lr_scheduler.step()

                    # Update local Actor network
                    # Compute Actor loss which comes from the Off-Policy Deterministic
                    # Policy gradient theorem,
                    # see http://proceedings.mlr.press/v32/silver14.pdf and https://arxiv.org/abs/1509.02971  # noqa
                    actions_pred = self.actor_local(states_mb)
                    actor_loss = -self.critic_local(states_mb, actions_pred).mean()
                    actor_losses.append(actor_loss.data.item())

                    # Minimize the Actor loss
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # Clip the gradient to avoid taking huge steps in the gradient update
                    torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 0.1)
                    actor_optimizer.step()
                    actor_lr_scheduler.step()

                    # Update Critic and Actor target Networks
                    self.soft_update(self.critic_local, self.critic_target, tau_critic)
                    self.soft_update(self.actor_local, self.actor_target, tau_actor)

            # Plot the shape of the function and a function with approximately optimal
            # shape (regarding the cumulative reward) found by a gridsearch over lambd and
            # psi parameters
            if plots:
                plt.clf()
                self.actor_local.eval()
                with torch.no_grad():
                    plt.subplot(2, 3, 1)
                    plt.plot(
                        signal_ones_neg[:, 0].numpy(),
                        self.actor_local(signal_ones_neg)[:, 0].data.numpy(),
                        label="model",
                    )
                    plt.plot(signal_ones_neg[:, 0].numpy(), result1, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 2)
                    plt.plot(
                        signal_zeros[:, 0].numpy(),
                        self.actor_local(signal_zeros)[:, 0].data.numpy(),
                        label="model",
                    )
                    plt.plot(signal_zeros[:, 0].numpy(), result2, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 3)
                    plt.plot(
                        signal_ones_pos[:, 0].numpy(),
                        self.actor_local(signal_ones_pos)[:, 0].data.numpy(),
                        label="model",
                    )
                    plt.plot(signal_ones_pos[:, 0].numpy(), result3, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 4)
                    sns.distplot(states_mb[:, 0])

                display.clear_output(wait=True)
                if progress_bar is not None:
                    display.display(progress_bar)
                display.display(plt.gcf())
                sleep(0.0001)
                self.actor_local.train()

            # Save the Actor network weights after a number of episodes each time
            if (episode % freq) == 0:
                model_file = weights + "ddpg_" + str(episode) + ".pth"
                torch.save(self.actor_local.state_dict(), model_file)
                # print('\nSaved model to ' + model_file + '\n')

        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
