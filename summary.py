# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: deep_rl_for_portfolio_optimization
#     language: python
#     name: deep_rl_for_portfolio_optimization
# ---

# We present the main results we obtained with Deep Reinforcement Learning on the three
# tractable cost models considered in the paper. 
# 我们展示了我们在论文中考虑的三个可处理成本模型上使用深度强化学习获得的主要结果。

# %matplotlib inline 
#在jupyter notebook中使用matplotlib绘图

# +
import os
import seaborn as sns #用于绘制热力图的工具包
import matplotlib.pyplot as plt #用于绘制图形的工具包
import numpy as np #用于数值计算的工具包
import pandas as pd #用于数据处理的工具包
from tqdm import tqdm_notebook #用于显示进度条的工具包
import warnings #用于忽略警告的工具包

# local imports
from agent import Agent #导入Agent类
from env import Environment #导入Environment类
from evaluation import test_models, plot_bars, plot_function #导入test_models, plot_bars, plot_function函数

##### 本文件的主要作用就是演示三种交易模型
# -

warnings.filterwarnings("ignore") #忽略警告

# + [markdown] toc-hr-collapsed=false
# # Linear trading costs with risk penalty
# -

# $$
# c(\pi_t, p_t) = -\pi_t p_t + \lambda \pi_t^2 + \psi | \pi_t - \pi_{t-1}|;\hspace{2mm}
# \lambda, \psi > 0
# $$

# ***Approximate Optimal Solution***

# for simplicity we will consider an approximate optimal solution which shape is
# parameterized making this solution easily found by a simple girdSearch. #一个简单的网格搜索，找到最佳参数
#函数的主要算法 
# $$ f\left(\pi_{t-1}, p_t \right) = \begin{cases} \frac{1}{2\widetilde{\lambda}}\left(p_t
# - \widetilde{\psi} \right) - \pi_{t-1}; \hspace{2mm} p_t \ge \widetilde{\psi} +
# 2\widetilde{\lambda} \pi_{t-1} \\ 0 \hspace{13mm}; \hspace{2mm} -\widetilde{\psi} +
# 2\widetilde{\lambda} \pi_{t-1} \le p_t \le \widetilde{\psi} + 2\widetilde{\lambda}
# \pi_{t-1} \\ \frac{1}{2\widetilde{\lambda}}\left( p_t + \widetilde{\psi} \right)-
# \pi_{t-1}; \hspace{2mm} p_t \le -\widetilde{\psi} + 2\widetilde{\lambda} \pi_{t-1}
# \end{cases} $$
#
# Parameters $\widetilde{\lambda}, \widetilde{\psi}$ can be found with a gridSearch.

# ***Environment***
#
# We set our environment with the follwing parameters:
# $$
# \begin{cases}
# \theta = 0.1 \\
# \sigma = 0.1 \\
# T = 5000 \\
# \lambda = 0.3 \\
# \psi = 4
# \end{cases}
# $$

# +
# Environment parameters

SIGMA = 0.1 #波动率
THETA = 0.1 #OU过程的参数
T = 5000 #时间长度
LAMBD = 0.3 #lambda系数
PSI = 4 #psi 交易的惩罚、或者收益
# -
# 初始化一个Environment
env = Environment(
    sigma=SIGMA, theta=THETA, T=T, lambd=LAMBD, psi=PSI, cost="trade_l1", scale_reward=1
)
#----------------------------------------------------------------------------------start ------------------------
### ***Optimal Solution*** 模式一: Grid,网格模式
# ***GridSearch*** 
# Grid,网格模式
#
# We perform a GridSearch to look for the optimal parameters $\widetilde{\lambda},
# \widetilde{\psi}$ using the average cumulative reward over $10$ episodes.
# 我们执行一个GridSearch来寻找最佳参数$\widetilde{\lambda}, \widetilde{\psi}$，使用10个episode的平均累积奖励。
# +
# %%time
random_state = 1024 #随机种子
n_episodes = 10 #episode的数量 
rng = np.random.RandomState(random_state) #随机种子生成器
random_states = rng.randint(0, int(1e6), size=n_episodes) #生成随机种子

lambds = np.linspace(0.2, 0.6, 10) #一组lambds。指定的间隔内返回均匀间隔的数字
psis = np.linspace(0.8, 1.2, 10) #一组psis，用于选举最佳参数

# 1st dim: lambdas; 2nd dim: psis; 3rd dim: episodes
#初始化一个收益聚合
scores_episodes = np.empty((len(lambds), len(psis), n_episodes))
scores = np.empty((len(lambds), len(psis)))  # 1st dim: lambdas; 2nd dim: psis

#遍历lambds
for i, lambd in tqdm_notebook(list(enumerate(lambds))):
    for j, psi in tqdm_notebook(list(enumerate(psis))):
        #模拟10个episode，test_apply返回收益
        score, score_episode, _, _, _ = env.test_apply(
            total_episodes=n_episodes, random_states=random_states, lambd=lambd, psi=psi
        )
        #填充收益集合
        scores[i, j] = score#收益
        scores_episodes[i, j, :] = list(score_episode.values())#收益集合
        # print('lambd=%.1f , psi=%.1f -> score=%.3f \n' % (lambd, psi, score))
# -
#作图
plt.figure(figsize=(8, 6))
sns.heatmap(#绘制热力图
    pd.DataFrame(
        scores,
        index=pd.Index(np.round(lambds, 2), name=r"$\widetilde{\lambda}$"),
        columns=pd.Index(np.round(psis, 2), name=r"$\widetilde{\psi}$"),
    )
)
plt.show()#显示图片

# +
i_max = np.argmax(scores) // scores.shape[0] #返回最大值的索引
j_max = np.argmax(scores[i_max, :]) #返回最大值的索引

lambd_max, psi_max = lambds[i_max], psis[j_max] #最大值的lambd和psi
print("lambd_max=%.2f , psi_max=%.2f" % (lambd_max, psi_max))

#过程：
#首先，通过创建一个 Agent 实例来初始化智能体，设置了智能体的一些参数，
#接下来，设置了训练过程中的一些参数
#调用 agent.train() 方法开始训练智能体。在训练过程中，智能体与环境进行交互，并根据交互经验来更新自己的参数。
#训练过程中会保存智能体的模型参数到指定的路径，以便后续测试和分析。
#调用 test_models() 方法对训练得到的智能体模型进行测试。测试过程中会加载之前保存的模型参数，并在环境中运行一定数量的测试周期，评估智能体的性能。

#进行一些额外的测试，比如在随机种子和一些特定的测试周期下运行模型，以评估其性能。
#最后，通过绘图和输出结果来展示训练和测试的结果，比如智能体的收益曲线、持仓情况等。

# We find that $\widetilde{\lambda}=0.47, \widetilde{\psi}=0.93$

# ## Training
#采用DDPG算法进行训练
# The agent is an actor critic architecture, both actor and critic networks are composed #
# of:
# - an inital fully connected layer of size $16$ with a reLU activation,#一个全连接层
# - a hidden fully connected layer of size $16$ with a reLU activation and #一个隐藏层和激活函数
# - a linear output layer of size $1$. #一个线性输出层
#
# agent使用其当前策略探索环境，添加的OU过程。参数为$\theta=1, \sigma=1$，将经验放入大小为$10^6$的回放缓冲区，
# 并且使用优先经验重放，每$50$个时间步采样大小为$512$的批次。
# The agent explores the environment with its current policy, an additive OU noise of
# parameters $\theta=1, \sigma=1$, it puts the experiences in a replay buffer of size
# $10^6$ and, we use Prioritized Experience Replay to sample a batch of size $512$ each
# $50$ time steps when we run a learning iteration.
#
# #agent是一个actor - critic结构，由actor网络和critic网络组成
# Before the training process starts, we run a pretraining phase and fill the replay
# buffer with $1000$ experiences generated by exploring the environment with the initial
# Actor network plus the additional OU noise. 
# #在训练过程开始之前，我们运行一个预训练阶段，
# 并使用初始Actor网络加上额外的OU噪声探索环境生成1000个经验填充回放缓冲区。

# +
# Agent parameters
MAX_STEPS = 5000 #最大步数
MEMORY_TYPE = "prioritized" #类型
SLIDING = "oldest" #滑动窗口
BATCH_SIZE = 2 ** 9 #步长
MAX_SIZE = int(1e6) #最大大小

# Training parameters
TOTAL_EPISODES = 200  # 总的训练周期数。每个周期包括从环境开始到结束的一系列交互步骤。
TOTAL_STEPS = 1000 # 每个训练周期中的总步数。每一步是智能体与环境交互的一次。
FREQ = 10 #频率 控制多久保存一次模型参数或输出训练信息的频率。例如，如果设置为10，则每10个周期保存一次模型参数或输出一次训练信息。
LEARN_FREQ = 50 #控制多久更新一次智能体的参数。例如，如果设置为50，则每50个步骤更新一次智能体的参数。
TAU_ACTOR = 0.3 #用于软更新（soft update）的时间常数，用于更新 Actor 网络的参数。软更新是一种渐进地将目标网络的参数更新为当前网络参数的方法。
TAU_CRITIC = 0.1 #用于软更新的时间常数，用于更新 Critic 网络的参数。。
LR_ACTOR = 1e-3 #Actor 网络的学习率。学习率控制了参数在每次更新时的调整大小。
LR_CRITIC = 1e-2 # Critic 网络的学习率。
WEIGHTS_DECAY_ACTOR = 0 #Actor 网络的权重衰减系数。权重衰减用于惩罚模型参数的大小，有助于防止过拟合。
WEIGHTS_DECAY_CRITIC = 0 #Critic 网络的权重衰减系数。
FC1_UNITS_ACTOR = 16 #Actor 网络中第一个全连接层的神经元数量。
FC2_UNITS_ACTOR = 16 #Actor 网络中第二个全连接层的神经元数量。
FC1_UNITS_CRITIC = 16 #Critic 网络中第一个全连接层的神经元数量。
FC2_UNITS_CRITIC = 16 #Critic 网络中第二个全连接层的神经元数量

############################
#这些参数在训练强化学习模型时非常重要，它们控制了训练的速度、稳定性和性能。通常情况下，
# 这些参数需要根据具体的问题和模型进行调整和优化，以获得最佳的训练结果。
##########################
# +
# %%time
#定义一个智能体，并且初始化对应的参数
agent = Agent(
    max_size=MAX_SIZE,
    max_step=MAX_STEPS,
    memory_type=MEMORY_TYPE,
    sliding=SLIDING,
    batch_size=BATCH_SIZE,
)
# 文件路径，用于测试成本 实验线性交易成本
path = "Experiment_linear_trading_cost_true_per/"
if not os.path.exists(path + "weights/"):
    os.makedirs(path + "weights/")
#发起训练
agent.train(
    env=env,
    total_episodes=TOTAL_EPISODES,
    tau_actor=TAU_ACTOR,
    tau_critic=TAU_CRITIC,
    lr_actor=LR_ACTOR,
    lr_critic=LR_CRITIC,
    weight_decay_actor=WEIGHTS_DECAY_ACTOR,
    weight_decay_critic=WEIGHTS_DECAY_CRITIC,
    total_steps=TOTAL_STEPS,
    weights=path + "weights/",
    freq=FREQ,
    fc1_units_actor=FC1_UNITS_ACTOR,
    fc2_units_actor=FC2_UNITS_ACTOR,
    fc1_units_critic=FC1_UNITS_CRITIC,
    fc2_units_critic=FC2_UNITS_CRITIC,
    learn_freq=LEARN_FREQ,
    plots=True,
    lambd=lambd_max,
    psi=psi_max,
    tensordir=path + "runs/", #tensorboard文件路径
    mile=100, #每100步保存一次模型
    decay_rate=1e-6, #学习率衰减率
)
# -

# ***Evaluation***
#
#在10个新的测试episode上评估模型，并将其与近似最优解进行比较。
# We evaluate our models and compare them with the approximate optimal solution on $10$
# new test episodes.

path_weights = path + "weights/"
#score_cumsum:累积分数
#pnls:收益
#positions:持仓
scores, scores_episodes, scores_cumsum, pnls, positions = test_models(#测试模型 
    path_weights, env, n_episodes=10, fc1_units=16, fc2_units=16
) #评估模型 

random_state = 1024 #
n_episodes = 10 #
rng = np.random.RandomState(random_state)
#random_stats:随机种子
random_states = rng.randint(0, int(1e6), size=n_episodes)
score, score_episode, scores_cumsum_opt, pnls_opt, positions_opt = env.test_apply(# 运行测试模型
    total_episodes=n_episodes, random_states=random_states, lambd=lambd_max, psi=psi_max
)
scores[-1] = score #最后一个收益分数
scores_episodes[-1] = score_episode #最后一个收益


# Given a predictor signal, we visualize both the positions taken by our agent and those
# taken by the approximate optimal solution, we find that the positions evolve very
# similarly .
#给定预测器信号，将智能体所占据的位置和那些位置可视化 采用近似最优解，发现位置的演变非常相似
#以下代码都是作图用，没有啥号看的，
# +
env.reset(random_state=730001) #重置环境
plt.figure(figsize=(15, 6)) #绘制图像

plt.subplot(1, 2, 1)
plt.plot(positions[110][730001], label="DDPG", color="g")
plt.plot(positions_opt[730001], label="OPT", color="r")
plt.plot(env.signal[1:], label="signal$", color="y")
plt.xlim(300, 600)
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$p_t, \pi_t$", fontsize=15)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(positions[110][730001], label="DDPG", color="g")
plt.plot(positions_opt[730001], label="OPT", color="r")
plt.xlim(300, 600)
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$\pi_t$", fontsize=15)
plt.legend()

plt.savefig("positions_penalty.png")
plt.show()
# -

plot_bars(scores) # plot_bars函数的作用：模型评估得分的条形图

sorted_scores = sorted(scores.values(), reverse=True)#排序 scores
print("Optimal agent score   : %.2f" % sorted_scores[0])
print("Best DDPG agent score : %.2f" % sorted_scores[1])

path_weights = path + "weights/"
models_keys = [0, 10, 50, 90, 110]
plot_function(
    path_weights,
    env,
    models_keys,
    low=-4,
    high=4,
    lambd=lambd_max,
    psi=psi_max,
    fc1_units=16,
    fc2_units=16,
)
#--------------------------------------------end-------------------------------------------------------
#--------------------------------------------start----------------------------------------------------
#线性交易成本与风险约束
# # Linear trading costs with risk constraint

# We can model the problem in two ways here: #；两种建模函数
#
# We keep the reward: $r(\pi_t, p_t) = \pi_t p_t - \psi | \pi_t - \pi_{t-1}|;\hspace{2mm}
# \lambda, \psi > 0$ and clip actions in the interval $\left[-M, M\right]$; this means
# that an agent does not get penalized for making action $a$ taking its position $\pi$
# beyond $M$, **i.e** $|\pi + a|>M$, since its next position $\pi'$ is s.t $|\pi'|=M$.
#
# The second way is harder, it does not involve clipping the positions but rather
# penalizes the agent for making actions taking its position outside the interval
# $\left[-M, M \right]$. We can do this by adding a smooth penalty to the reward, and we
# choose a $tanh$ barrier in the following way:
# $$ r(\pi_t, p_t) = \pi_t p_t - \psi | \pi_t - \pi_{t-1}| - \beta\left( tanh\left[
# \alpha\left( |\pi_t| - M - \gamma\right)\right] + 1\right);\hspace{2mm} \lambda, \psi,
# \alpha, \beta, \gamma > 0 $$
# This $tanh$ penalty is smooth and does not diverge for high $|\pi|$ values which makes
# it useful in training stabilization compared to a constant penalty or an exponential
# one.

# ***Optimal Solution***
#
# for simplicity we will consider an approximate optimal solution which shape is
# parameterized making this solution easily found by a simple grid-search.
#
# $$
# f\left(\pi_{t-1}, p_t \right) =
# \begin{cases}
# M - \pi_{t-1}; \hspace{2mm} p_t > \widetilde{q} \\
# 0 \hspace{13mm}; \hspace{2mm} |p_t| \le \widetilde{q} \\
# -M - \pi_{t-1}; \hspace{2mm} p_t < -\widetilde{q}
# \end{cases}
# $$
#
# Parameters $\widetilde{q}$ can be found with a gridSearch.

# ***Environment***
#
# First, we will work with the first setting and only use an experience generator to get
# some insight about how to properly explore and learn. Then we will work on both settings
# in the exploration-learning scheme training.
#
# We set our environment with the follwing parameters:
# $$
# \begin{cases}
# \theta = 0.1 \\
# \sigma = 0.1 \\
# T = 5000 \\
# M = 2 \\
# \psi = 4
# \end{cases}
# $$

# +
# Environment parameters


#第一种建模方式：
# 奖励函数定义为 $r(\pi_t, p_t) = \pi_t p_t - \psi | \pi_t - \pi_{t-1}|$。
# 这意味着智能体在每一步的奖励由当前持仓量 $\pi_t$ 与价格 $p_t$ 的乘积以及持仓量变化的绝对值的惩罚组成。
# 此外，动作（交易）被限制在区间 $[-M, M]$ 内，这意味着智能体不会受到在该区间外进行交易的惩罚。
SIGMA = 0.1
THETA = 0.1
T = 5000
MAXPOS = 2
PSI = 4.0
# -

env = Environment(
    sigma=SIGMA,
    theta=THETA,
    T=T,
    psi=PSI,
    cost="trade_l1",
    squared_risk=False,
    max_pos=MAXPOS,
    clip=True,
    penalty="tanh",
)

alpha, beta, gamma = 10, 10, MAXPOS / 4

#tanh_penalty函数的作用：tanh平滑惩罚障碍
def tanh_penalty(pi): 
    # tanh平滑惩罚障碍
    return beta * (np.tanh(alpha * (abs(pi) - MAXPOS - gamma)) + 1)

plt.plot(np.linspace(0, 5), tanh_penalty(np.linspace(0, 5)), label="tanh_penalty")
plt.axvline(MAXPOS, color="r", label="MAXPOS")
plt.title("tanh smooth penalty barrier", fontsize=15)
plt.legend()
plt.savefig("tanh_barrier.png")
plt.show()

# +
# Agent parameters
MAX_STEPS = 5000
MEMORY_TYPE = "prioritized"
SLIDING = "oldest"
BATCH_SIZE = 2 ** 9 # batch size 批处理大小
MAX_SIZE = int(1e6)

# Training parameters
TOTAL_EPISODES = 200  # set it to 501 for better convergence
TOTAL_STEPS = 10000
FREQ = 10
LEARN_FREQ = 50
TAU_ACTOR = 0.3
TAU_CRITIC = 0.1
LR_ACTOR = 1e-3
LR_CRITIC = 1e-2
WEIGHTS_DECAY_ACTOR = 0
WEIGHTS_DECAY_CRITIC = 0
FC1_UNITS_ACTOR = 16
FC2_UNITS_ACTOR = 16
FC1_UNITS_CRITIC = 16
FC2_UNITS_CRITIC = 16

# +
# %%time
agent = Agent(
    max_size=MAX_SIZE,
    max_step=MAX_STEPS,
    memory_type=MEMORY_TYPE,
    sliding=SLIDING,
    batch_size=BATCH_SIZE,
    alpha=0.6,
    theta=0.1,
)

path = "Experiment_maxpos/"
if not os.path.exists(path + "weights/"):
    os.makedirs(path + "weights/")

agent.train(
    env=env,
    total_episodes=TOTAL_EPISODES,
    tau_actor=TAU_ACTOR,
    tau_critic=TAU_CRITIC,
    lr_actor=LR_ACTOR,
    lr_critic=LR_CRITIC,
    weight_decay_actor=WEIGHTS_DECAY_ACTOR,
    weight_decay_critic=WEIGHTS_DECAY_CRITIC,
    total_steps=TOTAL_STEPS,
    weights=path + "weights/",
    freq=FREQ,
    fc1_units_actor=FC1_UNITS_ACTOR,
    fc2_units_actor=FC2_UNITS_ACTOR,
    fc1_units_critic=FC1_UNITS_CRITIC,
    fc2_units_critic=FC2_UNITS_CRITIC,
    learn_freq=LEARN_FREQ,
    plots=True,
    thresh=0.95,
    tensordir=path + "runs/",
    mile=100,
    decay_rate=1e-5,
)

# -

path_weights = path + "weights/"
scores, scores_episodes, scores_cumsum, pnls, positions = test_models(#定义测试模型
    path_weights, env, n_episodes=10, fc1_units=16, fc2_units=16
)

random_state = 1024
n_episodes = 10
rng = np.random.RandomState(random_state)
random_states = rng.randint(0, int(1e6), size=n_episodes)
score, score_episode, scores_cumsum_opt, pnls_opt, positions_opt = env.test_apply( #测试交易
    total_episodes=n_episodes, random_states=random_states, thresh=0.95
)
scores[-1] = score
scores_episodes[-1] = score_episode

plot_bars(scores) #收益曲线图
#作图
models_keys = [0, 20, 80, 120, 140]
plot_function( #绘制函数 path_weights
    path_weights,
    env,
    models_keys,
    low=-4,
    high=4,
    lambd=0.3,
    fc1_units=16,
    fc2_units=16,
    thresh=0.95,
    clip=True,
)

# +
env.reset(random_state=989115)
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(positions[140][989115], label="DDPG", color="g")
plt.plot(positions_opt[989115], label="OPT", color="r")
plt.plot(env.signal[1:], label="signal$", color="y")
plt.xlim(0, 300)
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$p_t, \pi_t$", fontsize=15)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(positions[140][989115], label="DDPG", color="g")
plt.plot(positions_opt[989115], label="OPT", color="r")
plt.xlim(0, 300)
plt.ylim(-4, 4)
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$\pi_t$", fontsize=15)
plt.legend()

plt.savefig("positions_constraint_per.png")
plt.show()
#------------------------------------------------模型三：影响成本（impact cost）的方形成本模型 ----------------------------------
# -
#影响成本
#在这种成本模型中，除了之前提到的奖励函数外，还引入了一个影响成本（impact cost）的方形成本模型。让我来解释一下：
#影响成本是指在交易市场中买卖资产所带来的价格波动。它是由于交易量大、市场流动性低等因素导致的。影响成本的存在使得买卖交易不再是零和游戏，因为交易一方的操作可能会影响市场价格，从而影响另一方的收益。
#这种成本模型中，考虑了一个方形的影响成本模型。具体来说，成本模型定义为：
#𝐶(𝜋𝑡,𝑎𝑡)=𝑐⋅𝑎𝑡2C(πt ,at )=c⋅at2
#其中：$C(\pi_t, a_t)$ 是在持仓量为 $\pi_t$ 时执行动作 $a_t$ 所产生的成本。$c$ 是影响成本的系数，表示每单位交易量的成本。
#这种成本模型表明，当智能体执行一个动作时，所产生的成本与动作的平方成正比。这意味着，动作的大小对成本的影响是非线性的，而且随着动作的增加而增加。
#引入影响成本模型可以更真实地模拟金融市场中的交易成本，并且有助于智能体学习更优化的交易策略，考虑到成本的影响。
# # Squared impact model

# We consider the following cost model incorporating a squared impact cost:
#
# $$
# c(\pi_t, p_t) = -\pi_t p_t + \lambda \pi_t^2 + \phi \left( \pi_t -
# \pi_{t-1}\right)^2;\hspace{2mm} \lambda, \phi > 0 #影响成本模型的定义
# $$
#
# Given the predictor signal $\left( p_t\right)_t$ the optimal position $\pi_t$ at time
# $t$ has the following form
# $$
# \pi_t = b\times EMA_a\left( p, t\right)
# $$
# Where $EMA_a\left( p, t\right)$ denotes the exponential moving average of preditor
# signal $p$ at time $t$ with a decay of $0 < a \le 1$ and $b > 0$
#
# We can now perform a grid-search over parameters $a$ and $b$

# ***Environment***
#
# We set our environment with the follwing parameters: #设置环境参数
# $$
# \begin{cases}
# \theta = 0.1 \\
# \sigma = 0.1 \\
# T = 5000 \\
# \lambda = 0.3 \\
# \phi = 1
# \end{cases}
# $$

# +
# Environment parameters

SIGMA = 0.1
THETA = 0.1
T = 5000
LAMBD = 0.3
PSI = 1
# -
#初始化一个Environment
env = Environment(
    sigma=SIGMA,
    theta=THETA,
    T=T,
    lambd=LAMBD,
    psi=PSI,
    cost="trade_l2",
    scale_reward=10,
)

# +
# Agent parameters
MAX_STEPS = 5000
MEMORY_TYPE = "prioritized"
SLIDING = "oldest"
BATCH_SIZE = 2 ** 9
MAX_SIZE = int(1e6)

# Training parameters
TOTAL_EPISODES = 200  # set it to 501 for better convergence
TOTAL_STEPS = 10000
FREQ = 10
LEARN_FREQ = 50
TAU_ACTOR = 0.3
TAU_CRITIC = 0.1
LR_ACTOR = 1e-3
LR_CRITIC = 1e-2
WEIGHTS_DECAY_ACTOR = 0
WEIGHTS_DECAY_CRITIC = 0
FC1_UNITS_ACTOR = 16
FC2_UNITS_ACTOR = 16
FC1_UNITS_CRITIC = 16
FC2_UNITS_CRITIC = 16

# +
# %%time
#设置squared_cost 路径
path = "Experiment_squared_cost/"
if not os.path.exists(path + "weights/"):
    os.makedirs(path + "weights/")

agent = Agent(
    max_size=MAX_SIZE,
    max_step=MAX_STEPS,
    memory_type=MEMORY_TYPE,
    sliding=SLIDING,
    batch_size=BATCH_SIZE,
    alpha=0.6,
    theta=0.1,
)
# agent = Agent(max_size = MAX_SIZE, max_step=MAX_STEPS, memory_type=MEMORY_TYPE,
# sliding=SLIDING, batch_size=BATCH_SIZE, alpha=0.6, point_max=2, n_points=5)
agent.train(
    env=env,
    total_episodes=TOTAL_EPISODES,
    tau_actor=TAU_ACTOR,
    tau_critic=TAU_CRITIC,
    lr_actor=LR_ACTOR,
    lr_critic=LR_CRITIC,
    weight_decay_actor=WEIGHTS_DECAY_ACTOR,
    weight_decay_critic=WEIGHTS_DECAY_CRITIC,
    total_steps=TOTAL_STEPS,
    weights=path + "weights/",
    freq=FREQ,
    fc1_units_actor=FC1_UNITS_ACTOR,
    fc2_units_actor=FC2_UNITS_ACTOR,
    fc1_units_critic=FC1_UNITS_CRITIC,
    fc2_units_critic=FC2_UNITS_CRITIC,
    learn_freq=LEARN_FREQ,
    plots=True,
    lambd=0.33,
    psi=0.47,
    tensordir=path + "runs/",
    mile=100,
    decay_rate=1e-5,
)
# -

random_state = 1024
n_episodes = 10
rng = np.random.RandomState(random_state)
random_states = rng.randint(0, int(1e6), size=n_episodes)
score, score_episode, scores_cumsum_opt, pnls_opt, positions_opt = env.test_apply(
    total_episodes=n_episodes, random_states=random_states, lambd=0.33, psi=0.47
)

path_weights = path + "weights/"
scores, scores_episodes, scores_cumsum, pnls, positions = test_models(
    path_weights, env, n_episodes=10, fc1_units=16, fc2_units=16
)

random_state = 1024
n_episodes = 10
rng = np.random.RandomState(random_state)
random_states = rng.randint(0, int(1e6), size=n_episodes)
score, score_episode, scores_cumsum_opt, pnls_opt, positions_opt = env.test_apply(
    total_episodes=n_episodes, random_states=random_states, thresh=0.95
)
scores[-1] = score
scores_episodes[-1] = score_episode

plot_bars(scores)

models_keys = [0, 40, 60, 100, 120]
plot_function(
    path_weights,
    env,
    models_keys,
    low=-4,
    high=4,
    lambd=0.3,
    fc1_units=16,
    fc2_units=16,
    thresh=0.95,
    clip=True,
)
