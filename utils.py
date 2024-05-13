import numpy as np


#创建一个ou 信号量
# Ornstein-Uhlenbeck（奥恩斯坦-乌伦贝克）
def build_ou_process(T=100000, theta=0.1, sigma=0.1, random_state=None):
    """
    Description
    ---------------
    Build a discrete OU process signal of length T starting at p_0=0:
    ```
    #OU过程表达为离散形式 
    #OU过程有下面的随机微分方程 d x t = θ ( μ − x t ) d t + σ d W t dx_t = \theta(\mu - x_t)dt + \sigma dW_tdx t =θ(μ−x t )dt+σdW
    #可以理解为深度学习算法必要得一个过程，用于探索数据的特征
        p_t - p_{t-1} = -theta*p_{t-1} + sigma*epsilon_t;
    ```
    (epsilon_t)_t are standard normal random variables


    Parameters:
    ---------------
    T : Int, length of the signal. #信号量长度
    theta : Float>0, parameter of the OU process. #OU过程的参数
    sigma : Float>0, parameter of the OU process. #OU过程的参数
    random_state : None or Int, if Int, generate the same sequence of noise each time. #随机种子

    Returns
    ---------------
    np.array of shape (T,), the OU signal generated. #返回一个，OU信号量
    """
    X = np.empty(T)
    t = 0
    x = 0.0
    if random_state is not None:#如果随机种子不为空
        rng = np.random.RandomState(random_state)
        normals = rng.normal(0, 1, T)

    else:#如果随机种子为空
        normals = np.random.normal(0, 1, T)

    for t in range(T):#遍历T
        x += -x * theta + sigma * normals[t]
        X[t] = x
    X /= sigma * np.sqrt(1.0 / 2.0 / theta)#归一化
    return X

#计算收益率
def get_returns(signal, random_state=None):
    """
    Description
    ---------------
    Compute the returns r_t = p_t + eta_t, where p_t is the signal and eta_t is a Gaussian
    white noise.

    Parameters
    ---------------
    signal : 1D np.array, the signal computed as a sample path of an OU process.#信号量
    random_state : Int or None:
        - if None, do not use a random state (useful to simulate different paths each time
          running the simulation).#如果没有随机种子，不使用随机种子
        - if Int, use a random state (useful to compare different experimental results).#如果有随机种子，使用随机种子

    Returns
    ---------------
    1D np.array containing the returns
    """

    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return signal + rng.normal(size=signal.size)

    else:
        return signal + np.random.normal(size=signal.size)
