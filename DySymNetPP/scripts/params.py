from .functions import *


class Params:
    """Symbolic Network architecture parameters"""
    # 取样过程的备选算符 Optional operators during sampling
    funcs_avail = [Identity(), 
                   Sin(), 
                   Cos(), 
                   Tan(),
                   Exp(), 
                   Log(), 
                   Sqrt(),
                   Square(), 
                   Pow(3),
                   Pow(4),
                   Pow(5),
                   Pow(6),
                   Plus(), 
                   Sub(), 
                   Product(), 
                   Div()
                   ]
    n_layers = [2, 3, 4, 5] # 备选网络层数 Optional network layer numbers
    num_func_layer = [2, 3, 4, 5, 6] # 备选每层算符数 Optional number of operators per layer
    
    """Symbolic Network training parameters"""
    learning_rate2 = 1e-2
    reg_weight = 5e-3
    threshold = 0.05
    trials = 1 # 训练失败后重试次数 Trial number of symbolic network training
    n_epochs1 = 10001
    n_epochs2 = 10001
    summary_step = 1000
    clip_grad = True # 是否使用梯度裁剪 Clip gradients or not?
    max_norm = 1 # 法向梯度裁剪的阈值 Maximum norm threshold for gradient clipping
    window_size = 50 # 适应性梯度裁剪的窗口大小 Window size for adaptive gradient clipping
    refine_constants = True # 是否优化表达式常数（BFGS）Optimize expression constants or not(BFGS)
    n_restarts = 1 # BFGS常数优化重启次数 Number of restarts for BFGS constant optimization
    add_bias = False # 是否加和偏置项 Add bias term or not?
    verbose = True # 是否打印训练过程信息 Print training information or not?
    use_gpu = False # 是否使用cuda-gpu Use GPU or not?
    plot_reward = False # 是否打印表达式奖励函数值 Print expression reward function value or not?
    
    """Controller.py参数 Reinforcement Learning Training Parameters"""
    num_epochs = 500
    batch_size = 10
    if isinstance(n_layers, list) or isinstance(num_func_layer, list):
        input_size = max(len(n_layers), len(num_func_layer))
    else:
        input_size = len(funcs_avail)
    optimizer = 'Adam'
    hidden_size = 32
    embedding_size = 16
    learning_rate1 = 0.0006
    risk_seeking = True
    risk_factor = 0.5
    entropy_weight = 0.005
    reward_type = 'mse' # mse nrmse
    
    """数据集参数 Dataset Parameters"""
    N_TRAIN = 100 # 训练集大小 Number of training samples
    N_VAL = 100 # 验证集大小 Number of validation samples
    NOISE = 0 # 训练集噪声水平标准误差 Standard deviation of training dataset noise level
    DOMAIN = (-1, 1)  # Domain of dataset - range from which we sample x. Default (-1, 1)
    # DOMAIN = np.array([[0, -1, -1], [1, 1, 1]]) # Use this format if each input variable has a different domain
    # domain[0]第一个向量为所有自变量定义域下限， domain[1]第二个向量为所有自变量定义域上限
    # 当原始函数有logarithmic terms时，需要手动设置训练和测试集定义域范围，否则无法正确计算loss和mse
    N_TEST = 100 # 测试集大小 Number of test samples
    DOMAIN_TEST = (-2, 2) # 测试集域范围 Domain of test dataset - should be larger than training domain to test extrapolation. Default (-2, 2)
    var_names = [f'x_{i}' for i in range(1, 21)]
    
    """结果保存目录 Result saving directory"""
    results_dir = './results/test'
