import os
import paddle
import json
import time
import sympy as sp
import pandas as pd
from scipy.optimize import minimize
from .scripts.functions import *
from .scripts import functions as functions
import collections
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Float
from .scripts.controller import Agent
from .scripts import pretty_print
from .scripts.regularization import L12Smooth
from .scripts.symbolic_network import SymbolicNet
from sklearn.metrics import r2_score
from .scripts.params import Params
from .scripts.utils import nrmse, R_Square, MSE, Relative_Error


def generate_data(func, N, range_min, range_max):
    """Generates datasets."""
    free_symbols = sp.sympify(func).free_symbols
    x_dim = free_symbols.__len__()
    sp_expr = sp.lambdify(free_symbols, func)
    x = (range_max - range_min) * paddle.rand(shape=[N, x_dim]) + range_min
    x_np = x.numpy()
    y_np = [[sp_expr(*x_i)] for x_i in x_np]
    y = paddle.to_tensor(data=y_np)
    #y = paddle.to_tensor(data=[[sp_expr(*x_i)] for x_i in x])
    return x, y


class TimedFun:

    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError('Time is over.')
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value


class SymboliRegression:

    def __init__(self, config, func=None, func_name=None, data_path=None):
        """
        Args:
            config: All configs in the Params class, type: Params
            func: the function to be predicted, type: str
            func_name: the name of the function, type: str
            data_path: the path of the data, type: str
        """
        self.data_path = data_path
        self.X = None
        self.y = None
        self.funcs_per_layer = None
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.input_size = config.input_size # number of operators
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.n_layers = config.n_layers
        self.num_func_layer = config.num_func_layer
        self.funcs_avail = config.funcs_avail
        self.optimizer = config.optimizer
        self.auto = False
        self.add_bias = config.add_bias
        self.threshold = config.threshold
        
        self.clip_grad = config.clip_grad
        self.max_norm = config.max_norm
        self.window_size = config.window_size
        self.refine_constants = config.refine_constants
        self.n_restarts = config.n_restarts
        self.reward_type = config.reward_type
        
        if config.use_gpu:
            self.device = str('cuda').replace('cuda', 'gpu')
        else:
            self.device = paddle.CPUPlace()
        print('Use Device:', self.device)
        
        # Standard deviation of random distribution for weight initializations
        self.init_sd_first = 0.1
        self.init_sd_last = 1.0
        self.init_sd_middle = 0.5
        
        self.config = config
        
        self.func = func
        self.func_name = func_name
        
        # Generate data or load data from file
        if self.func is not None:
            # add noise
            if config.NOISE > 0:
                self.X, self.y = generate_data(func, self.config.N_TRAIN,
                    self.config.DOMAIN[0], self.config.DOMAIN[1])
                y_rms = paddle.sqrt(x=paddle.mean(x=self.y ** 2))
                scale = config.NOISE * y_rms
                self.y += paddle.empty(shape=[tuple(self.y.shape)[-1]]).normal_(mean=0, std=scale)
                self.x_test, self.y_test = generate_data(func, self.config.
                    N_TRAIN, range_min=self.config.DOMAIN_TEST[0],
                    range_max=self.config.DOMAIN_TEST[1])
                
            else:
                self.X, self.y = generate_data(func, self.config.N_TRAIN,
                    self.config.DOMAIN[0], self.config.DOMAIN[1])
                self.x_test, self.y_test = generate_data(func, self.config.
                    N_TRAIN, range_min=self.config.DOMAIN_TEST[0],
                    range_max=self.config.DOMAIN_TEST[1])
                
        else:
            self.X, self.y = self.load_data(self.data_path)
            self.x_test, self.y_test = self.X, self.y
            
        self.dtype = self.X.dtype # the data type determine the paramaeter type of the model
        
        if isinstance(self.n_layers, list) or isinstance(self.num_func_layer, list):
            print('*' * 25, 'Start Sampling...', '*' * 25 + '\n')
            self.auto = True
            
        self.agent = Agent(auto=self.auto, input_size=self.input_size,
            hidden_size=self.hidden_size, num_funcs_avail=len(self.
            funcs_avail), n_layers=self.n_layers, num_funcs_layer=self.
            num_func_layer, device=self.device, dtype=self.dtype)
        # self.agent = self.agent.to(self.dtype)
        
        if not os.path.exists(self.config.results_dir):
            os.makedirs(self.config.results_dir)
            
        func_dir = os.path.join(self.config.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)
        self.results_dir = func_dir
        
        self.now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        # save hyperparameters
        args = {'date': self.now_time, 
                'add_bias': config.add_bias,
                'train_domain': config.DOMAIN, 
                'test_domain': config.DOMAIN_TEST, 
                'num_epochs': config.num_epochs, 
                'batch_size':config.batch_size, 
                'input_size': config.input_size,
                'hidden_size': config.hidden_size, 
                'risk_factor': config.risk_factor, 
                'n_layers': config.n_layers, 
                'num_func_layer':config.num_func_layer, 
                'funcs_avail': str([func.name for func in config.funcs_avail]), 
                'init_sd_first': 0.1, 
                'init_sd_last': 1.0,
                'init_sd_middle': 0.5, 
                'noise_level': config.NOISE
                }
        with open(os.path.join(self.results_dir, 'args_{}.txt'.format(self.func_name)), 'a') as f:
            f.write(json.dumps(args))
            f.write('\n')
        f.close()

    def solve_environment(self):
        epoch_best_expressions = []
        epoch_best_rewards = []
        epoch_mean_rewards = []
        epoch_mean_r2 = []
        epoch_best_r2 = []
        epoch_best_relative_error = []
        epoch_mean_relative_error = []
        best_expression, best_performance, best_relative_error = None, float('-inf'), float('inf')
        early_stopping = False
        
        # log the expressions acquired in all epochs
        f1 = open(os.path.join(self.results_dir, 'eq_{}_all.txt'.format(self.func_name)), 'a')
        f1.write('\n{}\t\t{}\n'.format(self.now_time, self.func_name))
        f1.write('{}\t\tReward\t\tR2\t\tExpression\t\tnum_layers\t\tnum_funcs_layer\t\tfuncs_per_layer\n'.format(self.reward_type))
        
        # log the best expressions of each epoch
        f2 = open(os.path.join(self.results_dir, 'eq_{}_summary.txt'.format(self.func_name)), 'a')
        f2.write('\n{}\t\t{}\n'.format(self.now_time, self.func_name))
        f2.write('Epoch\t\tReward\t\tR2\t\tExpression\n')
        
        if self.optimizer == 'Adam':
            optimizer = paddle.optimizer.Adam(parameters=self.agent.parameters(), 
                                              learning_rate=self.config.learning_rate1,
                                              weight_decay=0.0)
        else:
            optimizer = paddle.optimizer.RMSProp(parameters=self.agent.parameters(), 
                                                 learning_rate=self.config.learning_rate1,
                                                 weight_decay=0.0, epsilon=1e-08, rho=0.99)
        IPS_all = []    
        for i in range(self.num_epochs):
            print('******************** Epoch {:02d} ********************'.format(i))
            expressions = []
            rewards = []
            r2 = []
            relative_error_list = []
            batch_log_probs = paddle.zeros(shape=[self.batch_size])
            batch_entropies = paddle.zeros(shape=[self.batch_size])
            
            j = 0
            while j < self.batch_size:
                (error, R2, eq, log_probs, entropies, num_layers,
                    num_func_layer, funcs_per_layer_name, IPS_temp) = self.play_episodes()
                IPS_all.append(IPS_temp)
                # play an episode
                # if the expression is invalid, resample the structure of symbolic network
                if 'x_1' not in str(eq) or eq is None:
                    R2 = 0.0
                if 'x_1' in str(eq) and self.refine_constants:
                    res = self.bfgs(eq, self.X, self.y, self.n_restarts)
                    eq = res['best expression']
                    R2 = res['R2']
                    error = res['error']
                    relative_error = res['relative error']
                else:
                    relative_error = 100
                    
                reward = 1 / (1 + error)
                print('Final expression: ', eq)
                print('Test R2: ', R2)
                print('Test error: ', error)
                print('Relative error: ', relative_error)
                print('Reward: ', reward)
                print('\n')
                
                f1.write('{:.8f}\t\t{:.8f}\t\t{:.8f}\t\t{:.8f}\t\t{}\t\t{}\t\t{}\t\t{}\n'
                        .format(error, relative_error, reward, R2, eq, num_layers, num_func_layer, funcs_per_layer_name))
                
                if R2 > 0.99:
                    print('~ Early Stopping Met ~')
                    print('Best expression: ', eq)
                    print('Best reward:     ', reward)
                    print(f'{self.config.reward_type} error:      ', error)
                    print('Relative error:  ', relative_error)
                    early_stopping = True
                    break
                
                batch_log_probs[j] = log_probs
                batch_entropies[j] = entropies
                expressions.append(eq)
                rewards.append(reward)
                r2.append(R2)
                relative_error_list.append(relative_error)
                j += 1
                
            if early_stopping:
                f2.write('{}\t\t{:.8f}\t\t{:.8f}\t\t{}\n'.format(i, reward,R2, eq))
                break
            
            # a batch expression
            ## reward
            rewards = paddle.to_tensor(data=rewards, place=self.device)
            best_epoch_expression = expressions[np.argmax(rewards.cpu())]
            epoch_best_expressions.append(best_epoch_expression)
            epoch_best_rewards.append(max(rewards).item())
            epoch_mean_rewards.append(paddle.mean(x=rewards).item())
            
            ## R2
            r2 = paddle.to_tensor(data=r2, place=self.device)
            best_r2_expression = expressions[np.argmax(r2.cpu())]
            epoch_best_r2.append(max(r2).item())
            epoch_mean_r2.append(paddle.mean(x=r2).item())
            
            epoch_best_relative_error.append(relative_error_list[np.argmax(r2.cpu())])
            
            # log the best expression of a batch
            f2.write('{}\t\t{:.8f}\t\t{:.8f}\t\t{:.8f}\t\t{}\n'
                     .format(i,relative_error_list[np.argmax(r2.cpu())], 
                             max(rewards).item(), max(r2).item(), best_r2_expression))
            
            # save the best expression from the beginning to now
            if max(r2) > best_performance:
                best_performance = max(r2)
                best_expression = best_r2_expression
                best_relative_error = min(epoch_best_relative_error)
                
            if self.config.risk_seeking:
                threshold = np.quantile(rewards.cpu(), self.config.risk_factor)
            indices_to_keep = paddle.to_tensor(data=[j for j in range(len(rewards)) if rewards[j] > threshold], place=self.device)
            if len(indices_to_keep) == 0:
                print('Threshold removes all expressions. Terminating.')
                break
            
            # Select corresponding subset of rewards, log_probs, and entropies
            sub_rewards = paddle.index_select(x=rewards, axis=0, index=indices_to_keep)
            sub_log_probs = paddle.index_select(x=batch_log_probs, axis=0,index=indices_to_keep)
            sub_entropies = paddle.index_select(x=batch_entropies, axis=0,index=indices_to_keep)
            
            # Compute risk seeking and entropy gradient
            risk_seeking_grad = paddle.sum(x=(sub_rewards - threshold) * sub_log_probs, axis=0)
            entropy_grad = paddle.sum(x=sub_entropies, axis=0)
            
            # Mean reduction and clip to limit exploding gradients
            risk_seeking_grad = paddle.clip(x=risk_seeking_grad / (self.
                config.risk_factor * len(sub_rewards)), min=-1e6, max=1e6)
            entropy_grad = self.config.entropy_weight * paddle.clip(x=
                entropy_grad / (self.config.risk_factor * len(sub_rewards)),
                min=-1e6, max=1e6)
            
            # Compute loss and update parameters
            loss = -1 * (risk_seeking_grad + entropy_grad)
            optimizer.clear_gradients(set_to_zero=False)
            loss.backward()
            optimizer.step()
            
        f1.close()
        f2.close()
        
        # Save the rewards
        f3 = open(os.path.join(self.results_dir, 'reward_{}_{}.txt'.format(
            self.func_name, self.now_time)), 'w')
        for i in range(len(epoch_mean_rewards)):
            f3.write('{} {:.8f}\n'.format(i + 1, epoch_mean_rewards[i]))
        f3.close()
        
        # Plot reward curve
        if self.config.plot_reward:
            plt.plot([(i + 1) for i in range(len(epoch_mean_rewards))],
                epoch_mean_rewards)
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.title('Reward over Time ' + self.now_time)
            plt.show()
            plt.savefig(os.path.join(self.results_dir, 'reward_{}_{}.png'.
                format(self.func_name, self.now_time)))
        IPS_filtered = [ips for ips in IPS_all if ips is not None]
        if early_stopping:
            return eq, R2, error, relative_error, np.array(IPS_filtered).mean()
        else:
            return best_expression, best_performance.item(), 1 / max(rewards).item() - 1, best_relative_error, np.array(IPS_filtered).mean()

    def bfgs(self, eq, X, y, n_restarts):
        variable = self.vars_name
        
        # Parse the expression and get all the constant terms
        expr = eq
        c = symbols('c0:10000') # Suppose we have at most n constants: c0 ... cn-1
        consts = list(expr.atoms(Float)) # Only count floating-point coefficients, do not consider power exponents 
        # (can we consider to optimize power terms ??? )
        consts_dict = {c[i]: const for i, const in enumerate(consts)} # map between c_i and unoptimized constants
        
        for c_i, val in consts_dict.items():
            expr = expr.subs(val, c_i)

        def loss(expr, X):
            diffs = []
            for i in range(tuple(X.shape)[0]):
                curr_expr = expr
                for idx, j in enumerate(variable):
                    curr_expr = sp.sympify(curr_expr).subs(j, X[i, idx])
                diff = curr_expr - y[i]
                diffs.append(diff)
            return np.mean(np.square(diffs))
        
        # Lists where all restarted will be appended
        F_loss = []
        RE_list = []
        R2_list = []
        consts_ = []
        funcs = []
        
        print('Constructing BFGS loss...')
        loss_func = loss(expr, X)
        
        for i in range(n_restarts):
            x0 = np.array(consts, dtype=float)
            s = list(consts_dict.keys())
            # BFGS optimization
            fun_timed = TimedFun(fun=sp.lambdify(s, loss_func, modules=['numpy']), stop_after=int(10000000000.0))
            if len(x0):
                minimize(fun_timed.fun, x0, method='BFGS')
                consts_.append(fun_timed.x)
            else:
                consts_.append([])
                
            final = expr
            for i in range(len(s)):
                final = sp.sympify(final).replace(s[i], fun_timed.x[i])
                
            funcs.append(final)
            
            values_np = {x: X[:, idx].numpy() for idx, x in enumerate(variable)}
            #values = {x: X[:, idx] for idx, x in enumerate(variable)}
            y_pred = sp.lambdify(variable, final)(**values_np)
            y_pred = paddle.to_tensor(y_pred)
            if isinstance(y_pred, float):
                print('y_pred is float: ', y_pred, type(y_pred))
                R2 = 0.0
                loss_eq = 10000
            else:
                y_pred = paddle.where(condition=paddle.isinf(x=y_pred), x=
                    10000.0, y=y_pred)
                y_pred = paddle.where(condition=y_pred.clone().detach() > 
                    10000.0, x=10000.0, y=y_pred)
                R2 = max(0.0, R_Square(y.squeeze(), y_pred))
                loss_eq = paddle.mean(x=paddle.square(x=y.squeeze() - y_pred)
                    ).item()
                relative_error = paddle.mean(x=paddle.abs(x=(y.squeeze() -
                    y_pred) / y.squeeze())).item()
            R2_list.append(R2)
            F_loss.append(loss_eq)
            RE_list.append(relative_error)
        best_R2_id = np.nanargmax(R2_list)
        best_consts = consts_[best_R2_id]
        best_expr = funcs[best_R2_id]
        best_R2 = R2_list[best_R2_id]
        best_error = F_loss[best_R2_id]
        best_re = RE_list[best_R2_id]
        
        return {'best expression': best_expr, 
                'constants': best_consts,
                'R2': best_R2, 
                'error': best_error, 
                'relative error': best_re}

    def play_episodes(self):
        ############################### Sample a symbolic network ##############################
        init_state = paddle.rand(shape=(1, self.input_size), dtype=self.dtype) # initialize the input state
        
        if self.auto:
            (num_layers, num_funcs_layer, action_index, log_probs, entropies
                ) = self.agent(init_state)
            self.n_layers = num_layers
            self.num_func_layer = num_funcs_layer
        else:
            action_index, log_probs, entropies = self.agent(init_state)
        self.funcs_per_layer = {}
        self.funcs_per_layer_name = {}
        
        for i in range(self.n_layers):
            layer_funcs_list = list()
            layer_funcs_list_name = list()
            for j in range(self.num_func_layer):
                layer_funcs_list.append(self.funcs_avail[action_index[i, j]])
                layer_funcs_list_name.append(self.funcs_avail[action_index[i, j]].name)
            self.funcs_per_layer.update({(i + 1): layer_funcs_list})
            self.funcs_per_layer_name.update({(i + 1): layer_funcs_list_name})
            
        # let binary functions follow unary functions
        for layer, funcs in self.funcs_per_layer.items():
            unary_funcs = [func for func in funcs if isinstance(func,
                BaseFunction)]
            binary_funcs = [func for func in funcs if isinstance(func,
                BaseFunction2)]
            sorted_funcs = unary_funcs + binary_funcs
            self.funcs_per_layer[layer] = sorted_funcs
            
        print('Operators of each layer obtained by sampling: ', self.funcs_per_layer_name)
        
        ############################### Start training ##############################
        error_test, r2_test, eq, IPS = self.train(self.config.trials)
        
        return (error_test, r2_test, eq, log_probs, entropies, 
                self.n_layers, self.num_func_layer, self.funcs_per_layer_name, IPS)

    def train(self, trials=1):
        """Train the network to find a given function"""
        
        data, target = self.X.to(self.device), self.y.to(self.device)
        test_data, test_target = self.x_test.to(self.device), self.y_test.to(self.device)
        
        self.x_dim = tuple(data.shape)[-1]
        
        self.vars_name = [f'x_{i}' for i in range(1, self.x_dim + 1)] # variable names
        
        width_per_layer = [len(f) for f in self.funcs_per_layer.values()]
        n_double_per_layer = [functions.count_double(f) for f in self.funcs_per_layer.values()]
        
        if self.auto:
            init_stddev = [self.init_sd_first] + [self.init_sd_middle] * (self.n_layers - 2) + [self.init_sd_last]
        
        # Arrays to keep track of various quantities as a function of epoch
        loss_list = [] # Total loss (MSE + regularization)
        error_list = [] # MSE
        reg_list = [] # Regulariztion
        error_test_list = [] # Test error mse
        r2_test_list = [] # Test R2
        
        error_test_final = []
        r2_test_final = []
        eq_list = []

        def log_grad_norm(net):
            sqsum = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    sqsum += (p.grad ** 2).sum().item()
            return np.sqrt(sqsum)
        
        retrain_num = 0
        trial = 0
        while 0 <= trial < trials:
            print('Training on function ' + self.func_name + ' Trial ' +
                str(trial + 1) + ' out of ' + str(trials))

            # reinitialize for each trial
            if self.auto:
                net = SymbolicNet(self.n_layers, 
                                  x_dim=self.x_dim, funcs=self.funcs_per_layer, 
                                  initial_weights=None, 
                                  init_stddev=init_stddev, 
                                  add_bias=self.add_bias).to(self.device)
            else:
                net = SymbolicNet(self.n_layers, 
                                  x_dim=self.x_dim, 
                                  funcs=self.funcs_per_layer, 
                                  initial_weights=[
                                    # kind of a hack for truncated normal distribution
                                    paddle.mod(x=paddle.normal(mean=0, std=self.init_sd_first, shape=(self.x_dim, width_per_layer[0] + n_double_per_layer[0])), 
                                    y=paddle.to_tensor(2, dtype=paddle.normal(mean=0,std=self.init_sd_first, shape=(self.x_dim, width_per_layer[0] + n_double_per_layer[0])).dtype)),
                                    # binary operator has two inputs
                                    paddle.mod(x=paddle.normal(mean=0, std=self.init_sd_middle, shape=(width_per_layer[0], width_per_layer[1] + n_double_per_layer[1])), 
                                    y=paddle.to_tensor(2, dtype=paddle.normal(mean=0, std=self.init_sd_middle, shape=(width_per_layer[0], width_per_layer[1] + n_double_per_layer[1])).dtype)),

                                    paddle.mod(x=paddle.normal(mean=0, std=self.init_sd_middle, shape=(width_per_layer[1], width_per_layer[2] + n_double_per_layer[2])), 
                                    y=paddle.to_tensor(2, dtype=paddle.normal(mean=0, std=self.init_sd_middle, shape=(width_per_layer[1], width_per_layer[2] + n_double_per_layer[2])).dtype)),
                                    paddle.mod(x=paddle.normal(mean=0, std=self.init_sd_last, shape=(width_per_layer[-1], 1)), 
                                    y=paddle.to_tensor(2, dtype=paddle.normal(mean=0, std=self.init_sd_last, shape=(width_per_layer[-1], 1)).dtype))
                                    ]).to(self.device)
            # net.to(self.dtype)
            loss_val = np.nan
            restart_flag = False
            while np.isnan(loss_val):
                # training restarts if gradients blow up
                criterion = paddle.nn.MSELoss()
                optimizer = paddle.optimizer.RMSProp(parameters=net.parameters(), 
                                                     learning_rate=self.config.learning_rate2,
                                                     rho=0.9, epsilon=1e-10, 
                                                     momentum=0.0, centered=False,
                                                     weight_decay=0.0)
                # adaptive learning rate
                lmbda = lambda epoch: 0.1
                tmp_lr = paddle.optimizer.lr.MultiplicativeDecay(lr_lambda=lmbda, learning_rate=optimizer.get_lr())
                optimizer.set_lr_scheduler(tmp_lr)
                scheduler = tmp_lr

                if self.clip_grad:
                    que = collections.deque()

                net.train() # Set the model to training mode

                start_time = time.time()

                # First stage of training, preceded by 0th warmup stage
                for epoch in range(self.config.n_epochs1 + 2000):
                    optimizer.clear_gradients(set_to_zero=False) # zero the parameters' gradient
                    outputs = net(data) # forward pass
                    regularization = L12Smooth(a=0.01)
                    mse_loss = criterion(outputs, target)

                    reg_loss = regularization(net.get_weights_tensor())
                    loss = mse_loss + self.config.reg_weight * reg_loss
                    loss.backward()

                    if self.clip_grad:
                        grad_norm = log_grad_norm(net)
                        que.append(grad_norm)
                        if len(que) > self.window_size:
                            que.popleft()
                            clip_threshold = 0.1 * sum(que) / len(que)
                            paddle.nn.utils.clip_grad_norm_(parameters=net.parameters(), 
                                                            max_norm=clip_threshold,
                                                            norm_type=2)
                        else:
                            paddle.nn.utils.clip_grad_norm_(parameters=net.parameters(), 
                                                            max_norm=self.max_norm,
                                                            norm_type=2)
                    optimizer.step()

                    # Summary
                    if epoch % self.config.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)
                        with paddle.no_grad():
                            test_outputs = net(test_data) # [num_points, 1] same as test_target
                            if self.reward_type == 'mse':
                                test_loss = paddle.nn.functional.mse_loss(input=test_outputs, label=test_target)
                            elif self.reward_type == 'nrmse':
                                test_loss = nrmse(test_target, test_outputs)
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)
                            test_outputs = paddle.where(condition=paddle.isnan(x=test_outputs), 
                                                        x=paddle.full_like(x=test_outputs, fill_value=100), 
                                                        y=test_outputs)
                            r2 = R_Square(test_target, test_outputs)
                            r2_test_list.append(r2)

                        if self.config.verbose:
                            print('Epoch: {}\tTotal training loss: {}\tTest {}: {}'
                                  .format(epoch, loss_val, self.reward_type,error_test_val))
                        if np.isnan(loss_val) or loss_val > 1000: # if loss goes to nan, restart training
                            restart_flag = True
                            break

                    if epoch == 2000:
                        scheduler.step() # lr /= 10

                end_time = time.time()
                epoch1_avetime = (end_time - start_time)/(self.config.n_epochs1 + 2000)
                IPS1 = epoch1_avetime/test_target.shape[0]
                if restart_flag:
                    break

                scheduler.step() # lr /= 10

                start_time = time.time()

                for epoch in range(self.config.n_epochs2):
                    optimizer.clear_gradients(set_to_zero=False)
                    outputs = net(data)
                    regularization = L12Smooth(a=0.01)
                    mse_loss = criterion(outputs, target)
                    reg_loss = regularization(net.get_weights_tensor())
                    loss = mse_loss + self.config.reg_weight * reg_loss
                    loss.backward()
                    if self.clip_grad:
                        grad_norm = log_grad_norm(net)
                        que.append(grad_norm)
                        if len(que) > self.window_size:
                            que.popleft()
                            clip_threshold = 0.1 * sum(que) / len(que)
                            paddle.nn.utils.clip_grad_norm_(parameters=net.parameters(), 
                                                            max_norm=clip_threshold,
                                                            norm_type=2)
                        else:
                            paddle.nn.utils.clip_grad_norm_(parameters=net.parameters(), 
                                                            max_norm=self.max_norm,
                                                            norm_type=2)
                    optimizer.step()

                    if epoch % self.config.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)
                        with paddle.no_grad():
                            test_outputs = net(test_data)
                            if self.reward_type == 'mse':
                                test_loss = paddle.nn.functional.mse_loss(input
                                    =test_outputs, label=test_target)
                            elif self.reward_type == 'nrmse':
                                test_loss = nrmse(test_target, test_outputs)
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)
                            test_outputs = paddle.where(condition=paddle.
                                isnan(x=test_outputs), x=paddle.full_like(x
                                =test_outputs, fill_value=100), y=test_outputs)
                            r2 = R_Square(test_target, test_outputs)
                            r2_test_list.append(r2)
                        if self.config.verbose:
                            print(
                                'Epoch: {}\tTotal training loss: {}\tTest {}: {}'
                                .format(epoch, loss_val, self.reward_type,
                                error_test_val))
                        if np.isnan(loss_val) or loss_val > 1000:
                            break
                end_time = time.time()
                epoch2_avetime = (end_time - start_time)/self.config.n_epochs2
                IPS2 = epoch2_avetime/test_target.shape[0]

                
                print('Epoch1 time per sample: {} seconds/sample\nEpoch2 : {} seconds/sample. N_sampel: {}'
                          .format(IPS1, IPS2, test_target.shape[0]))
             
            if restart_flag:
                retrain_num += 1
                if retrain_num == 5:
                    return 10000, None, None, None
                continue

            # After the training, the symbolic expression is translated into an expression by pruning
            with paddle.no_grad():
                weights = net.get_weights()
                if self.add_bias:
                    biases = net.get_biases()
                else:
                    biases = None
                expr = pretty_print.network(weights, self.funcs_per_layer,self.vars_name, self.threshold, self.add_bias, biases)

            # Results of the training trials
            error_test_final.append(error_test_list[-1])
            r2_test_final.append(r2_test_list[-1])
            eq_list.append(expr)

            trial += 1

        error_expr_sorted = sorted(zip(error_test_final, r2_test_final, eq_list), key=lambda x: x[0])
        print('error_expr_sorted', error_expr_sorted)
        return error_expr_sorted[0][0], error_expr_sorted[0][1], error_expr_sorted[0][2], (IPS1 + IPS2)/2

    def load_data(self, path):
        data = pd.read_csv(path)
        if tuple(data.shape)[1] < 2:
            raise ValueError('CSV file must contain at least 2 columns.')
        x_data = data.iloc[:, :-1]
        y_data = data.iloc[:, -1:]
        X = paddle.to_tensor(data=x_data.values, dtype='float32')
        y = paddle.to_tensor(data=y_data.values, dtype='float32')
        return X, y


if __name__ == '__main__':
    # Configuration parameters
    config = Params()

    # Example 1: Input ground truth function, auto generate training data to extract the symbolic expression
    SR = SymboliRegression(config=config, func='x_1 + x_2', func_name='x_1+x_2')
    eq, R2, error, relative_error, IPS_ave = SR.solve_environment()
    print('Expression: ', eq)
    print('R2: ', R2)
    print('error: ', error)
    print('relative_error: ', relative_error)
    print('log(1 + MSE): ', np.log(1 + error))
    print('IPS average: ', IPS_ave)

    # Example 2: Input CSV file, use it to train and extract the symbolic expression
    SR = SymboliRegression(config=config, func_name='Nguyen-1', data_path='./data/Nguyen-1.csv', )
    eq, R2, error, relative_error, IPS_ave = SR.solve_environment()
    print('Expression: ', eq)
    print('R2: ', R2)
    print('error: ', error)
    print('relative_error: ', relative_error)
    print('log(1 + MSE): ', np.log(1 + error))
    print('IPS average: ', IPS_ave)
