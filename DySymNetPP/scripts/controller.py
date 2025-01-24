import sys
sys.path.append('/home/aistudio/DySymNetPP')
from .paddle_utils import *
import paddle



class Agent(paddle.nn.Layer):

    def __init__(self, auto, input_size, hidden_size, num_funcs_avail,
        n_layers, num_funcs_layer, device=None, dtype='float32'):
        super(Agent, self).__init__()
        self.auto = auto # 是否使用给定的网络结构
        self.num_funcs_avail = num_funcs_avail # 每层可用算符种类 Optional operator category per layer
        self.n_layers = n_layers # 可用网络层数 Optional number of layers
        self.num_funcs_layer = num_funcs_layer # 每层可用算符数量 Optional number of operators per layer
        self.dtype = dtype
        
        if device is not None:
            self.device = device
        else:
            self.device = 'cpu'
            
        if self.auto:
            self.n_layer_decoder = paddle.nn.Linear(in_features=hidden_size,
                out_features=len(self.n_layers))
            self.num_funcs_layer_decoder = paddle.nn.Linear(in_features=
                hidden_size, out_features=len(self.num_funcs_layer))
            self.max_input_size = max(len(self.n_layers), len(self.
                num_funcs_layer))
            self.dynamic_lstm_cell = LSTMCell(input_size=self.
                max_input_size, hidden_size=hidden_size)
            self.embedding = paddle.nn.Linear(in_features=self.
                num_funcs_avail, out_features=len(self.num_funcs_layer))
            
        self.lstm_cell = LSTMCell(input_size=input_size, hidden_size=
                                  hidden_size)
        self.decoder = paddle.nn.Linear(in_features=hidden_size,
            out_features=self.num_funcs_avail) # 输出概率分布 output probability distribution
        self.n_steps = n_layers
        self.hidden_size = hidden_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h_t = paddle.zeros(shape=[1, self.hidden_size], dtype=self.dtype)
        c_t = paddle.zeros(shape=[1, self.hidden_size], dtype=self.dtype)
        return h_t, c_t

    def forward(self, input):
        if self.auto:
            if tuple(input.shape)[-1] < self.max_input_size:
                input = paddle.nn.functional.pad(x=input, pad=(0, self.
                    max_input_size - tuple(input.shape)[0]), mode=
                    'constant', value=0, pad_from_left_axis=False)
            assert tuple(input.shape)[-1
                                      ] == self.max_input_size, 'Error: the input dim of the first step is not equal to the max dim'
            
            h_t, c_t = self.hidden
            
            # 1. 取样网络层数 First sample the number of layers
            h_t, c_t = self.dynamic_lstm_cell(input, (h_t, c_t)) # [batch_size, hidden_size]
            n_layer_logits = self.n_layer_decoder(h_t) # [batch_size, len(n_layers)]
            n_layer_probs = paddle.nn.functional.softmax(x=n_layer_logits,
                axis=-1)
            dist = paddle.distribution.Categorical(logits=n_layer_probs)
            # dist = paddle.distribution.Categorical(logits=paddle.log(n_layer_probs))
            # paddle.distribution.Categorical实际接受的是probs参数
            # dist = paddle.distribution.Categorical(logits=n_layer_logits)
            action_index1 = dist.sample([1])
            log_prob1 = dist.log_prob(action_index1)
            entropy1 = dist.entropy() # convert problem
            num_layers = self.n_layers[action_index1]
            
            # 2. 取样每层算符数量 Second sample the number of operators per layer
            input = n_layer_logits
            if tuple(input.shape)[-1] < self.max_input_size:
                input = paddle.nn.functional.pad(x=input, pad=(0, self.
                    max_input_size - tuple(input.shape)[-1]), mode=
                    'constant', value=0, pad_from_left_axis=False)
            h_t, c_t = self.dynamic_lstm_cell(input, (h_t, c_t))
            n_funcs_layer_logits = self.num_funcs_layer_decoder(h_t) # [batch_size, len(num_funcs_layer)]
            n_funcs_layer_probs = paddle.nn.functional.softmax(x=
                n_funcs_layer_logits, axis=-1)
            dist = paddle.distribution.Categorical(logits=n_funcs_layer_probs) # paddle.distribution.Categorical目前不支持probs参数输入，使用logits参数代替
            # dist = paddle.distribution.Categorical(logits=n_funcs_layer_logits)
            action_index2 = dist.sample([1])
            log_prob2 = dist.log_prob(action_index2)
            entropy2 = dist.entropy()
            num_funcs_layer = self.num_funcs_layer[action_index2]
            
            # 3. 取样每层算符 Third sample the operators per layer
            input = n_funcs_layer_logits
            if tuple(input.shape)[-1] < self.max_input_size:
                input = paddle.nn.functional.pad(x=input, pad=(0, self.
                    max_input_size - tuple(input.shape)[0]), mode=
                    'constant', value=0, pad_from_left_axis=False)
            outputs = []
            for t in range(num_layers):
                h_t, c_t = self.dynamic_lstm_cell(input, (h_t, c_t))
                output = self.decoder(h_t) # [batch_size, len(func_avail)]
                outputs.append(output)
                input = self.embedding(output)
                
            outputs = paddle.stack(x=outputs).squeeze(axis=1) # [n_layers, len(funcs)]
            probs = paddle.nn.functional.softmax(x=outputs, axis=-1)
            dist = paddle.distribution.Categorical(logits=probs) # paddle.distribution.Categorical目前不支持probs参数输入，使用logits参数代替
            action_index3 = dist.sample(shape=[num_funcs_layer,]).transpose([1, 0]) # [num_layers, num_func_layer]
            log_probs = dist.log_prob(action_index3)
            #log_probs = dist.log_prob(action_index3.transpose([1, 0])).transpose([1, 0]) # [num_layers, num_func_layer] compute the log probability of the action distribution
            entropies = dist.entropy() # convert problem 3  # [num_layers] compute the entropy of the action distribution
            log_probs, entropies = paddle.sum(x=log_probs), paddle.sum(x=
                entropies)
            log_probs = log_probs + log_prob1 + log_prob2
            entropies = entropies + entropy1 + entropy2
            
            return num_layers, num_funcs_layer, action_index3, log_probs, entropies
        
        # 网络层数与每层算符数量固定，只取样每层算符 
        # Fix the number of layers and the number of operators per layer,
        # only sample the operators, each layer is different
        else:
            outputs = []
            h_t, c_t = self.hidden
            for i in range(self.n_steps):
                h_t, c_t = self.lstm_cell(input, (h_t, c_t))
                output = self.decoder(h_t) # [batch_size, num_choices]
                outputs.append(output)
                input = output
                
            outputs = paddle.stack(x=outputs).squeeze(axis=1) # [num_steps, num_choices]
            probs = paddle.nn.functional.softmax(x=outputs, axis=-1)
            dist = paddle.distribution.Categorical(logits=probs) # paddle.distribution.Categorical目前不支持probs参数输入，使用logits参数代替
            action_index = dist.sample(shape=[self.num_funcs_layer,]).transpose([1, 0]) # [num_layers, num_func_layer]
            log_probs = dist.log_prob(action_index)
            #log_probs = dist.log_prob(action_index.transpose([1, 0])).transpose([1, 0]) # [num_layers, num_func_layer]
            entropies = dist.entropy() # [num_layers]
            log_probs, entropies = paddle.sum(x=log_probs), paddle.sum(x=
                entropies)
            return action_index, log_probs, entropies
