import paddle
"""Contains the symbolic regression neural network architecture."""
import numpy as np
from . import functions as functions


class SymbolicLayer(paddle.nn.Layer):
    """Neural network layer for symbolic regression where activation functions correspond to primitive functions.
    Can take multi-input activation functions (like multiplication)"""

    def __init__(self, funcs=None, initial_weight=None, init_stddev=0.1,
        in_dim=None, add_bias=False):
        """
        funcs: List of activation functions, using utils.functions
        initial_weight: (Optional) Initial value for weight matrix
        variable: Boolean of whether initial_weight is a variable or not
        init_stddev: (Optional) if initial_weight isn't passed in, this is standard deviation of initial weight
        """
        super().__init__()
        if funcs is None:
            funcs = functions.default_func
        self.initial_weight = initial_weight
        self.W = None
        self.built = False
        self.add_bias = add_bias
        self.output = None
        self.n_funcs = len(funcs)
        self.funcs = [func.paddle for func in funcs]
        self.n_double = functions.count_double(funcs)
        self.n_single = self.n_funcs - self.n_double
        self.out_dim = self.n_funcs + self.n_double
        if self.initial_weight is not None:
            self.W = paddle.base.framework.EagerParamBase.from_tensor(tensor
                =self.initial_weight.clone().detach())
            self.built = True
        else:
            self.W = paddle.base.framework.EagerParamBase.from_tensor(tensor
                =paddle.mod(x=paddle.normal(mean=0.0, std=init_stddev,
                shape=(in_dim, self.out_dim)), y=paddle.to_tensor(2, dtype=
                paddle.normal(mean=0.0, std=init_stddev, shape=(in_dim,
                self.out_dim)).dtype)))
            if add_bias:
                self.b = paddle.base.framework.EagerParamBase.from_tensor(
                    tensor=paddle.mod(x=paddle.normal(mean=0.0, std=
                    init_stddev, shape=(1, self.out_dim)), y=paddle.
                    to_tensor(2, dtype=paddle.normal(mean=0.0, std=
                    init_stddev, shape=(1, self.out_dim)).dtype)))

    def forward(self, x):
        """Multiply by weight matrix and apply activation units"""
        if self.add_bias:
            g = paddle.matmul(x=x, y=self.W) + self.b
        else:
            g = paddle.matmul(x=x, y=self.W)
        output = [] #self.output = paddle.empty([0])#
        in_i = 0
        out_i = 0
        while out_i < self.n_single:
            output.append(self.funcs[out_i](g[:, in_i]))
            in_i += 1
            out_i += 1
        while out_i < self.n_funcs:
            output.append(self.funcs[out_i](g[:, in_i], g[:, in_i + 1]))
            in_i += 2
            out_i += 1
        self.output = paddle.stack(x=output, axis=1)
        return self.output

    def get_weight(self):
        return self.W.cpu().detach().numpy()

    def get_bias(self):
        return self.b.cpu().detach().numpy()

    def get_weight_tensor(self):
        return self.W.clone()


class SymbolicNet(paddle.nn.Layer):
    """Symbolic regression network with multiple layers. Produces one output."""

    def __init__(self, symbolic_depth, x_dim, funcs=None, initial_weights=
        None, init_stddev=0.1, add_bias=False):
        super(SymbolicNet, self).__init__()
        self.depth = symbolic_depth
        self.funcs = funcs
        self.add_bias = add_bias
        layer_in_dim = [x_dim] + [len(funcs[i + 1]) for i in range(self.depth)]
        if initial_weights is not None:
            layers = [SymbolicLayer(funcs=funcs[i + 1], initial_weight=
                initial_weights[i], in_dim=layer_in_dim[i], add_bias=self.
                add_bias) for i in range(self.depth)]
            self.output_weight = (paddle.base.framework.EagerParamBase.
                from_tensor(tensor=initial_weights[-1].clone().detach()))
        else:
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayer(funcs=self.funcs[i + 1], init_stddev=
                init_stddev[i], in_dim=layer_in_dim[i], add_bias=self.
                add_bias) for i in range(self.depth)]
            self.output_weight = (paddle.base.framework.EagerParamBase.
                from_tensor(tensor=paddle.rand(shape=(layers[-1].n_funcs, 1))))
            if add_bias:
                self.output_bias = (paddle.base.framework.EagerParamBase.
                    from_tensor(tensor=paddle.rand(shape=(1, 1))))
        self.hidden_layers = paddle.nn.Sequential(*layers)

    def forward(self, input):
        h = self.hidden_layers(input)
        return paddle.matmul(x=h, y=self.output_weight)

    def get_weights(self):
        """Return list of weight matrices"""
        return [self.hidden_layers[i].get_weight() for i in range(self.depth)
            ] + [self.output_weight.cpu().detach().numpy()]

    def get_biases(self):
        return [self.hidden_layers[i].get_bias() for i in range(self.depth)
            ] + [self.output_bias.cpu().detach().numpy()]

    def get_weights_tensor(self):
        """Return list of weight matrices as tensors"""
        return [self.hidden_layers[i].get_weight_tensor() for i in range(
            self.depth)] + [self.output_weight.clone()]


class SymbolicLayerL0(SymbolicLayer):

    def __init__(self, in_dim=None, funcs=None, initial_weight=None,
        init_stddev=0.1, bias=False, droprate_init=0.5, lamba=1.0, beta=2 /
        3, gamma=-0.1, zeta=1.1, epsilon=1e-06):
        super().__init__(in_dim=in_dim, funcs=funcs, initial_weight=
            initial_weight, init_stddev=init_stddev)
        self.droprate_init = droprate_init if droprate_init != 0 else 0.5
        self.use_bias = bias
        self.lamba = lamba
        self.bias = None
        self.in_dim = in_dim
        self.eps = None
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.epsilon = epsilon
        if self.use_bias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(tensor
                =0.1 * paddle.ones(shape=(1, self.out_dim)))
        self.qz_log_alpha = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.normal(mean=np.log(1 - self.droprate_init) - np.
            log(self.droprate_init), std=0.01, shape=(in_dim, self.out_dim)))

    def quantile_concrete(self, u):
        """Quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = paddle.nn.functional.sigmoid(x=(paddle.log(x=u) - paddle.log(x=
            1.0 - u) + self.qz_log_alpha) / self.beta)
        return y * (self.zeta - self.gamma) + self.gamma

    def sample_u(self, shape, reuse_u=False):
        """Uniform random numbers for concrete distribution"""
        if self.eps is None or not reuse_u:
            device = str('cuda' if paddle.device.cuda.device_count() >= 1 else
                'cpu').replace('cuda', 'gpu')
            self.eps = paddle.rand(shape=shape).to(device) * (1 - 2 * self.
                epsilon) + self.epsilon
        return self.eps

    def sample_z(self, batch_size, sample=True):
        """Use the hard concrete distribution as described in https://arxiv.org/abs/1712.01312"""
        if sample:
            eps = self.sample_u((batch_size, self.in_dim, self.out_dim))
            z = self.quantile_concrete(eps)
            return paddle.clip(x=z, min=0, max=1)
        else:
            pi = paddle.nn.functional.sigmoid(x=self.qz_log_alpha)
            return paddle.clip(x=pi * (self.zeta - self.gamma) + self.gamma,
                min=0.0, max=1.0)

    def get_z_mean(self):
        """Mean of the hard concrete distribution"""
        pi = paddle.nn.functional.sigmoid(x=self.qz_log_alpha)
        return paddle.clip(x=pi * (self.zeta - self.gamma) + self.gamma,
            min=0.0, max=1.0)

    def sample_weights(self, reuse_u=False):
        z = self.quantile_concrete(self.sample_u((self.in_dim, self.out_dim
            ), reuse_u=reuse_u))
        mask = paddle.clip(x=z, min=0.0, max=1.0)
        return mask * self.W

    def get_weight(self):
        """Deterministic value of weight based on mean of z"""
        return self.W * self.get_z_mean()

    def loss(self):
        """Regularization loss term"""
        return paddle.sum(x=paddle.nn.functional.sigmoid(x=self.
            qz_log_alpha - self.beta * np.log(-self.gamma / self.zeta)))

    def forward(self, x, sample=True, reuse_u=False):
        """Multiply by weight matrix and apply activation units"""
        if sample:
            h = paddle.matmul(x=x, y=self.sample_weights(reuse_u=reuse_u))
        else:
            w = self.get_weight()
            h = paddle.matmul(x=x, y=w)
        if self.use_bias:
            h = h + self.bias
        output = []
        in_i = 0
        out_i = 0
        while out_i < self.n_single:
            output.append(self.funcs[out_i](h[:, in_i]))
            in_i += 1
            out_i += 1
        while out_i < self.n_funcs:
            output.append(self.funcs[out_i](h[:, in_i], h[:, in_i + 1]))
            in_i += 2
            out_i += 1
        output = paddle.stack(x=output, axis=1)
        return output


class SymbolicNetL0(paddle.nn.Layer):
    """Symbolic regression network with multiple layers. Produces one output."""

    def __init__(self, symbolic_depth, in_dim=1, funcs=None,
        initial_weights=None, init_stddev=0.1):
        super(SymbolicNetL0, self).__init__()
        self.depth = symbolic_depth
        self.funcs = funcs
        layer_in_dim = [in_dim] + self.depth * [len(funcs)]
        if initial_weights is not None:
            layers = [SymbolicLayerL0(funcs=funcs, initial_weight=
                initial_weights[i], in_dim=layer_in_dim[i]) for i in range(
                self.depth)]
            self.output_weight = (paddle.base.framework.EagerParamBase.
                from_tensor(tensor=initial_weights[-1].clone().detach()))
        else:
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayerL0(funcs=funcs, init_stddev=init_stddev[
                i], in_dim=layer_in_dim[i]) for i in range(self.depth)]
            self.output_weight = (paddle.base.framework.EagerParamBase.
                from_tensor(tensor=paddle.rand(shape=(self.hidden_layers[-1
                ].n_funcs, 1)) * 2))
        self.hidden_layers = paddle.nn.Sequential(*layers)

    def forward(self, input, sample=True, reuse_u=False):
        h = input
        for i in range(self.depth):
            h = self.hidden_layers[i](h, sample=sample, reuse_u=reuse_u)
        h = paddle.matmul(x=h, y=self.output_weight)
        return h

    def get_loss(self):
        return paddle.sum(x=paddle.stack(x=[self.hidden_layers[i].loss() for
            i in range(self.depth)]))

    def get_weights(self):
        """Return list of weight matrices"""
        return [self.hidden_layers[i].get_weight().cpu().detach().numpy() for
            i in range(self.depth)] + [self.output_weight.cpu().detach().
            numpy()]
