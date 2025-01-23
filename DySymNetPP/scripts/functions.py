"""Functions for use with symbolic regression.

These functions encapsulate multiple implementations (sympy, Tensorflow, numpy, paddle) of a particular function so that the
functions can be used in multiple contexts."""
import numpy as np
import sympy as sp
import paddle

class BaseFunction:
    """Abstract class for primitive functions"""

    def __init__(self, norm=1):
        self.norm = norm

    def sp(self, x):
        """Sympy implementation"""
        return None

    def paddle(self, x):
        """No need for base function"""
        return None

    def tf(self, x):
        """Automatically convert sympy to TensorFlow"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'tensorflow')(x)

    def np(self, x):
        """Automatically convert sympy to numpy"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)


class Constant(BaseFunction):

    def paddle(self, x):
        return paddle.ones_like(x=x)

    def sp(self, x):
        return 1

    def np(self, x):
        return np.ones_like


class Identity(BaseFunction):

    def __init__(self):
        super(Identity, self).__init__()
        self.name = 'id'

    def paddle(self, x):
        return x / self.norm

    def sp(self, x):
        return x / self.norm

    def np(self, x):
        return np.array(x) / self.norm


class Square(BaseFunction):

    def __init__(self):
        super(Square, self).__init__()
        self.name = 'pow2'

    def paddle(self, x):
        return paddle.square(x=x) / self.norm

    def sp(self, x):
        return x ** 2 / self.norm

    def np(self, x):
        return np.square(x) / self.norm


class Pow(BaseFunction):

    def __init__(self, power, norm=1):
        BaseFunction.__init__(self, norm=norm)
        self.power = power
        self.name = 'pow{}'.format(int(power))

    def paddle(self, x):
        return paddle.pow(x=x, y=self.power) / self.norm

    def sp(self, x):
        return x ** self.power / self.norm


class Sin(BaseFunction):

    def __init__(self):
        super().__init__()
        self.name = 'sin'

    def paddle(self, x):
        return paddle.sin(x=x) / self.norm

    def sp(self, x):
        return sp.sin(x) / self.norm


class Cos(BaseFunction):

    def __init__(self):
        super(Cos, self).__init__()
        self.name = 'cos'

    def paddle(self, x):
        return paddle.cos(x=x) / self.norm

    def sp(self, x):
        return sp.cos(x) / self.norm


class Tan(BaseFunction):

    def __init__(self):
        super(Tan, self).__init__()
        self.name = 'tan'

    def paddle(self, x):
        return paddle.tan(x=x) / self.norm

    def sp(self, x):
        return sp.tan(x) / self.norm


class Sigmoid(BaseFunction):

    def paddle(self, x):
        return paddle.nn.functional.sigmoid(x=x) / self.norm

    def sp(self, x):
        return 1 / (1 + sp.exp(-20 * x)) / self.norm # 为什么-20*x？

    def np(self, x):
        return 1 / (1 + np.exp(-20 * x)) / self.norm

    def name(self, x):
        return 'sigmoid(x)'


class Exp(BaseFunction):

    def __init__(self):
        super().__init__()
        self.name = 'exp'

    def paddle(self, x):
        return paddle.exp(x=x)

    def sp(self, x):
        return sp.exp(x)


class Log(BaseFunction):

    def __init__(self):
        super(Log, self).__init__()
        self.name = 'log'

    def paddle(self, x):
        return paddle.log(x=paddle.abs(x=x) + 1e-6) / self.norm

    def sp(self, x):
        return sp.log(sp.Abs(x) + 1e-6) / self.norm


class Sqrt(BaseFunction):

    def __init__(self):
        super(Sqrt, self).__init__()
        self.name = 'sqrt'

    def paddle(self, x):
        return paddle.sqrt(x=paddle.abs(x=x)) / self.norm

    def sp(self, x):
        return sp.sqrt(sp.Abs(x)) / self.norm


class BaseFunction2:
    """Abstract class for primitive functions with 2 inputs"""

    def __init__(self, norm=1.0):
        self.norm = norm

    def sp(self, x, y):
        """Sympy implementation"""
        return None

    def paddle(self, x, y):
        return None

    def tf(self, x, y):
        """Automatically convert sympy to TensorFlow"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'tensorflow')(x, y)

    def np(self, x, y):
        """Automatically convert sympy to numpy"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)


class Product(BaseFunction2):

    def __init__(self, norm=0.1):
        super().__init__(norm=norm)
        self.name = '*'

    def paddle(self, x, y):
        return x * y / self.norm

    def sp(self, x, y):
        return x * y / self.norm


class Plus(BaseFunction2):

    def __init__(self, norm=1.0):
        super().__init__(norm=norm)
        self.name = '+'

    def paddle(self, x, y):
        return (x + y) / self.norm

    def sp(self, x, y):
        return (x + y) / self.norm


class Sub(BaseFunction2):

    def __init__(self, norm=1.0):
        super().__init__(norm=norm)
        self.name = '-'

    def paddle(self, x, y):
        return (x - y) / self.norm

    def sp(self, x, y):
        return (x - y) / self.norm


class Div(BaseFunction2):

    def __init__(self):
        super(Div, self).__init__()
        self.name = '/'

    def paddle(self, x, y):
        return x / (y + 1e-06)

    def sp(self, x, y):
        return x / (y + 1e-06)


def count_inputs(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction):
            i += 1
        elif isinstance(func, BaseFunction2):
            i += 2
    return i


def count_double(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction2):
            i += 1
    return i


default_func = [Product(), Plus(), Sin()]
