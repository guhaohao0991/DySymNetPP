import paddle
"""Methods for regularization to produce sparse networks.

L2 regularization mostly penalizes the weight magnitudes without introducing sparsity.
L1 regularization promotes sparsity.
L1/2 promotes sparsity even more than L1. However, it can be difficult to train due to non-convexity and exploding
gradients close to 0. Thus, we introduce a smoothed L1/2 regularization to remove the exploding gradients."""


class L12Smooth(paddle.nn.Layer):

    def __init__(self, a):
        super(L12Smooth, self).__init__()
        self.a = a

    def forward(self, input_tensor):
        """input: predictions"""
        return self.l12_smooth(input_tensor, self.a)

    def l12_smooth(self, input_tensor, a=0.05):
        """Smoothed L1/2 norm"""
        if type(input_tensor) == list:
            return sum([self.l12_smooth(tensor) for tensor in input_tensor])
        smooth_abs = paddle.where(condition=paddle.abs(x=input_tensor) < a,
            x=paddle.pow(x=input_tensor, y=4) / (-8 * a ** 3) + paddle.
            square(x=input_tensor) * 3 / 4 / a + 3 * a / 8, y=paddle.abs(x=
            input_tensor))
        return paddle.sum(x=paddle.sqrt(x=smooth_abs))
