import paddle

class LSTMCell(paddle.nn.LSTMCell):
    def forward(self, inputs, states = None):
        return super().forward(inputs, states)[1]

def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm
