import paddle

class LSTMCell(paddle.nn.LSTMCell):
    def forward(self, inputs, states = None):
        return super().forward(inputs, states)[1]
