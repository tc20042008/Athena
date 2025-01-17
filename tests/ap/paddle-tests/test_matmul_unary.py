# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import unittest

import numpy as np
import utils

import paddle
from paddle.static import InputSpec


def trivial_matrix_unary(x, y):
    out = paddle.matmul(x, y)
    return paddle.scale(out, scale=0.1)


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = trivial_matrix_unary

    def forward(self, x, y):
        out = self.fn(x, y)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [65536, 128]
        self.x = paddle.randn(self.x_shape, dtype="float16")
        self.x.stop_gradient = False

        self.y_shape = [128, 32]
        self.y = paddle.randn(self.y_shape, dtype="float16")
        self.y.stop_gradient = False

    def eval_symbolic(self, use_cinn, profile):
        net = CINNSubGraphNet()
        input_spec = [
            InputSpec(shape=self.x_shape, dtype='float16'),
            InputSpec(shape=self.y_shape, dtype='float16')
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y)
        if profile:
            # warmup
            for i in range(10):
                out = net(self.x, self.y)

            # repeat
            paddle.base.core.nvprof_start()
            for i in range(1000):
                out = net(self.x, self.y)
            paddle.base.core.nvprof_stop()
        return out

    def test_eval_symbolic(self):
        profile = False
        cinn_out = self.eval_symbolic(use_cinn=True, profile=profile)
        dy_out = self.eval_symbolic(use_cinn=False, profile=profile)
        if not profile:
            np.testing.assert_allclose(
                cinn_out.numpy(), dy_out.numpy(), rtol=1e-02, atol=1e-02
            )


if __name__ == '__main__':
    unittest.main()
