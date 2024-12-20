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


def trivial_reduce_trivial(x):
    y = paddle.exp(x)
    z = y - x
    z = z + y
    z = z + y
    return z.sum()


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = trivial_reduce_trivial

    def forward(self, x):
        out = self.fn(x)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [64]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def test_eval_symbolic(self):
        pass


class TestCinnExpSubGraph(TestCinnSubGraphBase):
    def eval_symbolic(self, use_cinn):
        net = CINNSubGraphNet()
        input_spec = [InputSpec(shape=[None], dtype='float32')]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), rtol=1e-06, atol=1e-06
        )


if __name__ == '__main__':
    unittest.main()
