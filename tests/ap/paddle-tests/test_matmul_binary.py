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

import utils

import paddle
from paddle.static import InputSpec


def trivial_matrix_binary(x, y, b):
    out = paddle.matmul(x, y)
    return out + b


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = trivial_matrix_binary

    def forward(self, x, y, b):
        out = self.fn(x, y, b)
        return out


class TestAPMatmulBinary(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.dtype = "float32"

        self.x_shape = [256, 256]
        self.x = paddle.randn(self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False

        self.y_shape = [256, 512]
        self.y = paddle.randn(self.y_shape, dtype=self.dtype)
        self.y.stop_gradient = False

        self.b_shape = [256, 512]
        self.b = paddle.randn(self.b_shape, dtype=self.dtype)
        self.b.stop_gradient = False

    def eval_symbolic(self, use_cinn, profile):
        net = CINNSubGraphNet()
        input_spec = [
            InputSpec(shape=self.x_shape, dtype=self.dtype),
            InputSpec(shape=self.y_shape, dtype=self.dtype),
            InputSpec(shape=self.b_shape, dtype=self.dtype),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        with utils.profile_context(profile):
            out = net(self.x, self.y, self.b)
        return out

    def test_eval_symbolic(self):
        profile = False
        cinn_out = self.eval_symbolic(use_cinn=True, profile=profile)
        dy_out = self.eval_symbolic(use_cinn=False, profile=profile)
        if not profile:
            utils.check_result(self.dtype, cinn_out.numpy(), dy_out.numpy())


if __name__ == "__main__":
    unittest.main()
