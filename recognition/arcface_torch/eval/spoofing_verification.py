"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import numpy as np
import torch
import scipy

@torch.no_grad()
def test(data_set, backbone, batch_size):
    print('testing verification..')
    data = data_set[0]
    liveness_list = data_set[1]
    time_consumed = 0.0
    probs = None
    ba = 0
    while ba < data.shape[0]:
        bb = min(ba + batch_size, data.shape[0])
        count = bb - ba
        _data = data[bb - batch_size: bb]
        time0 = datetime.datetime.now()
        img = ((_data / 255) - 0.5) / 0.5
        net_out: torch.Tensor = backbone(img)
        _, _probs = net_out
        _probs = _probs.detach().cpu().numpy()
        _probs = scipy.special.softmax(_probs, axis=1)
        time_now = datetime.datetime.now()
        diff = time_now - time0
        time_consumed += diff.total_seconds()
        if probs is None:
            probs = np.zeros((data.shape[0], _probs.shape[1]))
        probs[ba:bb, :] = _probs[(batch_size - count):, :]
        ba = bb

    positives = 0
    counter = 0
    for i in range(len(liveness_list)):
        if abs(liveness_list[i] - probs[i,1]) < 0.5:
            positives += 1
        counter += 1
    accuracy = float(positives) / counter
    return accuracy