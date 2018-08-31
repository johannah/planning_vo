import torch
import numpy as np
import tensorflow as tf
from rudder_utils import TriangularValueEncoding as PTTriangularValueEncoding
from TeLL.utility.misc_tensorflow import TriangularValueEncoding
from IPython import embed
sess = tf.InteractiveSession()

def run_tse(step=3,mt=40,spl=10):
    tspan = int(mt/spl)
    tte = TriangularValueEncoding(max_value=mt, triangle_span=tspan)
    tfext = tte.encode_value(tf.constant(step, dtype=tf.int32))
    print(tfext.eval())

    pte = PTTriangularValueEncoding(max_value=mt, triangle_span=tspan)
    dval = torch.tensor(step).int()
    ptext = pte.encode_value(dval)
    print(ptext.numpy())
    print('---------')


def test_tse():
    run_tse(44,40,10)
    run_tse(4,40,10)
    run_tse(0,40,10)
    run_tse(10,40,10)
    run_tse(44,70,3)
    run_tse(4,70,3)
    run_tse(0,70,3)
    run_tse(10,70,3)


test_tse()
sess.close()
