import theano.tensor as T
import theano
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Op:
    def __init__(self, input=None, op=None):
        self.op = op
        if op is None:
            self.op = theano.shared(name='op', value=None)


        self.input=input
        if input is None:
            self.input = theano.shared(name='input', value=None)


    def mse(self, input):
        return T.sum(T.pow(input - T.dot(input, self.op), 2))


class Scalar:
    def __init__(self, scale=1):
        self.scale = scale

    def mse(self, input):
        return T.pow(input-scale, 2)


if __name__=="__main__":
    llen = T.scalar('llen', dtype=theano.config.floatX)
    angle_of_rot = T.scalar('angle_of_rot', dtype=theano.config.floatX)
    steps = T.scalar('steps', dtype='int32')
    start_x = T.scalar('start_x', dtype=theano.config.floatX)
    start_y = T.scalar('start_y', dtype=theano.config.floatX)

    rot_x = start_x*T.cos(angle_of_rot) - start_y*T.sin(angle_of_rot)
    rot_y = start_x*T.sin(angle_of_rot) + start_y*T.cos(angle_of_rot)

    rotate = theano.function(
                name='rotate',
                inputs=[start_x, start_y, angle_of_rot],
                outputs=[rot_x, rot_y]
            )
