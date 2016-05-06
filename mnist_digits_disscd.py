"""
This program trains on every digit of MNIST separately (digits from 0 to 9).
Data used for untraining is that of every other digit i.e.
if the program is train on digit i then it is untrained on all j != i where
j is in [0:10)
"""
import theano.tensor as T
import theano
import gzip
from rbm import test_rbm
import pickle
import numpy


if __name__=="__main__":

    with gzip.open('mnist.pkl.gz', 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    mnist_x, data_y = train_set
    test_x, test_y = test_set

    for i in range(10):

        res = test_rbm(dataset = mnist_x[data_y == i],
                    neg_dataset = mnist_x[data_y != i],
                    learning_rate = 0.1,
                    training_epochs = 10,
                    batch_size = 20,
                    output_folder = "disscd_digit_%d" % i,
                    n_hidden = 500,
                    k = 10,
                    pcd=False)

        input = T.matrix('input')
        en = res.get_energy(input)
        ef = theano.function(inputs=[input], outputs=[en])


        persistent_vis_chain = theano.shared(
            numpy.asarray(
                test_x[test_y == i],
                dtype=theano.config.floatX
            )
        )

        plot_every = 5
        # plot every defines the number of steps between each sampling
        (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            res.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=plot_every,
            name="gibbs_vhv"
        )


        # update our persistent chain with the last calculated result
        updates.update({persistent_vis_chain: vis_samples[-1]})

        # mean squared error calculation
        mse = T.sum((persistent_vis_chain - T.nnet.sigmoid(presig_vis[-1]))**2, axis=1)
        reconstruction_cost = mse
        error_var = T.var(mse)

        # returns matrix of reconstruction cost, and error variance of mean squared error
        sample_fn = theano.function(
            [],
            [
                reconstruction_cost,
                error_var
            ],
            updates=updates,
            name='sample_fn'
        )

        # computing reconstruction error and reconstruction variance
        # for test data.
        re,rv = sample_fn()

        # calculating energy measure for test data
        val = numpy.asarray(ef(test_x[test_y == i]))
        #print("shape of energy : ", val.shape)
        print("Energy of digit %d is : " %i, numpy.mean(val))


        #print("shape of reconstruction : ", re.shape)
        print("Reconstruction error of %d digit: " % i, numpy.mean(re))
        print("Reconstruction variance of %d digit: " % i, rv)
