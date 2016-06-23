"""
Code largely taken from http://deeplearning.net/tutorial/code/rbm.py
It has been modified and cleaned for use in this project.

Original comments have been removed. Only comments by author of the DISS-CD project have been retained
for project evaluation.
"""

from __future__ import print_function

import timeit

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images

import pickle
import gzip

import getopt, sys
import inspect

from itertools import cycle

class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        neg_input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        The constructor function of the RBM is used to initialize an RBM.
        All variables are symbolic.

        :param neg_input: negative input training set
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized from the Xavier's distribution i.e
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        # shared variables are created due to usage in several points in theano computation graph
        if hbias is None:
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input and neg_input

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.neg_input = neg_input
        if not neg_input:
            self.neg_input = T.matrix('neg_input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng

        # easily accessible model parameters of the RBM
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        """
        Free energy computation simplifies a lot of
        the gradient expressions.
        Since, theano implements symbolic differentiation,
        free energy is used to express variables
        """
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        '''
        Implements one step of gibbs sampling. Requires initialization of the
        hidden state.
        '''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        '''
        Implements one step of gibbs sampling. Requires initialization of the visible
        state.
        '''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of Diss-CD-k or PCD-k
        Note that the program implements only two modes of training:
            * Persistent Contrastive Divergence (k step)
            * Dissimilar Contrastive Divergence (k step)
        Ordinary CD-k is not implemented in this program
        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # for DISS-CD, chain_starts at a dissimilar data point
        if persistent is None:
            _, _, chain_start = self.sample_h_given_v(self.neg_input)
        else:
            chain_start = persistent

        # perform actual negative phase
        # a gibbs step is performed k times to obtain the fantasy particle
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))

        # the gradient computation must not propagate back through the gibbs sampling steps.
        # thus, the chain_end (ending configuration) is specified as constant
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is implemented as a proxy for persistent contrastive divergence
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # cross entropy error is used to track progress in Diss-CD. Note that, final error values are computed using mean squared error.
            # Furthermore, due to unsupervised nature of training, the error value is of no consequence to the training method.
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        xi = T.round(self.input)

        fe_xi = self.free_energy(xi)

        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        fe_xi_flip = self.free_energy(xi_flip)

        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error
        Due to the way theano optimizes expressions, the
        pre_sigmoid_nv activations are required.
        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

    def get_energy(self, v_sample):
       """ This function defines a way to obtain
       energy of a given set of samples.
       """
       # If the normalizing constant for the space is assumed to be the same for two differnt RBMs,
       # then the energies can be compared to compare performance of the nets.
       pre_sigmoid_h, h_mean, h_sample = self.sample_h_given_v(v_sample)
       vbias_term = T.dot(v_sample, self.vbias)
       hbias_term = T.dot(h_sample, self.hbias)

       # row-wise inner product is taken between v*W and h
       weight_term = T.sum(T.dot(v_sample, self.W)*h_sample, axis=1)
       return -(vbias_term + hbias_term + weight_term)


    def get_mse(self, v_sample, k):
        """
        Computes mean squared error and variance of error.
        """

        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(v_sample)

        chain_start = ph_sample
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
            )

        mse = T.sum((v_sample - T.nnet.sigmoid(pre_sigmoid_nvs[-1]))**2, axis=1)
        reconstruction_cost = T.mean(mse)
        # variance of error is indicative of the spread of the probability around the initializing input.
        error_var = T.var(mse)

        return [reconstruction_cost, error_var]

def test_rbm(
        learning_rate=0.1,
        training_epochs=15,
        dataset=None,
        neg_dataset=None,
        batch_size=20,
        output_folder='rbm_plots',
        n_hidden=500,
        k=10,
        pcd=False):

    # snippet prints received arguments to track the working of the program.
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    print ('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        print ("    %s = %s" % (i, values[i]))

    """
    This is demonstrated on MNIST.

    :param pcd: if True, then persistent contrastive divergence is done using the input dataset
    and the neg_dataset is not used. This may illicit a warning from the program due to unused variable.

    """

    train_set_x = theano.shared(numpy.asarray(dataset, dtype=theano.config.floatX), borrow=False)
    neg_train_set_x = theano.shared(numpy.asarray(neg_dataset, dtype=theano.config.floatX), borrow=False)


    # compute number of minibatches of training set and negative training set
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_neg_train_batches = neg_train_set_x.get_value(borrow=True).shape[0] // batch_size
    print("Training batches : %d, Neg Training Batches : %d" % (n_train_batches, n_neg_train_batches))

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    neg_index = T.lscalar() # index to neg [mini] batch

    x = T.matrix('x')
    neg_x = T.matrix('neg_x')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain. In case pcd is True, persistent chain
    # is initialized else it is set to None
    if pcd:
        persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)
    else:
        persistent_chain = None

    # construct the RBM class
    rbm = RBM(input=x, neg_input=neg_x, n_visible=28 * 28,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of PCD-k or Diss-CD-k
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=k)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # this function does all the magic
    train_rbm = theano.function(
        [index, neg_index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            neg_x: neg_train_set_x[neg_index * batch_size: (neg_index + 1) * batch_size]
        },
        name='train_rbm',
        on_unused_input='warn'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()



    # store nets for sampling later.
    trained_nets = {}


    # the negative training set and the positive training set may be of
    # different sizes. An iterable range is obtained in the following snippet.
    # the smaller dataset is cycled over until the larger one is exhausted
    if n_train_batches > n_neg_train_batches:
        indices = list(zip(range(n_train_batches), cycle(range(n_neg_train_batches))))
    else:
        indices = list(zip(cycle(range(n_train_batches)), range(n_neg_train_batches)))

    # go through training epochs
    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []

        for batch_index, neg_batch_index  in indices:
            mean_cost += [train_rbm(batch_index, neg_batch_index)]

        net_params = {'weight': rbm.W.get_value(borrow=False),
                      'hbias': rbm.hbias.get_value(borrow=False),
                      'vbias': rbm.vbias.get_value(borrow=False)}

        trained_nets[epoch] = net_params

        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()

        # after each epoch the weight filters are plotted for visualization.
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    # the trained nets are stored here for later use to avoid retraining.
    fname = '%s_trained_nets.pkl' % output_folder
    with open(fname, 'wb') as fo:
        pickle.dump({'data': trained_nets}, fo)

    print ('Training took %f minutes' % (pretraining_time / 60.))

    os.chdir('../')

    # return the trained rbm.
    return rbm


def usage():

    frm = "{:30}{:<50}"
    print(frm.format("--dataset=DATASET.pkl.gz", "dataset used for training the RBM"))
    print(frm.format("--neg-dataset=DATASET.pkl.gz","dataset used for negative training"))
    print(frm.format("--epochs=N","Number of epochs of training"))
    print(frm.format("--lr=", "learning rate"))
    print(frm.format("--k=k", "number of gibbs samplings"))
    print(frm.format("--output-folder=OUTPUT-FOLDER", "Output folder for storage"))
    print(frm.format("--n-hidden=n", "number of hidden units"))
    print(frm.format("--batch-size=bs", "batch size"))
    print(frm.format("--pcd", "flag to execute PCD else DISS-CD"))
    print(frm.format("--custom", "execute custom code added to program"))
    print(frm.format("--help", "Display this message"))



# get input function parses command line input
def get_input():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "",
                ["dataset=",
                 "neg-dataset=",
                 "epochs=",
                 "lr=",
                 "k=",
                 "output-folder=",
                 "n-hidden=",
                 "batch-size=",
                 "pcd",
                 "custom",
                 "help"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    return dict([(x[2:], y) for x, y in opts])

if __name__ == '__main__':

    opts = get_input()

    # Trains on MNIST and untrains on CIFAR.

    with gzip.open(opts.get('dataset', 'mnist.pkl.gz'), 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    mnist_x, data_y = train_set

    with gzip.open(opts.get('neg-dataset', 'cifar_disscd.pkl.gz'), 'rb') as f:
        try:
            data_dict = pickle.load(f, encoding='latin1')
        except:
            data_dict = pickle.load(f)

    cifar_x = data_dict['data']

    if 'help' in opts:
        usage()
        sys.exit(0)


    if 'custom' not in opts:

        persistentcd = True if 'pcd' in opts else False

        test_rbm(dataset=mnist_x,
                neg_dataset=cifar_x,
                learning_rate=float(opts.get('lr', 0.1)),
                training_epochs=int(opts.get('epochs', 15)),
                batch_size=int(opts.get('batch-size', 20)),
                output_folder=opts.get('output-folder', 'rbm_plots'),
                n_hidden=int(opts.get('n-hidden', 500)),
                k=int(opts.get('k', 15)),
                pcd=persistentcd)

    else:

    ###############
    # CUSTOM CODE #
    ###############

    # Enable to custom flag from command line and put custom code below this line to run.


    ## Examples

    # Example1 : Training on MNIST and untraining on CIFAR

        test_rbm(dataset=mnist_x,
                neg_dataset=cifar_x,
                learning_rate=0.2,
                training_epochs=15,
                batch_size=20,
                output_folder='mnist_diss_cd_cifar',
                n_hidden=500,
                k=15)


    # Example 2: Training on digit i and untraining on other digits
        tr_digit = []
        # Neg Training on other digits
        for digit in range(10):
            this_x = mnist_x[data_y == digit]
            other_x =  mnist_x[data_y != digit]

            test_rbm(dataset=this_x,
                    neg_dataset=other_x,
                    learning_rate=0.1,
                    training_epochs=15,
                    batch_size=20,
                    output_folder='mnist_diss_cd_%d' % digit,
                    n_hidden=500,
                    k=10)

    # Example 3: Training on MNIST and untraining on random matrices (28 x 28)

        # Neg Training on random input
        rand_rng = numpy.random.RandomState(1234)
        random_x = rand_rng.uniform(low=0, high=1, size=(50000, 28, 28))

        test_rbm(dataset=mnist_x,
                neg_dataset=random_x,
                learning_rate=0.15,
                training_epochs=15,
                batch_size=20,
                output_folder='mnist_diss_cd_random',
                n_hidden=500,
                k=15)



