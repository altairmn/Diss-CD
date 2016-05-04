from rbm import RBM
import theano.tensor as T
import theano

# energy can be calculated though


def energy(rbm, v_sample):
   pre_sigmoid_h, h_mean, h_sample = rbm.sample_h_given_v(v_sample)
   vbias_term = T.dot(v_sample, rbm.v_bias)
   hbias_term = T.dot(h_sample, rbm.h_bias)
   weight_term = T.sum(T.dot(v_sample, wts)*h_sample, axis=1)
   return -(vbias_term + hbias_term + weight_term)


def reconstruction_error(rbm, v_sample):
    _, _, chain_start = rbm.sample_h_given_v(v_sample)

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
        rbm.gibbs_hvh,
        outputs_info=[None, None, None, None, None, chain_start],
        n_steps=k,
        name="gibbs_hvh"
        )

    mse = T.sum((v_sample - T.nnet.sigmoid(pre_sigmoid_nvs[-1]))**2, axis=1)
    reconstruction_cost = T.mean(mse)
    error_var = T.var(mse)

    return [reconstruction_cost, error_var]





if __name__=="__main__":
    pass


