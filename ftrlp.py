#!/usr/bin/env python

"""
Pytorch Implementation of FTRL-proximal Algorithm

References:
    Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization, H. B. Mcmahan. AISTATS 2011.
    Ad Click Prediction: a View from the Trenches, SIGKDD2013.
"""

from torch.optim import Optimizer

class FTRLP(Optimizer):
    r"""
    FTRL-proximal algorithm is an online convex optimization algorithm proposed in 
    "Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization, H. B. Mcmahan. AISTATS 2011."

    Here we consider single-layer generalized linear model \hat{y} = g(x^T w)
    and arbitrary loss l(y, \hat{y}), in which case \nabla_w l = (d l)/(d \hat{y}) g'x
    Args:
        params (iterable) : iterable of parameters to optimize or dicts defining parameter groups
        alpha (float, optional): constant to determine lr
        beta (float, optional): constant to determine lr
        l1 (float, optional): l1 regularization parameter
        l2 (float, optional): l2 regularization parameter
    """
    def __init__(self, params, alpha = 0.02, beta = 1., l1 = 1., l2 = 0.):
        defaults = dict(alpha = alpha, beta = beta, l1 = l1, l2 = l2) 
        # TODO: only consider single-layer model, assertion on params
        super(FTRLP, self).__init__(params, defaults)

    def step(self, closure = None):
        """ perfroms a single optimization step
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss
        """
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # state initialization 
                if len(state) == 0:
                    state['step'] = 0
                    state['n'] = p.new().resize_as_(p).zero_() # square of gradient
                    state['z'] = p.new().resize_as_(p).zero_() # z_t = g_{1:t} - \sum_1^t \sigma_s w_s
                grad_squa, zt = state['n'], state['z']
                state['step'] += 1

                # perform per-coordinate lr update
                # TODO: sparse update, donnot know how to access input non-zero index
                p.data = -1. / (group['l2'] + (group['beta'] + grad_squa.sqrt()) /  \
                        group['alpha']) * (zt - zt.sign() * group['l1'])
                p.data[zt.abs() <= group['l1']] = 0.
        
        loss = None
        
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                grad_squa, zt  = state['n'], state['z']

                sigma = (grad_squa.addcmul(1., grad, grad).sqrt() - grad_squa.sqrt()) / group['alpha']
                zt.add_(grad - sigma * p.data)
                grad_squa.addcmul_(1., grad, grad)
        return loss

if __name__ == '__main__':
    main()
