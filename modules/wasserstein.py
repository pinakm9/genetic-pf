import tensorflow as tf

def cost_matrix(x, y, p=2):
    "Returns the cost matrix C_{ij}=|x_i - y_j|^p"
    x_col = tf.expand_dims(x,1)
    y_lin = tf.expand_dims(y,0)
    c = tf.reduce_sum((tf.abs(x_col-y_lin))**p,axis=2)
    return c

def tf_round(x, decimals):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def sinkhorn_loss(x, y, x_weights=None, y_weights=None, epsilon=0.01, num_iters=200, p=2):
    """
    Description:
        Given two emprical measures with locations x and y
        outputs an approximation of the OT cost with regularization parameter epsilon
        num_iter is the max. number of steps in sinkhorn loop
    
    Args:
        x,y:  The input sets representing the empirical measures.  Each are a tensor of shape (n,D)
        x_weights, y_weights: weights for ensembles x and y
        epsilon:  The entropy weighting factor in the sinkhorn distance, epsilon -> 0 gets closer to the true wasserstein distance
        num_iters:  The number of iterations in the sinkhorn algorithm, more iterations yields a more accurate estimate
        p: p value used to define the cost in Wasserstein distance
    
    Returns:
        The optimal cost or the (Wasserstein distance) ** p
    """
    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y, p=p)  # Wasserstein cost function
    
    # both marginals are fixed with equal weights
    if x_weights is None:
        n = x.shape[0]
        x_weights = tf.constant(1.0/n,shape=[n])

    if y_weights is None:
        n = y.shape[0]
        y_weights = tf.constant(1.0/n,shape=[n])
    # Elementary operations
    def M(u,v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + tf.expand_dims(u,1) + tf.expand_dims(v,0) )/epsilon
    def lse(A):
        return tf.reduce_logsumexp(A,axis=1,keepdims=True)
    
    log_x_w = tf.math.log(x_weights)
    log_y_w = tf.math.log(y_weights)

    u, v = 0. * x_weights, 0. * y_weights
    for _ in range(num_iters):
        u = epsilon * (log_x_w - tf.squeeze(lse(M(u, v)) )  ) + u
        v = epsilon * (log_y_w - tf.squeeze( lse(tf.transpose(M(u, v))) ) ) + v
    
    pi = tf.exp(M(u, v))
    cost = tf.reduce_sum(pi*C)
    return cost


def sinkhorn_div_tf(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=200, p=2):
    c = cost_matrix(x, y, p=p)
 
    if alpha is None:
        alpha = tf.ones(x.shape[0], dtype=x.dtype) / x.shape[0]

    if beta is None:
        beta = tf.ones(y.shape[0], dtype=y.dtype) / y.shape[0]

    log_alpha = tf.expand_dims(tf.math.log(alpha), 1)
    log_beta = tf.math.log(beta)

    f, g = 0. * alpha, 0. * beta
    f_, iter = 1. * alpha, 0
    
    while (tf.norm(f - f_, ord=1) / tf.norm(f_, ord=1) > 1e-3) and iter < num_iters:
        f_ = 1.0 * f
        f = - epsilon * tf.reduce_logsumexp(log_beta + (g - c) / epsilon, axis=1)
        g = - epsilon * tf.reduce_logsumexp(log_alpha + (tf.expand_dims(f, 1) - c) / epsilon, axis=0)
        iter += 1
    #print('iteration count = {}'.format(iter))

    OT_alpha_beta = tf.reduce_sum(f * alpha) + tf.reduce_sum(g * beta)
    
    c = cost_matrix(x, x, p=p)
    f = 0. * alpha
    f_, iter = 1. * alpha, 0
    log_alpha = tf.squeeze(log_alpha)
    while tf.norm(f - f_, ord=1) / tf.norm(f_, ord=1) > 1e-3 and iter < num_iters:
        f_ = 1.0 * f
        f = 0.5 * (f - epsilon * tf.reduce_logsumexp(log_alpha + (f - c) / epsilon, axis=1) )
        iter += 1
    #print(iter)

    c = cost_matrix(y, y, p=p)
    g = 0. * beta
    g_, iter = 1. * beta, 0
    while tf.norm(g - g_, ord=1) / tf.norm(g_, ord=1) > 1e-3 and iter < num_iters:
        g_ = 1.0 * g
        g = 0.5 * (g - epsilon * tf.reduce_logsumexp(log_beta + (g - c) / epsilon, axis=1) )
        iter += 1
    
    d = tf_round(OT_alpha_beta - tf.reduce_sum(f * alpha) - tf.reduce_sum(g * beta), 5)
    #print(d**0.5)
    return d#tf_round(OT_alpha_beta - tf.reduce_sum(f * alpha) - tf.reduce_sum(g * beta), 5)

