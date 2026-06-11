import numpy as np
import matplotlib.pyplot as plt
import math

def gradient_descent(X, y, W_in, b_in, alpha, max_iters, cost_function, gradient_function):
    J_history = []
    p_history = []

    W = W_in.copy()
    b = b_in

    for i in range(max_iters):
        dJ_dW, dJ_db = gradient_function(X, y, W, b)

        W = W - alpha * dJ_dW
        b = b - alpha * dJ_db

        J_history.append(cost_function(X, y, W, b))
        p_history.append([W, b])
        
        if i % math.ceil(max_iters / 10) == 0:
            print(f"Iteration {i:4}\n",
                  f"Cost {J_history[-1]:8.2f}\n",
                  f"dj_dW: {dJ_dW}\n",
                  f"dJ_db: {dJ_db}\n",
                  f"w: {W}, b:{b}\n")
                  
    return W, b, J_history, p_history

def compute_cost(X, y, W, b):
    m = X.shape[0]
    f_wb = np.dot(X, W) + b
    
    cost = np.sum((f_wb - y) ** 2)
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(X, y, W, b):
    m = X.shape[0]

    err = (np.dot(X, W) + b) - y

    dJ_dW = np.dot(X.T, err)
    dJ_db = np.sum(err)
    
    return dJ_dW / m, dJ_db / m

def main():
    m = 200  
    n = 3    
    X = 5 * np.random.rand(m, n)

    true_W = np.array([2.5, -1.2, 3.8])
    true_b = 4.5 
    noise = np.random.randn(m) * 0.5 
    y = np.dot(X, true_W) + true_b + noise
    
    W_init = np.zeros_like(true_W)
    b_init = 0
    max_iters = 10000
    tmp_alpha = 1.0e-2

    W_final, b_final, J_hist, p_hist = gradient_descent(X, y, W_init, b_init, tmp_alpha, max_iters, compute_cost, compute_gradient)
    print(f"W found by gradient descent: {W_final}")
    print(f"b found by gradient descent: {b_final:1.4f}")

    fig1, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = True, figsize = (12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration")
    ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             
    ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')   
    ax2.set_xlabel('iteration step')
    plt.show()

if __name__ == "__main__":
    main()