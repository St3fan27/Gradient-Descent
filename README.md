# Optimization via Gradient Descent for Multiple Linear Regression

## Description
This project contains a from-scratch implementation of the **Gradient Descent** algorithm in Python, used for training a Multiple Linear Regression model. The script generates synthetic data, applies the optimization algorithm to find the ideal parameters (weights and bias), and visualizes the learning process (the decrease of the cost function).

## Features
* **Synthetic Data Generation**: Creates a test dataset with randomly added noise, using a predefined set of weights and a bias.
* **Cost Function Calculation**: Implements the Mean Squared Error (MSE) to evaluate the model's performance at each iteration.
* **Gradient Calculation**: Determines the partial derivatives of the cost function with respect to the model's parameters.
* **Optimization Loop**: Iteratively updates the model parameters to minimize the cost.
* **Visualization**: Uses `matplotlib` to plot the cost against the number of iterations, illustrating the algorithm's convergence.

---

## Mathematical Concepts

The project is based on the following fundamental equations:

**1. The Prediction Model:**
For an input with multiple features, the prediction is calculated as the dot product between the feature vector and the weights vector, plus the bias:
$$f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

**2. The Cost Function (J):**
To evaluate the model's accuracy on a dataset with $m$ examples, we use the Mean Squared Error:
$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

**3. Parameter Update (Gradient Descent):**
The parameters are updated at each step using a learning rate $\alpha$:
$$w_j = w_j - \alpha \frac{\partial J}{\partial w_j}$$
$$b = b - \alpha \frac{\partial J}{\partial b}$$

---

## Requirements
To run this project, you need Python installed (Python 3.7+ recommended) and the following external libraries:
* `numpy`
* `matplotlib`

You can install the dependencies by running:
> pip install numpy matplotlib

---

## How to Run
1. Clone this repository or download the `gradient_descent.py` file.
2. Open a terminal in the directory where the file is located.
3. Run the script:
> python gradient_descent.py

---

## Output Example

In the console, you will see the algorithm's progress at every 10% of the iterations:
> Iteration 0
> Cost    25.50
> dj_dW: [...]
> dJ_db: [...]
> w: [...], b:[...]

Finally, the script will display the parameters found by the algorithm compared to the true ones and will open a window with two plots:
* **Cost vs. Iteration**: The evolution of the cost function throughout the entire training process.
* **Cost vs. Iteration (tail)**: A detailed view (zoom) on the final iterations to observe the fine convergence.