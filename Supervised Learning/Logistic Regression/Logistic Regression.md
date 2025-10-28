- we would like the predictions of our classification model to be between 0 and 1 since our output variable 𝑦 is either 0 or 1.
- This can be accomplished by using a "sigmoid function" which maps all input values to values between 0 and 1.
- $g(z) = \frac{1}{1+e^{-z}}\tag{1}$
- ```python
  def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g
  ```


### Decision Boundary
- **decision boundary** is the **line (or curve)** that **separates classes** in a classification problem.