$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$
where, n is the number of features, parameters $w_j$,  $b$, are updated simultaneously and where  

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
\end{align}
$$
* m is the number of training examples in the data set
*  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value
* $\frac{\partial J(\mathbf{w},b)}{\partial b}$ for the example can be computed directly and accumulated
    - in a second loop over all n features:
        - $\frac{\partial J(\mathbf{w},b)}{\partial w_j}$ is computed for each $w_j$.
   

```python
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw
```

### Example
Suppose we have:
- **2 examples** (`m = 2`) Row
- **2 features** (`n = 2`) Column

	So:
	
	$X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ , $y = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$
	
	and parameters:
	
	$w = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$ , $b = 0$

$$f_{w,b}(i) = X[i] \cdot w + b$$

| i   | X[i]   | y[i] | f_wb_i = dot(X[i], w)+b | error = f_wb_i - y[i] |
| --- | ------ | ---- | ----------------------- | --------------------- |
| 0   | [1, 2] | 2    | (1×0.5 + 2×0.5) = 1.5   | 1.5 − 2 = **−0.5**    |
| 1   | [3, 4] | 3    | (3×0.5 + 4×0.5) = 3.5   | 3.5 − 3 = **+0.5**    |
`errors = [-0.5, +0.5]`

#### Compute gradients manually (loop logic)
```python
dj_dw = [0, 0] # Gradient for w
dj_db = 0 # Gradient for b
```

For i = 0:

Now go **feature by feature (inner loop):**
```python
for j in range(n):                         
    dj_dw[j] = dj_dw[j] + err * X[i, j]
```

|j|X[i, j]|Update rule|New dj_dw[j]|
|---|---|---|---|
|0|1|dj_dw[0] += (-0.5 × 1)|-0.5|
|1|2|dj_dw[1] += (-0.5 × 2)|-1.0|
`err = -0.5`
`dj_db += err = -0.5`

After i=0:
`dj_dw = [-0.5, -1.0]`
`dj_db = -0.5`


For i = 1:
`err = +0.5`

| j   | X[i, j] | Update rule                   | New dj_dw[j] |
| --- | ------- | ----------------------------- | ------------ |
| 0   | 3       | dj_dw[0] += (+0.5 × 3) = +1.5 | 1.0          |
| 1   | 4       | dj_dw[1] += (+0.5 × 4) = +2.0 | 1.0          |
`dj_db += +0.5 → 0.0`
`dj_dw = [1.0, 1.0]`
`dj_db = 0.0`

#### Average over m examples
```python
dj_dw = dj_dw / m                                
dj_db = dj_db / m
```


#### Gradient Descent
```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing
```