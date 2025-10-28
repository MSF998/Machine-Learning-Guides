Feature engineering means **creating new input features (`x`)** from existing ones to help the model capture important patterns
- If the data is going to form a curve, we cannot fit a straight line
- So we engineer features like
	- $x_1 = x, \quad x_2 = x^2, \quad x_3 = x^3$
- So then the model becomes
	-  $f_{\mathbf{w},b}(x) = w_1x + w_2x^2 + w_3x^3 + b$.
	- Which is polynomial regression model

```python
import numpy as np

# Original feature
x = np.array([1, 2, 3, 4, 5])

# Feature engineering: add polynomial features
x2 = x ** 2
x3 = x ** 3

# Combine them into a feature matrix
X = np.stack((x, x2, x3), axis=1)

print("Engineered features:\n", X)
```

```css
[[  1   1   1]
 [  2   4   8]
 [  3   9  27]
 [  4  16  64]
 [  5  25 125]]
\
```