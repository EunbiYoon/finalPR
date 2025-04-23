# Import required libraries
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the digits dataset (composed of handwritten digits as 8x8 images)
digits = datasets.load_digits()

# Feature data (64 pixel values), Label data (digit classes from 0 to 9)
X = digits.data         # shape: (1797, 64)
X=pd.DataFrame()
y = digits.target       # shape: (1797,)
N = len(X)              # Total number of samples

print(X.columns)
print(y)
print(N)

# # Randomly select one digit image
# digit_to_show = np.random.choice(range(N), 1)[0]

# # Print the pixel values (attributes) and the class label of the selected digit
# print("Attributes:", X[digit_to_show])
# print("Class:", y[digit_to_show])

# # Reshape the selected digit to 8x8 and display it as a grayscale image
# plt.imshow(X[digit_to_show].reshape(8, 8), cmap='gray')
# plt.title(f"Digit: {y[digit_to_show]}")
# plt.axis('off')
# plt.show()
