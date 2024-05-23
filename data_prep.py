import numpy as np
import pandas as pd

df1 = pd.read_csv("./datastet/sign_mnist_test.csv", header = 0)

print(df1.data.head())