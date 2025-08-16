import pandas as pd
import os

os.chdir("cgmacros1.0/CGMacros")
data_all_sub = pd.read_csv("data_all_sub.csv")
print(data_all_sub.columns)