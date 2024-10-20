import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


data = np.random.rand(10000, 5)  
columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
df = pd.DataFrame(data, columns=columns)


min_max_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)


standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)


robust_scaler = RobustScaler()
df_robust = pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)


maxabs_scaler = MaxAbsScaler()
df_maxabs = pd.DataFrame(maxabs_scaler.fit_transform(df), columns=df.columns)


print("Min-Max Normalization:\n", df_minmax.head())
print("\nZ-Score Normalization:\n", df_standard.head())
print("\nRobust Scaling:\n", df_robust.head())
print("\nMax-Abs Scaling:\n", df_maxabs.head())


df_minmax.to_csv('minmax_normalized.csv', index=False)
df_standard.to_csv('zscore_normalized.csv', index=False)
df_robust.to_csv('robust_normalized.csv', index=False)
df_maxabs.to_csv('maxabs_normalized.csv', index=False)


df_sales = pd.read_csv('100_sales.csv')


numeric_columns = df_sales.select_dtypes(include=[np.number]).columns


scaler = StandardScaler()
df_sales_scaled = pd.DataFrame(scaler.fit_transform(df_sales[numeric_columns]), columns=numeric_columns)


print("\nZ-Score Normalized Sales Data:\n", df_sales_scaled.head())


df_sales_scaled.to_csv('sales_data_zscore_normalized.csv', index=False)


df_sales_mean_centered = df_sales[numeric_columns] - df_sales[numeric_columns].mean()


print("\nMean-Centered Sales Data:\n", df_sales_mean_centered.head())


df_sales_mean_centered.to_csv('sales_data_mean_centered.csv', index=False)
