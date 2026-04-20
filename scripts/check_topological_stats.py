#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('/Users/aritrabose/Library/CloudStorage/OneDrive-IBM/Research/Quantum/quvine/ppi_disease_v3/results/topological_checkpoint.csv')
print('Total networks processed:', len(df))
print('\nSummary statistics:')
for col in ['betti_0', 'betti_1', 'betti_2', 'betti_sum', 'euler_characteristic', 'persistence_entropy_H0', 'persistence_entropy_H1', 'persistence_entropy_H2']:
    non_zero = (df[col] != 0).sum()
    print(f'{col:30s}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, min={df[col].min():.4f}, max={df[col].max():.4f}, non-zero={non_zero}')

print(f'\nSample of networks with non-zero values:')
non_zero_mask = (df['betti_0'] != 0) | (df['betti_1'] != 0) | (df['betti_2'] != 0)
if non_zero_mask.any():
    print(df[non_zero_mask][['network_id', 'betti_0', 'betti_1', 'betti_2', 'betti_sum', 'euler_characteristic']].head(10))
else:
    print("All Betti numbers are zero - this suggests an issue with the computation")
    print("First few rows:")
    print(df[['network_id', 'betti_0', 'betti_1', 'betti_2']].head())

# Made with Bob
