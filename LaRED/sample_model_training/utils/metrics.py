import numpy as np

__all__ = ['corrcoef', 'mae', 'max', 'mape']


def corrcoef(arr1,arr2):
    corrcoef = np.corrcoef(arr1,arr2)
    return corrcoef[0][1]

def mae(arr1,arr2):
    mae = np.sum(np.absolute(arr1-arr2)) / len(arr1)
    return mae*1000

def max(arr1, arr2):
    max = np.max(np.absolute(arr1 - arr2))
    return max*1000

def mape(arr1, arr2):
    mape = np.mean(np.abs((np.array(arr1) - np.array(arr2)) / np.array(arr1))) * 100
    return mape