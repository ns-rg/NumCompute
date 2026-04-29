import numpy as np

from numcompute.benchmarking import run_performance_table
       
data = np.random.rand(10000)

run_performance_table(data)
        