import tensorflow as tf
import numpy as np
import sys
sys.path.append('Non linear signature optimal stopping')
from Deep_signatures_optimal_stopping import DeepLongstaffSchwartzPricer, DeepDualPricer

# Test DeepLongstaffSchwartzPricer
print("Creating Longstaff-Schwartz pricer...")
ls_pricer = DeepLongstaffSchwartzPricer(
    N1=3,
    T=1.0,
    r=0.05,
    mode="American Option",
    layers=2,
    nodes=8,
    activation_function='tanh'
)
print("Longstaff-Schwartz pricer created successfully")

# Test DeepDualPricer
print("\nCreating Dual pricer...")
dual_pricer = DeepDualPricer(
    N1=3,
    N=10,
    T=1.0,
    r=0.05,
    layers=2,
    nodes=8,
    activation_function='relu'
)
print("Dual pricer created successfully")

# This validates that our classes can be instantiated properly
print("\nTest completed successfully") 