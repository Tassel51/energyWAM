"""Numerical guards must not be mistaken for unknown runtime failures."""
import pytest
from energywam.training.resource_monitor import classify_exit

@pytest.mark.parametrize('message',[
 'RuntimeError: non-finite parameter gradients in 3 tensors at step 64',
 'RuntimeError: non-finite loss at step 6',
 'ValueError: nonfinite validation MSE',
 'FloatingPointError: nonfinite final candidate',
 'RuntimeError: nonfinite conditional loss',
])
def test_numerical_failure_signatures(message):
 assert classify_exit(1,message,{})=='numerical_failure'

def test_memory_failures_and_explicit_guards_remain_distinct():
 assert classify_exit(1,'CUDA out of memory',{})=='gpu_memory_allocation_failure'
 assert classify_exit(-9,'',{'oom_kill':1})=='container_oom_correlated_kill'
 assert classify_exit(-15,'',{},'Fixed train MSE exceeded baseline')=='experiment_guard'
