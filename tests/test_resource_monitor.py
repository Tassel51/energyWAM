from energywam.training.resource_monitor import classify_exit,memory_events,ResourceMonitor
import os


def test_exit_classification_distinguishes_gpu_cpu_and_unknown_kill():
 assert classify_exit(1,'torch.OutOfMemoryError: CUDA out of memory',{})=='gpu_memory_allocation_failure'
 assert classify_exit(1,'DefaultCPUAllocator: cannot allocate memory',{})=='cpu_memory_allocation_failure'
 assert classify_exit(-9,'',{'oom_kill':1})=='container_oom_correlated_kill'
 assert classify_exit(-9,'',{'oom_kill':0})=='sigkill_unknown_cause'
 assert classify_exit(1,'No space left on device',{})=='disk_full'
 assert classify_exit(1,'RuntimeError: nonfinite conditional loss',{})=='numerical_failure'
 assert classify_exit(-15,'',{},'nonfinite loss')=='experiment_guard'
 assert classify_exit(1,'RuntimeError: nonfinite gradient in native microbatch check',{})=='numerical_failure'
 assert classify_exit(1,'RuntimeError: native microbatch gradient mismatch; do not select native2',{})=='gradient_equivalence_check_failed'
 assert classify_exit(0,'',{})=='completed'


def test_memory_events_preserves_historical_counts_for_delta(tmp_path):
 (tmp_path/'memory.events').write_text('max 9791\noom 0\noom_kill 0\n')
 assert memory_events(tmp_path)=={'max':9791,'oom':0,'oom_kill':0}


def test_monitor_writes_separate_process_and_container_scopes(tmp_path,monkeypatch):
 (tmp_path/'console.log').write_text('')
 monkeypatch.setattr('energywam.training.resource_monitor.subprocess.check_output',lambda *a,**k:'0,100,49140,10,50\n2,200,49140,20,60\n')
 monitor=ResourceMonitor(os.getpid(),[0],tmp_path,cgroup=tmp_path,interval=.01)
 sample=monitor.sample();assert len(sample['gpus'])==1 and sample['gpus'][0]['index']==0
 assert sample['process_rss_gib']>0
 monitor.start();monitor.finish(0)
 assert (tmp_path/'resource_trace.jsonl').exists()
 import json
 result=json.loads((tmp_path/'resource_summary.json').read_text());assert result['exit_classification']=='completed'
 assert result['memory_events_delta']=={}


def test_monitor_without_console_retains_resource_summary(tmp_path,monkeypatch):
    monkeypatch.setattr('energywam.training.resource_monitor.subprocess.check_output',lambda *a,**k:'0,100,49140,10,50\n')
    monitor=ResourceMonitor(os.getpid(),[0],tmp_path,cgroup=tmp_path,interval=.01)
    monitor.start();monitor.finish(0)
    import json
    assert json.loads((tmp_path/'resource_summary.json').read_text())['exit_classification']=='completed'
