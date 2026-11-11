"""Read-only resource telemetry scoped to an owned experiment process."""
import json,os,subprocess,threading,time
from pathlib import Path
import psutil


def memory_events(root):
    try:return {k:int(v) for line in (Path(root)/'memory.events').read_text().splitlines() for k,v in [line.split()]}
    except (OSError,ValueError):return {}


def classify_exit(code,console,event_delta,reason=None):
    if code==0:return 'completed'
    if reason:return 'experiment_guard'
    lower=console.lower()
    if 'cuda out of memory' in lower or 'torch.outofmemoryerror' in lower:return 'gpu_memory_allocation_failure'
    if 'no space left on device' in lower:return 'disk_full'
    if 'defaultcpuallocator' in lower or 'memoryerror' in lower:return 'cpu_memory_allocation_failure'
    if 'native microbatch gradient mismatch' in lower:return 'gradient_equivalence_check_failed'
    if any(x in lower for x in ['nonfinite conditional loss','nonfinite loss','nonfinite gradient',
                               'non-finite loss','non-finite parameter gradients',
                               'nonfinite validation mse','nonfinite final candidate',
                               'nonfinite iteration-aware diagnostic']):return 'numerical_failure'
    if code in [-9,137] and event_delta.get('oom_kill',0)>0:return 'container_oom_correlated_kill'
    if code in [-9,137]:return 'sigkill_unknown_cause'
    return 'runtime_or_signal_error'


class ResourceMonitor:
    def __init__(self,pid,gpus,folder,cgroup='/sys/fs/cgroup',interval=5.):
        self.pid,self.gpus,self.folder=pid,set(map(int,gpus)),Path(folder)
        self.cgroup=Path(cgroup);self.interval=interval;self.stop=threading.Event()
        self.events_before=memory_events(self.cgroup);self.started=time.time();self.cpu_processes={}
        self.summary={'pid':pid,'physical_gpus':sorted(self.gpus),'started_unix':self.started,
                      'cpu_percent_definition':'100 percent = one logical CPU; RSS/USS are process-tree CPU memory, not container total',
                      'container_scope':'Shared cgroup includes pre-existing processes and file cache; deltas are correlation, not proof of ownership',
                      'memory_events_before':self.events_before,'samples':0,'peaks':{},'gpu_peaks':{},'monitor_errors':[]}
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self):self.thread.start()
    def sample(self):
        row={'unix':time.time(),'elapsed_seconds':time.time()-self.started}
        try:
            progress=json.loads((self.folder/'speed_report.json').read_text())
            row['completed_updates']=progress['completed_steps']
            row['completed_sample_visits']=progress['completed_steps']*progress['global_batch']
        except (OSError,ValueError,KeyError):pass
        rss=uss=cpu=0.;alive=[]
        try:processes=[psutil.Process(self.pid)];processes+=processes[0].children(recursive=True)
        except psutil.Error:processes=[]
        for process in processes:
            try:
                key=(process.pid,process.create_time())
                p=self.cpu_processes.setdefault(key,process)
                cpu+=p.cpu_percent(None);rss+=p.memory_info().rss;alive.append(p.pid)
                uss+=getattr(p.memory_full_info(),'uss',0)
            except psutil.Error:pass
        row.update(process_pids=alive,process_rss_gib=rss/2**30,process_uss_gib=uss/2**30,
                   process_cpu_percent_one_core=cpu,system_cpu_percent=psutil.cpu_percent(None),
                   system_memory_available_gib=psutil.virtual_memory().available/2**30)
        for name in ['memory.current','memory.max','memory.peak']:
            try:
                val=(self.cgroup/name).read_text().strip();row['cgroup_'+name.replace('.','_')]=None if val=='max' else int(val)
            except (OSError,ValueError):pass
        try:
            stats=dict(line.split() for line in (self.cgroup/'memory.stat').read_text().splitlines())
            for key in ['anon','file','shmem','inactive_file','kernel','slab']:row['cgroup_'+key+'_gib']=int(stats.get(key,0))/2**30
        except (OSError,ValueError):pass
        row['memory_events']=memory_events(self.cgroup)
        try:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=4)
            row['gpus']=[dict(zip(['index','memory_used_mib','memory_total_mib','utilization_percent','power_w'],map(float,line.split(',')))) for line in output.splitlines() if int(line.split(',')[0]) in self.gpus]
        except Exception as error:row['gpu_query_error']=type(error).__name__
        return row
    def record(self,row):
        with (self.folder/'resource_trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        self.summary['samples']+=1;self.summary['last_sample']=row
        for key in ['process_rss_gib','process_uss_gib','process_cpu_percent_one_core','cgroup_memory_current','cgroup_anon_gib','cgroup_file_gib']:
            if row.get(key) is not None:self.summary['peaks'][key]=max(self.summary['peaks'].get(key,0),row[key])
        for gpu in row.get('gpus',[]):
            peaks=self.summary['gpu_peaks'].setdefault(str(int(gpu['index'])),{})
            for key in ['memory_used_mib','utilization_percent','power_w']:peaks[key]=max(peaks.get(key,0),gpu[key])
        temp=self.folder/'resource_summary.pending.json';temp.write_text(json.dumps(self.summary,indent=2));temp.replace(self.folder/'resource_summary.json')
    def run(self):
        while not self.stop.is_set():
            try:self.record(self.sample())
            except Exception as error:self.summary['monitor_errors'].append(type(error).__name__)
            self.stop.wait(self.interval)
    def finish(self,code,reason=None):
        self.stop.set();self.thread.join(timeout=60)
        if self.thread.is_alive():raise RuntimeError('resource monitor failed to terminate')
        after=memory_events(self.cgroup);delta={key:after.get(key,0)-value for key,value in self.events_before.items()}
        console=self.folder/'console.log'
        tail=''
        if console.exists():
            with console.open('rb') as stream:
                stream.seek(max(0,console.stat().st_size-65536));tail=stream.read().decode(errors='replace')
        self.summary.update(returncode=code,ended_unix=time.time(),wall_seconds=time.time()-self.started,
                            memory_events_after=after,memory_events_delta=delta,
                            exit_classification=classify_exit(code,tail,delta,reason),guard_reason=reason)
        self.record(self.sample())
