from energywam.training.offload import host_allocator_bytes


def test_old_torch_without_host_allocator_api_does_not_crash(monkeypatch):
    monkeypatch.delattr('torch.cuda.memory.host_memory_stats',raising=False)
    assert host_allocator_bytes() is None


def test_supported_host_allocator_reports_residency(monkeypatch):
    monkeypatch.setattr('torch.cuda.memory.host_memory_stats',lambda:{'allocated_bytes.current':12345},raising=False)
    assert host_allocator_bytes()==12345
