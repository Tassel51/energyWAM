"""Durable JSON sidecars, with bounded verification of remote-file visibility."""
import json
import os
from pathlib import Path
import tempfile
import time


def read_checkpoint_metadata(path, *, expected_meta=None, attempts=6):
    path=Path(path)
    for attempt in range(attempts):
        try:
            value=json.loads(path.read_text())
            if not isinstance(value,dict) or 'meta' not in value:
                raise ValueError('Checkpoint sidecar has no metadata object')
            if expected_meta is not None and value['meta']!=expected_meta:
                raise ValueError('Checkpoint/sidecar metadata mismatch')
            return value
        except (OSError,ValueError):
            if attempt+1==attempts:raise
            time.sleep(min(.25*2**attempt,4.))


def write_checkpoint_metadata(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.pending',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,default=str)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        directory=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        read_checkpoint_metadata(path,expected_meta=value['meta'])
    finally:
        temporary.unlink(missing_ok=True)
