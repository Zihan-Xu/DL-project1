"""Portable output-directory lock and atomic status writes; no project queue needed."""
import contextlib,json,os
from pathlib import Path
from datetime import datetime,timezone
OUTPUT=Path('.')
class Paused(RuntimeError):pass
class Cancelled(RuntimeError):pass
def utc_now():return datetime.now(timezone.utc).isoformat()
def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');os.replace(tmp,path)
def check_control(category):
    p=OUTPUT/'control.json'
    if p.exists():
        c=json.loads(p.read_text())
        if c.get('cancel'):raise Cancelled('Output control.json requests cancellation')
        if c.get('pause'):raise Paused('Output control.json requests pause')
@contextlib.contextmanager
def gpu_lock(category,run_id):
    """Prevents duplicate writers to this output directory, not other GPU workloads."""
    with (OUTPUT/'worker.lock').open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            f.seek(0,2)
            if f.tell()==0:f.write(b'0');f.flush()
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)
