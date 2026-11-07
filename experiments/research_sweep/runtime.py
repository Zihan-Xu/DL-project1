"""Shared coordination utilities. Category workers own their training implementations."""
from contextlib import contextmanager
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent


class Paused(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Windows readers can briefly deny replacement while holding the destination.
    deadline = time.monotonic() + 2.0
    retry_delay = 0.025
    while True:
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(retry_delay, remaining))
            retry_delay = min(retry_delay * 2, 0.2)


def read_control():
    return json.loads((STUDY / 'control.json').read_text(encoding='utf-8'))


def check_control(category):
    control = read_control()
    item = control.get('categories', {}).get(category, {})
    if control.get('cancel_all') or item.get('cancel'):
        raise Cancelled('User cancelled ' + category)
    if control.get('pause_all') or item.get('pause'):
        raise Paused('User paused ' + category)
    for request in control.get('requested_changes', []):
        if (request.get('status', 'pending') == 'pending'
                and request.get('category', 'all') in (category, 'all')
                and request.get('pause_needed', True)):
            raise Paused('Pending user change for ' + category + ': ' + str(request.get('id', 'unlabeled')))
    return control


def write_status(category, **fields):
    path = STUDY / category / 'status.json'
    value = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    value.update(category=category, updated_at_utc=utc_now(), **fields)
    atomic_json(path, value)


def process_alive(pid):
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied is not proof of death.
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def gpu_lock(category, run_id):
    """One GPU holder; FIFO tickets avoid a fast worker monopolizing the device.

    Use for a whole run, release between runs. Call check_control each epoch after
    saving resumable state. Lock release also occurs on process termination.
    """
    queue = STUDY / 'gpu_queue'
    queue.mkdir(parents=True, exist_ok=True)
    ticket = queue / f'{time.time_ns():020d}_{os.getpid()}_{uuid.uuid4().hex}.json'
    atomic_json(ticket, dict(category=category, run_id=run_id, pid=os.getpid(), queued_at=utc_now()))
    lock_path = STUDY / 'gpu.lock'
    handle = lock_path.open('a+b')
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    acquired = False

    def unlock():
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_UN)

    try:
        while not acquired:
            check_control(category)
            hold_path = STUDY / 'external_gpu_hold.json'
            if hold_path.exists() and json.loads(hold_path.read_text(encoding='utf-8')).get('active'):
                time.sleep(1)
                continue
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError):
                time.sleep(1)
                continue
            for candidate in sorted(queue.glob('*.json')):
                entry = json.loads(candidate.read_text(encoding='utf-8'))
                if not process_alive(entry['pid']):
                    candidate.unlink(missing_ok=True)
            tickets = sorted(queue.glob('*.json'))
            if tickets and tickets[0] == ticket:
                acquired = True
            else:
                unlock()
                time.sleep(1)
        atomic_json(STUDY / 'gpu_holder.json', dict(category=category, run_id=run_id,
                    pid=os.getpid(), acquired_at=utc_now(), state='held'))
        yield
    finally:
        ticket.unlink(missing_ok=True)
        if acquired:
            atomic_json(STUDY / 'gpu_holder.json', dict(category=category, run_id=run_id,
                        pid=os.getpid(), released_at=utc_now(), state='released'))
            unlock()
        handle.close()
