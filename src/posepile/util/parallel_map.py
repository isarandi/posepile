import atexit
import ctypes
import multiprocessing
import os
import queue
import signal
import threading


def parallel_map_as_generator(fns_args_kwargs, n_workers=None, max_unconsumed=256, context='fork'):
    if n_workers is None:
        n_workers = min(len(os.sched_getaffinity(0)), 12)

    ctx = multiprocessing.get_context(context)
    pool = ctx.Pool(n_workers, initializer=init_worker_process)

    semaphore = threading.Semaphore(max_unconsumed)
    q = queue.Queue()
    end_of_sequence_marker = object()

    should_stop = False
    def producer():
        for fn, args, kwargs in fns_args_kwargs:
            if should_stop:
                break
            semaphore.acquire()
            q.put(pool.apply_async(fn, args, kwargs))

        q.put(end_of_sequence_marker)

    def stop():
        nonlocal should_stop
        should_stop = True
        pool.close()
        pool.terminate()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    atexit.register(stop)

    while (future := q.get()) is not end_of_sequence_marker:
        value = future.get()
        semaphore.release()
        yield value


def init_worker_process():
    _terminate_on_parent_death()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _terminate_on_parent_death():
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
