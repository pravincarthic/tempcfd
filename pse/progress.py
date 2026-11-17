"""MPI progress hook.

The mode scheduler's work counter is an RMA window on rank 0.  Without
hardware atomics or an async-progress thread, a Fetch_and_op from another
rank only completes when rank 0 enters the MPI library, so while rank 0
marches a mode the others wait.  The compute loops call poke(), which makes
a cheap MPI call (Iprobe) so the counter is served promptly; ranks no longer
idle and no core is spent on a progress thread.
"""
_hook = None


def set_hook(fn):
    global _hook
    _hook = fn


def poke():
    if _hook is not None:
        _hook()
