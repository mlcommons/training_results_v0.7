from collections import OrderedDict
from loky import get_reusable_executor
# from concurrent.futures import ProcessPoolExecutor
# from mpi4py.futures import MPIPoolExecutor

# A general class for async executing of functions
class AsyncExecutor(object):
    def __init__(self, max_workers=1):
        self.max_workers = max_workers
        self.pool = get_reusable_executor(max_workers=max_workers, timeout=None)
        # self.pool = ProcessPoolExecutor(max_workers=max_workers)
        # self.pool = MPIPoolExecutor(max_workers, main=False)
        self.tasks = OrderedDict()  # a dict of {tags: futures}

    def __del__(self):
        self.cancel(tag=None)
        self.pool.shutdown(wait=False)

    ####################
    # Executor functions
    ####################
    # submit given function and its arguments for async execution
    def submit(self, tag, fn, *args, **kwargs):
        self.tasks[tag] = self.pool.submit(fn, *args, **kwargs)

    def shutdown(wait=True):
        self.pool.shutdown(wait=True)
    #############################
    # functions on future objects
    #############################
    def cancel(self, tag=None):
        if tag:
            return self.tasks[tag].cancel()
        else:
            return {tag: self.tasks[tag].cancel() for tag in self.tasks.keys()}

    def cancelled(self, tag=None):
        if tag:
            return self.tasks[tag].cancelled()
        else:
            return {tag: self.tasks[tag].cancelled() for tag in self.tasks.keys()}

    def running(self, tag=None):
        if tag:
            return self.tasks[tag].running()
        else:
            return {tag: self.tasks[tag].running() for tag in self.tasks.keys()}

    def done(self, tag=None):
        if tag:
            return self.tasks[tag].done()
        else:
            return {tag: self.tasks[tag].done() for tag in self.tasks.keys()}

    def result(self, tag=None, timeout=None):
        if tag:
            return self.tasks[tag].result(timeout=timeout)
        else:
            return {tag: self.tasks[tag].result(timeout=timeout) for tag in self.tasks.keys()}

    def exception(self, tag=None, timeout=None):
        if tag:
            return self.tasks[tag].exception(timeout=timeout)
        else:
            return {tag: self.tasks[tag].exception(timeout=timeout) for tag in self.tasks.keys()}

    def add_done_callback(self, tag=None, fn=None):
        if tag:
            return self.tasks[tag].add_done_callback(fn=fn)
        else:
            return {tag: self.tasks[tag].add_done_callback(fn=fn) for tag in self.tasks.keys()}

    ######################
    # Management functions
    ######################
    # return result of a task and delete if successful
    # if blocking is true, wait timeout for the task to complete
    # if timeout is None, wait indefinitely
    def dequeue_if_done(self, tag, blocking=False, timeout=None):
        if self.done(tag=tag):
            result = self.result(tag=tag)
            del self.tasks[tag]
        elif blocking:
            result = self.result(tag=tag, timeout=timeout)
            del self.tasks[tag]
        else:
            result = None
        return result

    # return the result of last (LIFO) task or first task (FIFO) and delete if successful
    # if blocking is true, wait timeout for the task to complete
    # if timeout is None, wait indefinitely
    def pop_if_done(self, last=True, blocking=False, timeout=None):
        if len(self.tasks)==0:
            return None
        tag = next(iter(self.tasks.keys())) if last else next(reversed(self.tasks.keys()))
        return {tag: self.dequeue_if_done(tag=tag, blocking=blocking, timeout=timeout)}

    # pop the result of all done tasks
    def pop_done(self):
        if len(self.tasks)==0:
            return None
        done_tasks = {}
        tags = list(self.tasks.keys()) # make a copy of tags because we might mutate self.tasks
        for tag in tags:
            result = self.dequeue_if_done(tag, blocking=False)
            if result:
                done_tasks[tag] = result
        return done_tasks

    # return list of tags
    def tags(self):
        return self.tasks.keys()
