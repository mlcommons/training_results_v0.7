'''SSD with async support'''
from async_executor import AsyncExecutor

# Why is this file needed ?
# By default, AsyncExecutor starts new processes by forking the parent python interpreter (the newly
# forked child process is effectively identical to the parent process). OpenMPI doesn't like that
# and often crashes the child process.
# There are two ways to work around this issue:
#   1) fork the child processes before OpenMPI is initialized (this is what this file effectively does).
#   2) start new processes with a "spawn" context instead of "fork".

if __name__ == "__main__":
    async_executor = AsyncExecutor()
    from ssd_main import main
    main(async_executor)
