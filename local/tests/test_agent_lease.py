import fcntl
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from llm_away import resources

class AgentLeaseTests(unittest.TestCase):
    def test_switch_waits_for_previous_worker_to_release_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'client.lock'
            with path.open('a') as old,path.open('a') as new:
                fcntl.flock(old,fcntl.LOCK_EX)
                def release():
                    time.sleep(.15);fcntl.flock(old,fcntl.LOCK_UN)
                worker=threading.Thread(target=release);worker.start()
                resources.acquire_agent_lease(new,reconfigure=True)
                worker.join()
                with self.assertRaises(BlockingIOError):fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_normal_run_does_not_wait_or_steal_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'client.lock'
            with path.open('a') as old,path.open('a') as new:
                fcntl.flock(old,fcntl.LOCK_EX)
                with self.assertRaisesRegex(ValueError,'Another run command'):resources.acquire_agent_lease(new)

    def test_switch_timeout_keeps_existing_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'client.lock'
            with path.open('a') as old,path.open('a') as new:
                fcntl.flock(old,fcntl.LOCK_EX)
                with patch.object(resources.time,'monotonic',side_effect=[0,13]):
                    with self.assertRaisesRegex(ValueError,'still shutting down'):resources.acquire_agent_lease(new,True)
