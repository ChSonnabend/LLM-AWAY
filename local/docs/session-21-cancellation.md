# Session 21 / Hydra job 33066

Slurm accounting records `CANCELLED by 9898`, not `TIMEOUT`:

- Start: 2026-09-25 07:23:44 (scheduler timestamp).
- End: 2026-09-25 10:45:26, about 3 hours 22 minutes later.
- Scheduler limit: seven days.
- The remote account UID is 9898.
- The allocation's `release` file was written at the same cancellation timestamp.
- The local log reports healthy model responses until `COMPLETING`, then
  `CANCELLED` and cleanup after the remote job ended.

This is consistent with a framework release request. Historical logs do not
identify which client or whether it was a manual action or watchdog. The earlier
startup-timeout entry precedes a subsequent successful model load and does not
explain this cancellation of the loaded job.

Two automatic cancellation paths were found and removed: watchdog cleanup after
a local monitor dies, and handling SIGTERM/SIGINT as remote release. A stale local
cleanup flag can no longer make monitor refresh release a remotely active job.
The remote controller now requires explicit release intent and records release
requests, to protect against legacy watchdog calls and improve future attribution.
