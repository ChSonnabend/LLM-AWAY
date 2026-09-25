# Session 24 / 28 investigation (2026-09-25)

Mac session 24 uses Hydra Slurm job 33537. The model exited twice with code 137;
its job cgroup reported a 68,719,476,736-byte memory limit and peak, with two OOM
kills. Both attempts reached the GLM MTP draft-context initialization. The job
itself remained allocated. Restarting the model with MTP off loaded successfully
at 750,080 context tokens, and both health and a small inference request returned
HTTP 200. No allocation was cancelled or replaced.

Mac session 28 / Linux session 33 uses direct worker allocation d2ab35c53983 on
epn000. Its model log reports unknown architecture `glm5next`: the existing ROCm
build lacks the needed model implementation. A separate ROCm build uses pinned
GLM5-Next source revision 86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9.

The reported Linux SSH discovery timeout was not reproduced from the Mac: the
same remote inventory command returned in 1.45 seconds. Discovery now disables
X11 forwarding, uses SSH keepalives, catches timeouts, and caches catalogues per
host and model directory. The web path has a 10-second inventory bound and can
fall back to the existing model descriptor when inventory fails. Local CLI
loading uses the cached catalogue too, avoiding another immediate full scan.
