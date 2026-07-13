`pin_gpu_round_robin(num_gpus)`: an opt-in helper to pack several small sweep jobs
onto each GPU. Called at the top of a Hydra task, it sets `CUDA_VISIBLE_DEVICES`
to `job_index % num_gpus` so jobs round-robin across devices; run
`num_gpus * jobs_per_gpu` jobs concurrently (via your launcher's `n_jobs`) to
co-locate them. New "Packing small jobs onto GPUs" guide covers the joblib recipe,
Ray fractional-GPU, and MPS/MIG.
