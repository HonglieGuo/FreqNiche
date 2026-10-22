"""Transfer helpers; keep loader order, RNG draws and effective batch unchanged."""

from concurrent.futures import ThreadPoolExecutor
import torch


def _next_batch(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def iter_batches(loader, prefetch=False):
    """One CPU producer, one batch ahead; no cloned Dataset RNG or workers.

    Prime the first batch synchronously: the standard RandomSampler consumes
    the global torch base/sampler seeds at this point, then uses its own local
    generator. Subsequent Dataset calls consume only the Dataset's private
    numpy RNG. This helper is for this project's num_workers=0 loaders only.
    """
    if not prefetch:
        yield from loader
        return
    if loader.num_workers != 0:
        raise ValueError("Thread prefetch requires num_workers=0")
    iterator = iter(loader)
    valid, batch = _next_batch(iterator)
    if not valid:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="freqniche-data") as pool:
        while valid:
            pending = pool.submit(_next_batch, iterator)
            yield batch
            valid, batch = pending.result()  # propagates producer exceptions


def stage_batch(batch, device, enabled=True):
    if not enabled or device.type != "cuda":
        return batch
    # Keep metadata on CPU. The original call sites may still call .to(device),
    # which is a no-op for these already staged tensors.
    return {
        k: (v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor) and k not in {"center_idx", "is_isolated"}
            else v)
        for k, v in batch.items()
    }
