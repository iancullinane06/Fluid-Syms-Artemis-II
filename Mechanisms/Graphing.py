import time


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def update_progress_bar(current: int, total: int, start_time: float, width: int = 32) -> None:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    fraction = current / total
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.perf_counter() - start_time, 0.0)
    eta_seconds = 0.0 if current <= 0 else elapsed * \
        max(total - current, 0) / max(current, 1)
    print(
        f"\rPrecomputing frames: [{bar}] {current}/{total} ({fraction * 100:5.1f}%) | ETA {format_duration(eta_seconds)}",
        end="",
        flush=True,
    )
    if current >= total:
        print(f" | done in {format_duration(elapsed)}")
