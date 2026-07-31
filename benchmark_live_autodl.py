import argparse
import statistics
import time

import numpy as np
import torch

import after_local_ui as ui


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    index = int(round((len(values) - 1) * pct / 100.0))
    return values[index]


def main():
    parser = argparse.ArgumentParser(description="Benchmark AFTER live chunk latency on the current GPU.")
    parser.add_argument("--steps", default="1,2,4,6", help="Comma-separated nb_steps values to test.")
    parser.add_argument("--chunks", type=int, default=8, help="Measured chunks per nb_steps value.")
    parser.add_argument("--guidance", type=float, default=1.0)
    args = parser.parse_args()

    steps_list = [max(1, min(6, int(part.strip()))) for part in args.steps.split(",") if part.strip()]
    audio_ms = ui.CHUNK_SIZE / ui.SAMPLE_RATE * 1000.0
    rng = np.random.default_rng(1234)
    samples = (rng.standard_normal(ui.CHUNK_SIZE).astype("<f4") * 0.02)
    raw = samples.tobytes()

    print(f"device: {ui.DEFAULT_DEVICE}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"chunk: {ui.CHUNK_SIZE} samples = {audio_ms:.2f} ms audio")
    print("real-time target: p95 below chunk audio ms")

    for steps in steps_list:
        ui._reset_live_model(steps, args.guidance)
        # One extra warmup after reset so the measured loop is steadier.
        ui._process_live_chunk(raw, ui.SAMPLE_RATE, [0.0] * 6, steps, args.guidance, 0.0, 1.0, 0.0, 0.85, 1.0, 0.0, 320.0, 0.35)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(args.chunks):
            start = time.perf_counter()
            ui._process_live_chunk(raw, ui.SAMPLE_RATE, [0.0] * 6, steps, args.guidance, 0.0, 1.0, 0.0, 0.85, 1.0, 0.0, 320.0, 0.35)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

        avg = statistics.mean(times)
        p95 = percentile(times, 95)
        ratio = p95 / audio_ms
        status = "OK" if ratio <= 1.0 else "BACKLOG"
        print(f"nb_steps={steps}: avg={avg:.1f} ms p95={p95:.1f} ms realtime_ratio={ratio:.2f} {status}")


if __name__ == "__main__":
    main()