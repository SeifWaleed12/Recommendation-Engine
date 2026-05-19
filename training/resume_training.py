import os
import sys
import time
from pathlib import Path

# Set environment variable to fix Protobuf mismatch
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_all import run_step, step_encode_sbert, step_build_cbf_faiss, step_train_two_tower, step_train_ranker

def main():
    total_start = time.time()
    timings = {}

    print("=== RESUMING TRAINING FROM STEP 5 ===")
    
    steps = [
        ("5a. Encode Items (SBERT)", step_encode_sbert),
        ("5b. Build CBF FAISS Index", step_build_cbf_faiss),
        ("6. Train Two-Tower + Neural FAISS", step_train_two_tower),
        ("7. Train LightGBM Ranker", step_train_ranker),
    ]

    for name, func in steps:
        try:
            elapsed = run_step(name, func)
            timings[name] = elapsed
        except Exception as e:
            print(f"\nFATAL: Pipeline failed at step '{name}': {e}")
            break

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("RESUME PIPELINE SUMMARY")
    print(f"{'='*60}")
    for step_name, elapsed in timings.items():
        print(f"  {step_name}: {elapsed:.1f}s")
    print(f"  Total: {total_time:.1f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
