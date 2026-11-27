import argparse
import numpy as np

def inspect(npz_path):
    print(f"\n==================================================")
    print(f" Inspecting: {npz_path}")
    print(f"==================================================")
    data = np.load(npz_path)
    
    print(f"{'Array Name':<16} | {'Shape':<14} | {'Dtype':<10} | {'Summary (Min / Max / Mean)'}")
    print("-" * 75)
    for key in data.files:
        arr = data[key]
        if arr.dtype.kind in {'f', 'i', 'u'}:
            stats = f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
        else:
            stats = f"non-numeric or complex"
        print(f"{key:<16} | {str(arr.shape):<14} | {str(arr.dtype):<10} | {stats}")
    
    print("-" * 75)
    # If it is a simulation file, show a glimpse of snapshot 0
    if "vm_true" in data.files:
        print("\n--- Snapshot #0 Sample Values ---")
        print(f"Hour of day: {data['hour'][0]:.2f}")
        print(f"Dropout Fraction: {data['drop_frac'][0]:.1%}")
        print(f"Active PMU count: {int(data['mask'][0].sum())} / {len(data['mask'][0])}")
        print(f"First 5 True Voltage Magnitudes (pu): {np.round(data['vm_true'][0][:5], 4)}")
        print(f"First 5 True Voltage Angles (rad):    {np.round(data['va_true'][0][:5], 4)}")
    elif "pmu_idx" in data.files:
        print("\n--- Feeder Metadata ---")
        print(f"Number of buses: {data['n_bus']}")
        print(f"Slack bus index: {data['slack']}")
        print(f"Installed PMU indices ({len(data['pmu_idx'])} total): {data['pmu_idx'].tolist()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect .npz power system dataset files")
    parser.add_argument("file", nargs="?", default="data_small/ieee33/train.npz", help="Path to .npz file")
    args = parser.parse_args()
    inspect(args.file)
