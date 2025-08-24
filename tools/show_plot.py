import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def show_latest_plot(artifacts_dir="artifacts"):
    """
    Finds the latest run directory and displays the return.png plot.
    """
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        print(f"Artifacts directory not found at: {artifacts_dir}")
        return

    run_dirs = sorted([d for d in artifacts_path.iterdir() if d.is_dir() and d.name.startswith('run_')], reverse=True)

    if not run_dirs:
        print(f"No run directories found in: {artifacts_dir}")
        return

    latest_run_dir = run_dirs[0]
    plot_path = latest_run_dir / "plots" / "return.png"

    if not plot_path.exists():
        print(f"Plot not found at: {plot_path}")
        return

    print(f"Showing plot from: {plot_path}")
    img = mpimg.imread(plot_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the latest learning curve plot.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Path to the artifacts directory.")
    args = parser.parse_args()
    show_latest_plot(args.artifacts_dir)
