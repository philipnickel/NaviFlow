import os
import sys
import time
import numpy as np

def is_gui_available():
    return "DISPLAY" in os.environ and os.environ["DISPLAY"]

# Setup mode
headless = not is_gui_available()

if headless:
    import matplotlib
    matplotlib.use("Agg")
else:
    import matplotlib.pyplot as plt
    plt.ion()

import matplotlib.pyplot as plt

def read_residuals(filepath):
    try:
        data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        return data
    except Exception:
        return np.zeros((0, 4))

def live_plot(filepath, refresh=2.0):
    # Resolve experiment folder and plot output path
    experiment_dir = os.path.dirname(os.path.dirname(filepath))
    residuals_dir = os.path.join(experiment_dir, "residuals")
    os.makedirs(residuals_dir, exist_ok=True)
    plot_path = os.path.join(residuals_dir, "residuals_plot.png")

    fig, ax = plt.subplots()
    u_line, = ax.plot([], [], label="u-residual")
    v_line, = ax.plot([], [], label="v-residual")
    cont_line, = ax.plot([], [], label="continuity")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.grid(True)
    ax.legend()

    print(f"{'🖥️ GUI mode' if not headless else '🧱 Headless mode'}: Monitoring {filepath}")
    print(f"📁 Saving plots to: {plot_path}")

    while True:
        if not os.path.exists(filepath):
            print(f"⏳ Waiting for {filepath}...")
            time.sleep(refresh)
            continue

        data = read_residuals(filepath)
        if data.shape[0] > 0:
            iters, u, v, cont = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            u_line.set_data(iters, u)
            v_line.set_data(iters, v)
            cont_line.set_data(iters, cont)
            ax.relim()
            ax.autoscale_view()

            if headless:
                fig.savefig(plot_path, dpi=150)
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()

        time.sleep(refresh)

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "residuals.log"
    live_plot(filepath)
