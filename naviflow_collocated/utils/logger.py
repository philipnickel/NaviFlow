import os
import numpy as np

class ResidualLogger:
    def __init__(self, results_dir, log_file="residuals.log",
                 log_every=20, flush_every=100, print_every=100,
                 divergence_factor=100000.0, stagnation_window=50,
                 allow_unsteady=False, convergence_tolerance=1e-4):
        
        self.log_every = log_every
        self.flush_every = flush_every
        self.print_every = print_every
        self.divergence_factor = divergence_factor
        self.stagnation_window = stagnation_window
        self.allow_unsteady = allow_unsteady
        self.convergence_tolerance = float(convergence_tolerance)

        self.buffer = []
        self.history = {"u": [], "v": [], "cont": [], "iters": []}
        self.diverging = False
        self.stalled = False
        self.converged = False

        self.log_path = os.path.join(results_dir, log_file)
        os.makedirs(results_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("iteration,u_residual,v_residual,continuity_residual\n")

    def update(self, i, u, v, cont):
        # Print status
        if self.print_every and i % self.print_every == 0:
            print(f"[{i:5d}] u={u:.2e}  v={v:.2e}  cont={cont:.2e}")

        # Track history
        self.history["iters"].append(i)
        self.history["u"].append(u)
        self.history["v"].append(v)
        self.history["cont"].append(cont)

        # Only buffer periodically
        if i % self.log_every == 0:
            self.buffer.append((i, u, v, cont))

        # Flush periodically
        if i % self.flush_every == 0 and self.buffer:
            self.flush()
            # Check for convergence and stagnation
            self._check_convergence()
            if not self.allow_unsteady:
                self._check_stagnation()

        # Detect divergence: residual blows up compared to prior min
        for key, r in {"u": u, "v": v, "cont": cont}.items():
            history = self.history[key][-self.stagnation_window:]
            if len(history) < 2:
                continue
            r_min = min(history)
            if r_min > 1e-20 and r > self.divergence_factor * r_min:
                print(f"Divergence detected in '{key}': {r_min:.2e} → {r:.2e}")
                self.diverging = True

    def _check_stagnation(self):
        def log_slope(r_hist):
            if len(r_hist) < self.stagnation_window:
                return 0.0
            log_r = np.log10(np.clip(r_hist[-self.stagnation_window:], 1e-20, None))
            x = np.arange(len(log_r))
            slope, _ = np.polyfit(x, log_r, 1)
            return slope

        s_u = log_slope(self.history["u"])
        s_v = log_slope(self.history["v"])
        s_c = log_slope(self.history["cont"])

        if max(abs(s_u), abs(s_v), abs(s_c)) < 0.0001:
            self.stalled = True
    
    def _check_convergence(self):
        # Get the latest residuals
        u_res = float(self.history["u"][-1])
        v_res = float(self.history["v"][-1])

        # Check if momentum residuals are below tolerance
        if max(u_res, v_res) < self.convergence_tolerance:
            #print(f"Solution converged: momentum residuals below tolerance {self.convergence_tolerance:.1e}")
            self.converged = True
            return True
        return False

    def flush(self):
        if not self.buffer:
            return
        with open(self.log_path, "a") as f:
            for i, u, v, cont in self.buffer:
                f.write(f"{i},{u},{v},{cont}\n")
        self.buffer.clear()

    def close(self):
        self.flush()

    def status(self):
        return {
            "diverging": self.diverging,
            "stalled": self.stalled,
            "converged": self.converged
        }
