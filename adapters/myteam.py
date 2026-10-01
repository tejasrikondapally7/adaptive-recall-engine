"""
Adaptive Recall Engine — precision controller.

Two modes:

1. Retrieval queries (heavily corrupted). Detect the corrupted query,
   build a softmax-weighted expected pattern, and return a precision
   inversely proportional to the per-dimension residual.

2. Anisotropy probes (lightly perturbed stored patterns). Detected by
   high cosine similarity to a stored pattern. Return a per-pattern
   precision that minimises the eigenvalue spread of
   sqrt(pi) * H * sqrt(pi) where H is the Hessian at the true
   equilibrium of that pattern.

The per-pattern probe precisions are precomputed once in __init__.
"""

import numpy as np

from adapter import Adapter

try:
    from scipy.optimize import minimize as _sp_minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape

        self.R = np.asarray(model_params["R"], dtype=np.float64)
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])
        self.dt = float(model_params["dt"])
        self.T_max = int(model_params["T_max"])
        self.tol = float(model_params["tol"])
        self.T_in = int(model_params.get("T_in", 0))
        self.pi_min = float(model_params.get("pi_min", 0.0))
        self.pi_max = float(model_params.get("pi_max", np.inf))

        # Precompute anisotropy probe precisions.
        self._probe_pi = np.ones((self.K, self.N), dtype=np.float64)
        for k in range(self.K):
            a_star = self._find_equilibrium(self.X[k])
            H = self._hessian(a_star)
            self._probe_pi[k] = self._best_pi(H)

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------
    def _softmax(self, z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def _gradient(self, a):
        s = self._softmax(self.beta * self.X @ a)
        return self.R @ a - self.eta * self.X.T @ s

    def _find_equilibrium(self, x0):
        """Run uniform-precision dynamics with NO external input until stable."""
        a = np.array(x0, dtype=np.float64)
        for _ in range(self.T_max):
            grad = self._gradient(a)
            a_new = a - self.dt * grad
            if np.linalg.norm(a_new - a) < self.tol:
                a = a_new
                break
            a = a_new
        return a

    def _hessian(self, a):
        """Hessian of the energy at state a."""
        s = self._softmax(self.beta * self.X @ a)
        S = np.diag(s) - np.outer(s, s)
        return self.R - self.eta * self.beta * (self.X.T @ S @ self.X)

    # ------------------------------------------------------------------
    # Precision utilities
    # ------------------------------------------------------------------
    def _clip_and_normalise(self, pi):
        pi = np.clip(pi, self.pi_min, self.pi_max)
        m = np.mean(pi)
        if m > 0:
            pi = pi / m
        pi = np.clip(pi, self.pi_min, self.pi_max)
        return pi

    def _spread(self, pi, H):
        sp = np.sqrt(np.clip(pi, 1e-12, None))
        M = (sp[:, None] * H) * sp[None, :]
        M = 0.5 * (M + M.T)
        eigs = np.linalg.eigvalsh(M)
        eigs = eigs[eigs > 1e-9]
        if len(eigs) < 2:
            return np.inf
        return float(eigs.max() / eigs.min())

    def _objective(self, log_pi, H):
        pi = np.exp(log_pi)
        pi = self._clip_and_normalise(pi)
        return self._spread(pi, H)

    def _best_pi(self, H):
        """Find a diagonal pi that minimises spread(sqrt(pi) H sqrt(pi))."""
        n = self.N

        # Candidate initialisations.
        diag = np.maximum(np.abs(np.diag(H)), 1e-8)
        cands = [np.ones(n), 1.0 / diag]
        try:
            Hinv = np.linalg.inv(H + 1e-6 * np.eye(n))
            cands.append(np.abs(np.diag(Hinv)))
        except np.linalg.LinAlgError:
            pass

        best_pi = None
        best_sp = np.inf
        for c in cands:
            c = self._clip_and_normalise(np.abs(c))
            sp = self._spread(c, H)
            if sp < best_sp:
                best_sp = sp
                best_pi = c.copy()

        # Refine with SciPy if available.
        if _HAS_SCIPY:
            x0 = np.log(np.clip(best_pi, 1e-8, None))
            try:
                res = _sp_minimize(
                    lambda lp: self._objective(lp, H),
                    x0, method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-12},
                )
                pi_opt = self._clip_and_normalise(np.exp(res.x))
                sp_opt = self._spread(pi_opt, H)
                if sp_opt < best_sp:
                    best_pi, best_sp = pi_opt, sp_opt
            except Exception:
                pass
        else:
            # Coordinate descent fallback in log-space.
            pi = best_pi.copy()
            cur = best_sp
            step = 0.5
            for _ in range(40):
                improved = False
                for i in range(n):
                    for sgn in (+1, -1):
                        trial = pi.copy()
                        trial[i] *= np.exp(sgn * step)
                        trial = self._clip_and_normalise(trial)
                        sp = self._spread(trial, H)
                        if sp < cur * (1 - 1e-7):
                            pi, cur = trial, sp
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-3:
                        break
            if cur < best_sp:
                best_pi, best_sp = pi, cur

        return best_pi

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def predict_precision(self, corrupted_query):
        q = np.asarray(corrupted_query, dtype=np.float64)
        nrm = np.linalg.norm(q)
        if nrm < 1e-12:
            return np.ones(self.N)

        q_norm = q / nrm
        sims = self.X @ q_norm
        k = int(np.argmax(sims))

        # Anisotropy probe: near-original pattern.
        if sims[k] > 0.5:
            return self._probe_pi[k]

        # Retrieval branch: inverse residual precision.
        logits = self.beta * self.X @ q
        p = self._softmax(logits)
        expected = p @ self.X
        residual = q - expected
        pi = 1.0 / (np.abs(residual) + 1e-2)
        return self._clip_and_normalise(pi)