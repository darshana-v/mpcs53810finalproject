import numpy as np
import pandas as pd

# Set random seed for reproducibility
# np.random.seed(42)
rng = np.random.default_rng(0)

# Parameters
num_students = 4000
num_colleges = 100
college_capacity = 20
applications_per_student = 10
tau = 0.15
years = 20
strategies = ["Top10", "Random10", "RTS", "EU", "Adaptive"]

# helper: generate market fundamentals
def generate_market():
    """Return (student_scores, college_scores, epsilon, delta)."""
    student_scores = rng.uniform(0, 1, num_students) # c_i
    college_scores = rng.uniform(0, 1, num_colleges) # S_j
    epsilon = rng.uniform(0, 1, (num_students, num_colleges)) # ε_ij
    delta = rng.uniform(0, 1, (num_colleges, num_students)) # δ_ji
    return student_scores, college_scores, epsilon, delta

# acceptance probability
def prob_accept(c_i, S_j):
    """Acceptance probability  exp(c_i - S_j)."""
    return np.exp(c_i - S_j)

# strategy blocks
def apps_top10(P):
    """Top-10 by perceived utility P."""
    return np.argsort(-P, axis=1)[:, :applications_per_student]

def apps_random():
    """Return a (num_students x K) array of distinct random colleges."""
    out = np.empty((num_students, applications_per_student), dtype=int)
    for i in range(num_students):
        out[i] = rng.choice(num_colleges,
                            size=applications_per_student,
                            replace=False)
    return out

def apps_rts(P, student_scores, college_scores):
    """Reach-Target-Safety (3-4-3 split)."""
    reach_mask  = college_scores[np.newaxis, :] > student_scores[:, None] + tau
    safe_mask   = college_scores[np.newaxis, :] < student_scores[:, None] - tau
    target_mask = ~(reach_mask | safe_mask)
    indices = np.empty((num_students, applications_per_student), int)
    for i in range(num_students):
        picks = []
        # reach (≤3)
        reach_cands = np.where(reach_mask[i])[0]
        picks += rng.choice(reach_cands,
                            size=min(3, len(reach_cands)),
                            replace=False).tolist()
        # target (≤4)
        target_cands = np.where(target_mask[i])[0]
        picks += rng.choice(target_cands,
                            size=min(4, len(target_cands)),
                            replace=False).tolist()
        # safety (fill to 10)
        safe_cands = np.where(safe_mask[i])[0]
        need = applications_per_student - len(picks)
        picks += rng.choice(safe_cands, size=min(need, len(safe_cands)),
                            replace=False).tolist()
        # pad randomly if still short (rare edge case)
        while len(picks) < applications_per_student:
            cand = rng.integers(num_colleges)
            if cand not in picks:
                picks.append(cand)
        indices[i] = picks
    return indices

def apps_eu(P, student_scores, college_scores):
    """10 highest expected utilities P_ij * Pr(accept)."""
    exp_u = P * np.exp(student_scores[:, None] - college_scores[None, :])
    return np.argsort(-exp_u, axis=1)[:, :applications_per_student]

# matching engine (one academic year)
def run_year(app_matrix, P, student_scores, college_scores):
    """Return final_matches, utilities."""
    accepted = np.zeros((num_students, num_colleges), dtype=bool)

    # conditional acceptances
    for stu in range(num_students):
        for col in app_matrix[stu]:
            if rng.random() < prob_accept(student_scores[stu], college_scores[col]):
                accepted[stu, col] = True

    # colleges choose up to capacity
    final_matches = np.full(num_students, -1)
    for col in range(num_colleges):
        stu_pool = np.where(accepted[:, col])[0]
        if len(stu_pool) > 0:
            chosen = rng.choice(stu_pool,
                                size=min(college_capacity, len(stu_pool)),
                                replace=False)
            for s in chosen:
                # Students might have multiple provisional offers. Keep best P_ij.
                if final_matches[s] == -1 or P[s, col] > P[s, final_matches[s]]:
                    final_matches[s] = col

    utilities = np.where(final_matches != -1,
                         P[np.arange(num_students), final_matches],
                         0.0)
    match_rate = (final_matches != -1).mean()
    avg_util   = utilities[final_matches != -1].mean() if match_rate > 0 else 0
    return final_matches, utilities, match_rate, avg_util

# experiment: compare 4 static strategies
def static_strategy_comparison():
    (c_i, S_j, eps, delta) = generate_market()
    P = S_j + eps
    results = []
    for strat in ["Top10", "Random10", "RTS", "EU"]:
        if strat == "Top10":
            apps = apps_top10(P)
        elif strat == "Random10":
            apps = apps_random()
        elif strat == "RTS":
            apps = apps_rts(P, c_i, S_j)
        elif strat == "EU":
            apps = apps_eu(P, c_i, S_j)
        m, u, r, a = run_year(apps, P, c_i, S_j)
        results.append(dict(strategy=strat,
                            match_rate=r,
                            avg_util=a))
    pd.DataFrame(results).to_csv("strategy_comparison.csv", index=False)

# experiment: multi-year adaptive learners
def adaptive_experiment():
    history = []
    # initial "naïve" strategy = Top10
    stretch = tau
    for year in range(1, years + 1):
        (c_i, S_j, eps, delta) = generate_market()
        P = S_j + eps

        # students decide using current stretch parameter
        apps = apps_rts(P, c_i, S_j) if year == 1 else apps_rts(P, c_i, S_j + rng.normal(0, 0.0, num_colleges))
        m, u, r, a = run_year(apps, P, c_i, S_j)
        history.append((year, r, a, stretch))

        # update rule: unsuccessful → widen safety band
        unmatched = (m == -1)
        matched = ~unmatched
        # if >50 % unmatched increase stretch, else tighten slightly
        stretch +=  0.02 if unmatched.mean() > 0.5 else -0.01
        stretch = np.clip(stretch, 0.05, 0.30)

    pd.DataFrame(history,
                 columns=["year", "match_rate", "avg_util", "tau"]).to_csv(
                 "adaptive_evolution.csv", index=False)

if __name__ == "__main__":
    static_strategy_comparison()
    adaptive_experiment()
    print("Finished. CSV files written:", "strategy_comparison.csv, adaptive_evolution.csv")