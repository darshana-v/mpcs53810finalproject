import pandas as pd, matplotlib.pyplot as plt
sc = pd.read_csv("strategy_comparison.csv")
ae = pd.read_csv("adaptive_evolution.csv")

# Figure 1 – match-rate by strategy
plt.figure()
plt.bar(sc["strategy"], sc["match_rate"])
plt.ylabel("Match rate")
plt.title("One-year comparison")
plt.savefig("fig_strategy_compare.png", dpi=300)

# Figure 2 – adaptive learning over 20 years
plt.figure()
plt.plot(ae["year"], ae["match_rate"], marker='o')
plt.xlabel("Year")
plt.ylabel("Match rate")
plt.title("Adaptive strategy: convergence")
plt.savefig("fig_adaptive.png", dpi=300)
