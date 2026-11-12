import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RF_NEXT_DAY_FPR = 0.0014  # random forest, next-day test (results_summary.txt)

plt.rcParams.update({"font.family": "serif", "font.size": 9})
dt = np.logspace(0, np.log10(60), 200)  # 1 s to 60 s

fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
for fpr, style, label in [(0.01, "-", r"FPR$_w$ 1%"),
                          (0.001, "--", "0.1%"),
                          (0.0001, ":", "0.01%")]:
    ax.plot(dt, 86400 / dt * fpr, style, color="black", lw=1.4, label=label)
ax.plot(dt, 86400 / dt * RF_NEXT_DAY_FPR, "-.", color="gray", lw=1.4,
        label=f"RF ({RF_NEXT_DAY_FPR:.2%})")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 60)
ax.set_ylim(0.1, 1e4)
ticks = [1, 2, 5, 10, 30, 60]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.minorticks_off()
ax.set_xlabel(r"Window length $\Delta t$ (s)")
ax.set_ylabel("False alarms per day")
ax.grid(True, which="major", color="0.88", lw=0.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
          fontsize=7, frameon=False, handlelength=2.2, columnspacing=1.0)

os.makedirs("results", exist_ok=True)
out = os.path.join("results", "fig_window_calculated.png")
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
print("Saved", out)
