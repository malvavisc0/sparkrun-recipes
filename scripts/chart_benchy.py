#!/usr/bin/env python3
"""Plot a llama-benchy CSV into a 4-panel PNG.

Usage:
    uvx --with pandas --with matplotlib python scripts/chart_benchy.py \
        benchmarks/<model>.csv benchmarks/<model>.png
"""
import re, sys, pandas as pd, matplotlib.pyplot as plt


def main(csv_path: str, png_path: str) -> None:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Column name normalization (llama-benchy CSV schema)
    colmap = {
        "test": "test_name",
        "t/s (total)": "t_s_mean",
        "t/s (req)": "t_s_req_mean",
        "peak t/s": "peak_ts_mean",
        "ttfr (ms)": "ttfr_mean",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    def parse_test(s: str) -> pd.Series:
        m = re.match(r"(ctx_pp|ctx_tg|pp|tg)\d*(?: @ d(\d+))? \(c(\d+)\)", str(s).strip())
        if not m:
            return pd.Series({"kind": None, "depth": None, "conc": None})
        return pd.Series({
            "kind": m.group(1),
            "depth": int(m.group(2) or 0),
            "conc": int(m.group(3)),
        })

    df = pd.concat([df, df["test_name"].apply(parse_test)], axis=1)
    df["tps"] = pd.to_numeric(df["t_s_mean"], errors="coerce")
    model = df["model"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"llama-benchy: {model}", fontsize=14)

    def lineplot(ax, kind: str, xcol: str, title: str, xlabel: str) -> None:
        sub = df[df.kind == kind]
        for grp, g in sub.groupby("depth" if xcol == "conc" else "conc"):
            g = g.sort_values(xcol)
            ax.plot(g[xcol], g.tps, marker="o", label=(
                f"d={grp}" if xcol == "conc" else f"c={grp}"
            ))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("t/s (total)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    lineplot(axes[0, 0], "pp", "conc",
             "Prompt processing (pp2048): t/s vs concurrency", "concurrency")
    axes[0, 0].set_xticks(sorted(df[df.kind == "pp"].conc.unique()))

    lineplot(axes[0, 1], "tg", "conc",
             "Token generation (tg128): t/s vs concurrency", "concurrency")
    axes[0, 1].set_xticks(sorted(df[df.kind == "tg"].conc.unique()))

    lineplot(axes[1, 0], "ctx_pp", "depth",
             "Context prompt processing: t/s vs depth", "depth")
    lineplot(axes[1, 1], "ctx_tg", "depth",
             "Context token generation: t/s vs depth", "depth")

    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    print("wrote", png_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: chart_benchy.py <input.csv> <output.png>")
    main(sys.argv[1], sys.argv[2])