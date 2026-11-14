"""Reproduce HW2 EDA from pinned public snapshots; never execute dataset code."""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def partition(key: str) -> str:
    # Stable across Python versions and input row ordering.
    value = int(digest("smart-router-hw2-seed42:" + key)[:8], 16) / 2**32
    return "profile" if value < 0.6 else "validation" if value < 0.8 else "test"


def load_tasks() -> list[dict]:
    lock = json.loads((ROOT / "data/sources.lock.json").read_text())
    for item in lock["files"]:
        content = (ROOT / item["file"]).read_bytes()
        upstream = content if item["url"].endswith(".gz") else gzip.decompress(content)
        if hashlib.sha256(upstream).hexdigest() != item["upstream_sha256"]:
            raise ValueError(f"Upstream checksum mismatch: {item['file']}")
    tasks = []
    for lang in ("en", "ru"):
        path = ROOT / f"data/raw/mgsm_{lang}.tsv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for i, (question, answer) in enumerate(csv.reader(stream, delimiter="\t")):
                tasks.append(
                    {
                        "task_id": f"mgsm:{i}:{lang}",
                        "pair_id": f"mgsm:{i}",
                        "group_id": f"mgsm:{i}",
                        "source": "MGSM",
                        "domain": "math",
                        "language": lang,
                        "prompt": question,
                        "target": answer,
                        "valid_target": bool(
                            re.fullmatch(r"-?\d+(\.\d+)?", answer.replace(",", ""))
                        ),
                        "numeric_separator": ("," in answer),
                        "reference_present": bool(answer.strip()),
                    }
                )
    for lang, code in (("en", "eng_Latn"), ("ru", "rus_Cyrl")):
        path = ROOT / f"data/raw/belebele_{code}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                # Released HF files have a date-like `ds`; do not assume it is a passage ID.
                # ds differs across translations, so group by source link (conservative).
                # Prompt hash also joins passages with identical normalized text below.
                pair = "belebele:" + digest(row["link"] + ":" + str(row["question_number"]))
                passage = "belebele:" + digest(row["link"])
                answers = [row[f"mc_answer{i}"] for i in range(1, 5)]
                prompt = row["flores_passage"] + "\n\n" + row["question"] + "\n"
                prompt += "\n".join(f"{i}. {a}" for i, a in enumerate(answers, 1))
                tasks.append(
                    {
                        "task_id": f"{pair}:{lang}",
                        "pair_id": pair,
                        "group_id": passage,
                        "source": "Belebele",
                        "domain": "general",
                        "language": lang,
                        "prompt": prompt,
                        "target": str(row["correct_answer_num"]),
                        "valid_target": str(row["correct_answer_num"]) in {"1", "2", "3", "4"}
                        and all(answers)
                        and row["dialect"] == code,
                        "reference_present": bool(row["correct_answer_num"]),
                        "duplicate_options": (len(set(answers)) != 4),
                        "passage_hash": digest(normalize(row["flores_passage"])),
                    }
                )
    with gzip.open(ROOT / "data/raw/HumanEval.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            try:
                ast.parse(row["prompt"] + row["canonical_solution"])
                ast.parse(row["test"])
                syntax_ok = True
            except SyntaxError:
                syntax_ok = False
            tasks.append(
                {
                    "task_id": row["task_id"],
                    "pair_id": row["task_id"],
                    "group_id": row["task_id"],
                    "source": "HumanEval",
                    "domain": "python",
                    "language": "en",
                    "prompt": row["prompt"],
                    "target": row["entry_point"],
                    "valid_target": syntax_ok,
                    "reference_present": all(
                        row[k].strip() for k in ("canonical_solution", "test", "entry_point")
                    ),
                    "test_asserts": sum(
                        isinstance(n, ast.Assert) for n in ast.walk(ast.parse(row["test"]))
                    ),
                }
            )
    return tasks


def merge_groups(tasks: list[dict]) -> None:
    """Keep translations, shared passages and normalized exact duplicates together."""
    parent = {row["group_id"]: row["group_id"] for row in tasks}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for row in tasks:
        for token in (
            "prompt:" + digest(normalize(row["prompt"])),
            "passage:" + row.get("passage_hash", row["task_id"]),
        ):
            if token in seen:
                a, b = find(row["group_id"]), find(seen[token])
                parent[max(a, b)] = min(a, b)
            seen[token] = row["group_id"]
    for row in tasks:
        row["group_id"] = find(row["group_id"])


def save_plot(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=160, facecolor="white")
    plt.close()


def run() -> dict:
    for path in (TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    for row in tasks:
        row.setdefault("numeric_separator", False)
        row.setdefault("duplicate_options", False)
    merge_groups(tasks)
    # Tokenizer is diagnostic only; the first run downloads its public vocabulary.
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    for row in tasks:
        row["prompt_hash"] = digest(normalize(row["prompt"]))
        row["characters"] = len(row["prompt"])
        row["utf8_bytes"] = len(row["prompt"].encode())
        row["tokens_proxy"] = len(encoder.encode(row["prompt"], disallowed_special=()))
        row["naive_split"] = partition(row["task_id"])
        row["grouped_split"] = partition(row["group_id"])
    frame = pd.DataFrame(tasks)
    metadata = frame.drop(columns=["prompt", "target", "passage_hash"])
    metadata.to_csv(TABLES / "task_metadata.csv", index=False)
    coverage = pd.crosstab(frame.domain, frame.language).reindex(
        index=["general", "math", "python"], columns=["en", "ru"], fill_value=0
    )
    coverage.to_csv(TABLES / "coverage.csv")
    lengths = frame.groupby(["source", "language"]).tokens_proxy.agg(
        count="count", minimum="min", median="median", p95=lambda x: x.quantile(0.95), maximum="max"
    )
    lengths.to_csv(TABLES / "lengths.csv")
    answers = frame[frame.source == "Belebele"].groupby(["language", "target"]).size()
    answers.to_csv(TABLES / "answer_positions.csv", header=["count"])
    split_table = frame.groupby(["source", "language", "grouped_split"]).size()
    split_table.to_csv(TABLES / "illustrative_splits.csv", header=["count"])
    pairs = (
        frame[frame.source.isin(["MGSM", "Belebele"])]
        .pivot(index="pair_id", columns="language", values="tokens_proxy")
        .dropna()
    )
    ratios = pairs.ru / pairs.en
    aligned = frame[frame.source.isin(["MGSM", "Belebele"])].pivot(
        index="pair_id", columns="language", values="target"
    )
    naive = frame.groupby("group_id").naive_split.nunique()
    grouped = frame.groupby("group_id").grouped_split.nunique()
    leaks = set(naive[naive > 1].index)
    summary = {
        "snapshot_date": "2026-11-14",
        "rows": len(frame),
        "columns": len(metadata.columns),
        "groups": int(frame.group_id.nunique()),
        "pairs": len(pairs),
        "sources": frame.source.value_counts().to_dict(),
        "coverage": coverage.to_dict(),
        "missing_prompts": int(frame.prompt.str.strip().eq("").sum()),
        "missing_references": int((~frame.reference_present).sum()),
        "invalid_targets": int((~frame.valid_target).sum()),
        "references_with_numeric_separators": int(frame.numeric_separator.sum()),
        "duplicate_option_rows": int(frame.duplicate_options.sum()),
        "eligible_rows": int((~frame.duplicate_options).sum()),
        "duplicate_ids": int(frame.task_id.duplicated().sum()),
        "duplicate_normalized_prompts": int(frame.prompt_hash.duplicated().sum()),
        "cross_language_target_disagreements": int((aligned.en != aligned.ru).sum()),
        "naive_leaking_groups": len(leaks),
        "naive_leaking_rows": int(frame.group_id.isin(leaks).sum()),
        "grouped_leaking_groups": int((grouped > 1).sum()),
        "token_proxy": "cl100k_base, tiktoken 0.12.0, no chat template",
        "ru_en_token_ratio_median": float(ratios.median()),
        "proxy_over_4096": int((frame.tokens_proxy > 4096).sum()),
        "proxy_over_512": int((frame.tokens_proxy > 512).sum()),
        "total_input_tokens_proxy": int(frame.tokens_proxy.sum()),
        "max_prompt_tokens_proxy": int(frame.tokens_proxy.max()),
        "humaneval_asserts": frame.loc[frame.source == "HumanEval", "test_asserts"]
        .describe()
        .to_dict(),
    }
    (ROOT / "reports/eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(coverage, cmap="Blues", vmin=0, vmax=900)
    for i in range(3):
        for j in range(2):
            n = coverage.iloc[i, j]
            ax.text(j, i, str(n), ha="center", va="center", color="white" if n > 500 else "black")
    ax.set(
        xticks=[0, 1],
        xticklabels=["English", "Russian"],
        yticks=range(3),
        yticklabels=["Reading comprehension", "Math", "Python"],
        title="Available tasks by domain and language",
    )
    fig.colorbar(image, ax=ax, label="Tasks")
    save_plot("coverage.png")
    plt.figure(figsize=(9, 4))
    cells = list(frame.groupby(["source", "language"], sort=True))
    plt.boxplot(
        [g.tokens_proxy for _, g in cells],
        tick_labels=[" / ".join(k) for k, _ in cells],
        showfliers=True,
    )
    plt.axhline(512, color="#b35806", linestyle="--", label="512 diagnostic tokens")
    plt.ylabel("cl100k_base tokens; no chat template")
    plt.title("Prompt lengths vary by language and source")
    plt.legend()
    save_plot("lengths.png")
    plt.figure(figsize=(7, 4))
    for source, group in pairs.groupby(pairs.index.str.split(":").str[0]):
        plt.scatter(group.en, group.ru, s=12, alpha=0.45, label=source)
    bound = float(pairs.max().max())
    plt.plot([0, bound], [0, bound], "k--", label="Equal token count")
    plt.xlabel("English prompt tokens")
    plt.ylabel("Russian prompt tokens")
    plt.title("Paired translations: tokenizer-dependent length")
    plt.legend()
    save_plot("translation_lengths.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    answers.unstack(0).plot.bar(ax=ax, rot=0, color=["#2166ac", "#b35806"])
    ax.set(
        xlabel="Correct answer position (1-based)",
        ylabel="Tasks",
        title="Belebele reference answer positions",
    )
    save_plot("answer_positions.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    values = [len(leaks), summary["grouped_leaking_groups"]]
    bars = ax.bar(
        ["Independent row assignment", "Shared group assignment"],
        values,
        color=["#b35806", "#2166ac"],
    )
    ax.bar_label(bars, padding=3)
    ax.set(
        ylabel="Groups present in multiple partitions",
        ylim=(0, max(values) * 1.18),
        title="Illustrative 60/20/20 split: leakage through related tasks",
    )
    save_plot("split_leakage.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = frame.groupby("source").group_id.nunique()
    counts.plot.bar(ax=ax, rot=0, color="#2166ac")
    for rect in ax.patches:
        ax.annotate(
            str(int(rect.get_height())),
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            ha="center",
            va="bottom",
        )
    ax.set(ylabel="Conservative dependency groups", title="Rows do not equal independent tasks")
    save_plot("independent_groups.png")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
