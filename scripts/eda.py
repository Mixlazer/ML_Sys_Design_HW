from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken

if __package__:
    from .fetch_data import read_lock, read_snapshot_text
else:
    from fetch_data import read_lock, read_snapshot_text

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"
MATH_NUMBER = re.compile(r"-?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")
LANGUAGES = {"en": "Английский", "ru": "Русский"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def partition(key: str) -> str:
    value = int(digest("smart-router-hw2-seed42:" + key)[:8], 16) / 2**32
    return "profile" if value < 0.6 else "validation" if value < 0.8 else "test"


def mgsm_target(answer: str) -> Decimal | None:
    answer = answer.strip()
    if MATH_NUMBER.fullmatch(answer):
        return Decimal(answer.replace(",", ""))
    return None


def mgsm_tasks(snapshots: dict[str, str], replacements: dict[str, list[int]]) -> list[dict]:
    tasks = []
    for language in LANGUAGES:
        replaced = set(replacements.get(language, []))
        rows = csv.reader(io.StringIO(snapshots[f"mgsm_{language}.tsv.gz"]), delimiter="\t")
        for index, (question, answer) in enumerate(rows):
            pair_id = f"mgsm:{index}"
            target = mgsm_target(answer)
            tasks.append(
                {
                    "task_id": f"{pair_id}:{language}",
                    "pair_id": pair_id,
                    "group_id": pair_id,
                    "source": "MGSM",
                    "domain": "math",
                    "language": language,
                    "prompt": question,
                    "target": target,
                    "valid_target": target is not None,
                    "numeric_separator": "," in answer and target is not None,
                    "reference_present": bool(answer.strip()),
                    "prompt_present": bool(question.strip()),
                    "upstream_revised": index + 1 in replaced,
                }
            )
    return tasks


def belebele_tasks(snapshots: dict[str, str]) -> list[dict]:
    tasks = []
    for language, dialect in (("en", "eng_Latn"), ("ru", "rus_Cyrl")):
        for row in map(json.loads, io.StringIO(snapshots[f"belebele_{dialect}.jsonl.gz"])):
            pair_id = "belebele:" + digest(f"{row['link']}:{row['question_number']}")
            passage_id = "belebele:" + digest(row["link"])
            passage = row["flores_passage"]
            question = row["question"]
            answers = [row[f"mc_answer{i}"] for i in range(1, 5)]
            normalized = list(map(normalize, answers))
            correct = str(row["correct_answer_num"])
            valid_position = correct in {"1", "2", "3", "4"}
            repeated_gold = valid_position and normalized.count(normalized[int(correct) - 1]) > 1
            formatted_answers = "\n".join(f"{i}. {answer}" for i, answer in enumerate(answers, 1))
            tasks.append(
                {
                    "task_id": f"{pair_id}:{language}",
                    "pair_id": pair_id,
                    "group_id": passage_id,
                    "source": "Belebele",
                    "domain": "general",
                    "language": language,
                    "prompt": f"{passage}\n\n{question}\n{formatted_answers}",
                    "target": correct,
                    "valid_target": valid_position
                    and all(normalized)
                    and row["dialect"] == dialect,
                    "reference_present": bool(str(row["correct_answer_num"] or "").strip()),
                    "prompt_present": bool(
                        passage.strip() and question.strip() and all(normalized)
                    ),
                    "duplicate_options": len(set(normalized)) != len(answers),
                    "duplicate_correct_option": repeated_gold,
                    "passage_hash": digest(normalize(passage)),
                }
            )
    return tasks


def python_reference(
    prompt: str, solution: str, test: str, entry_point: str
) -> tuple[bool, int | None]:
    try:
        solution_ast = ast.parse(prompt + solution)
        test_ast = ast.parse(test)
    except SyntaxError:
        return False, None

    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    has_entry_point = any(
        isinstance(node, functions) and node.name == entry_point for node in solution_ast.body
    )
    has_check = any(isinstance(node, functions) and node.name == "check" for node in test_ast.body)
    asserts = sum(isinstance(node, ast.Assert) for node in ast.walk(test_ast))
    return has_entry_point and has_check, asserts


def humaneval_tasks(snapshot: str) -> list[dict]:
    tasks = []
    for row in map(json.loads, io.StringIO(snapshot)):
        task_id = row["task_id"]
        prompt = row["prompt"]
        solution = row["canonical_solution"]
        test = row["test"]
        entry_point = row["entry_point"]
        valid_target, asserts = python_reference(prompt, solution, test, entry_point)
        tasks.append(
            {
                "task_id": task_id,
                "pair_id": task_id,
                "group_id": task_id,
                "source": "HumanEval",
                "domain": "python",
                "language": "en",
                "prompt": prompt,
                "target": entry_point,
                "valid_target": valid_target,
                "reference_present": all(field.strip() for field in (solution, test, entry_point)),
                "prompt_present": bool(prompt.strip()),
                "test_asserts": asserts,
            }
        )
    return tasks


def load_tasks(lock: dict | None = None) -> list[dict]:
    if lock is None:
        lock = read_lock()
    snapshots = {
        Path(item["file"]).name: read_snapshot_text(item)
        for item in [*lock["files"], *lock.get("audit_sources", [])]
    }
    replacements = json.loads(snapshots.get("mgsm_rev2_replaced_questions.json.gz", "{}"))
    return [
        *mgsm_tasks(snapshots, replacements),
        *belebele_tasks(snapshots),
        *humaneval_tasks(snapshots["HumanEval.jsonl.gz"]),
    ]


def merge_groups(tasks: list[dict]) -> None:
    parent = {row["group_id"]: row["group_id"] for row in tasks}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for row in tasks:
        tokens = [("prompt", digest(normalize(row["prompt"])))]
        if row.get("pair_id"):
            tokens.append(("pair", row["pair_id"]))
        if row.get("passage_hash"):
            tokens.append(("passage", row["passage_hash"]))
        for token in tokens:
            group = row["group_id"]
            if token in seen:
                first, second = sorted((find(group), find(seen[token])))
                parent[second] = first
            seen[token] = group

    for row in tasks:
        row["group_id"] = find(row["group_id"])


def prepare_frame(tasks: list[dict]) -> pd.DataFrame:
    merge_groups(tasks)
    encoder = tiktoken.get_encoding("cl100k_base")
    for row in tasks:
        for flag in (
            "numeric_separator",
            "duplicate_options",
            "duplicate_correct_option",
            "upstream_revised",
        ):
            row.setdefault(flag, False)
        prompt = row["prompt"]
        row["prompt_hash"] = digest(normalize(prompt))
        row["characters"] = len(prompt)
        row["utf8_bytes"] = len(prompt.encode("utf-8"))
        row["tokens_proxy"] = len(encoder.encode_ordinary(prompt))
        row["naive_split"] = partition(row["task_id"])
        row["grouped_split"] = partition(row["group_id"])
        row["eligible"] = (
            row["prompt_present"]
            and row["reference_present"]
            and row["valid_target"]
            and not row["duplicate_options"]
        )
    frame = pd.DataFrame.from_records(tasks)
    duplicates = frame.loc[frame.task_id.duplicated(), "task_id"]
    if not duplicates.empty:
        raise ValueError(f"Повторяющийся task_id: {duplicates.iloc[0]}")
    return frame


def translation_pairs(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        frame.loc[frame.source.isin(["MGSM", "Belebele"])]
        .pivot(index=["source", "pair_id"], columns="language", values=value_column)
        .reindex(columns=LANGUAGES)
    )


def test_exposure(frame: pd.DataFrame) -> dict:
    result = {}
    for name in ("naive_split", "grouped_split"):
        in_test = frame[name].eq("test")
        earlier_groups = frame.loc[~in_test, "group_id"]
        in_test_frame = frame.loc[in_test, ["source", "group_id"]].copy()
        in_test_frame["related_to_development"] = in_test_frame.group_id.isin(earlier_groups)
        result[name] = (
            in_test_frame.groupby("source")
            .related_to_development.agg(test_rows="size", related_rows="sum")
            .to_dict(orient="index")
        )
    return result


def summarize(
    frame: pd.DataFrame,
    coverage: pd.DataFrame,
    pairs: pd.DataFrame,
    snapshot_date: str,
    columns: int,
) -> dict:
    ratios = pairs.ru / pairs.en
    valid_references = frame.loc[frame.valid_target & frame.reference_present]
    aligned = translation_pairs(valid_references, "target").dropna()
    revised_pairs = frame.loc[frame.upstream_revised, "pair_id"].unique()
    naive = frame.groupby("group_id").naive_split.nunique()
    grouped = frame.groupby("group_id").grouped_split.nunique()
    leaks = naive[naive > 1].index
    return {
        "snapshot_date": snapshot_date,
        "rows": len(frame),
        "columns": columns,
        "groups": int(frame.group_id.nunique()),
        "pairs": len(pairs),
        "incomplete_translation_pairs": len(translation_pairs(frame, "tokens_proxy")) - len(pairs),
        "sources": frame.source.value_counts().to_dict(),
        "coverage": coverage.to_dict(),
        "missing_prompts": int((~frame.prompt_present).sum()),
        "missing_references": int((~frame.reference_present).sum()),
        "invalid_targets": int((~frame.valid_target).sum()),
        "references_with_numeric_separators": int(frame.numeric_separator.sum()),
        "duplicate_option_rows": int(frame.duplicate_options.sum()),
        "duplicate_correct_option_rows": int(frame.duplicate_correct_option.sum()),
        "eligible_rows": int(frame.eligible.sum()),
        "mgsm_rev2_replaced_rows_by_language": (
            frame.loc[frame.upstream_revised].groupby("language").size().to_dict()
        ),
        "mgsm_rev2_affected_pairs": len(revised_pairs),
        "eligible_rows_excluding_revised_pairs": int(
            (frame.eligible & ~frame.pair_id.isin(revised_pairs)).sum()
        ),
        "duplicate_ids": int(frame.task_id.duplicated().sum()),
        "duplicate_normalized_prompts": int(frame.prompt_hash.duplicated().sum()),
        "compared_cross_language_targets": len(aligned),
        "cross_language_target_disagreements": int((aligned.en != aligned.ru).sum()),
        "naive_leaking_groups": len(leaks),
        "naive_leaking_rows": int(frame.group_id.isin(leaks).sum()),
        "grouped_leaking_groups": int((grouped > 1).sum()),
        "heldout_test_exposure": test_exposure(frame),
        "token_proxy": f"cl100k_base, tiktoken {tiktoken.__version__}, no chat template",
        "ru_en_token_ratio_median": float(ratios.median()) if not ratios.empty else None,
        "ru_en_token_ratio_median_by_source": ratios.groupby("source").median().to_dict(),
        "proxy_over_4096": int((frame.tokens_proxy > 4096).sum()),
        "proxy_over_512": int((frame.tokens_proxy > 512).sum()),
        "total_input_tokens_proxy": int(frame.tokens_proxy.sum()),
        "max_prompt_tokens_proxy": int(frame.tokens_proxy.max()),
        "humaneval_asserts": frame.loc[frame.source == "HumanEval", "test_asserts"]
        .describe()
        .dropna()
        .to_dict(),
    }


def save_plot(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160, facecolor="white")
    plt.close(fig)


def plot_coverage(coverage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(coverage, cmap="Blues", aspect="auto", vmin=0)
    for i, row in enumerate(coverage.itertuples(index=False, name=None)):
        for j, count in enumerate(row):
            color = "white" if image.norm(count) > 0.6 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)
    ax.set(
        xticks=range(len(LANGUAGES)),
        xticklabels=list(LANGUAGES.values()),
        yticks=range(len(coverage.index)),
        yticklabels=["Понимание текста", "Математика", "Python"],
        title="Задания по доменам и языкам",
    )
    fig.colorbar(image, ax=ax, label="Число заданий")
    save_plot(fig, "coverage.png")


def plot_lengths(frame: pd.DataFrame, pairs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    cells = list(frame.groupby(["source", "language"], sort=True))
    ax.boxplot(
        [group.tokens_proxy for _, group in cells],
        tick_labels=[" / ".join(key) for key, _ in cells],
    )
    ax.set(ylabel="Токены cl100k_base", title="Длина условий по наборам и языкам")
    save_plot(fig, "lengths.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for source, group in pairs.groupby(level="source"):
        ax.scatter(group.en, group.ru, s=12, alpha=0.45, label=source)
    bound = float(pairs.en.max())
    ax.plot([0, bound], [0, bound], "k--", label="Равная длина")
    ax.set(
        xlabel="Токены в английском условии",
        ylabel="Токены в русском условии",
        title="Длина условий в парах переводов",
    )
    ax.legend()
    save_plot(fig, "translation_lengths.png")


def plot_answers(answers: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    answers.rename(columns=LANGUAGES).plot.bar(ax=ax, rot=0, color=["#2166ac", "#b35806"])
    ax.set(
        xlabel="Позиция правильного ответа",
        ylabel="Число заданий",
        title="Позиции эталонных ответов Belebele",
        ylim=(0, answers.max().max() * 1.35),
    )
    ax.legend()
    save_plot(fig, "answer_positions.png")


def plot_groups(frame: pd.DataFrame, summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = frame.groupby("source").agg(Строки=("task_id", "size"), Группы=("group_id", "nunique"))
    counts.plot.bar(ax=ax, rot=0, color=["#98bed9", "#2166ac"])
    for bars in ax.containers:
        ax.bar_label(bars, padding=3)
    ax.set(xlabel="", ylabel="Количество", ylim=(0, counts.max().max() * 1.18))
    ax.set_title("Строки и группы связанных заданий")
    save_plot(fig, "independent_groups.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    totals = [summary["naive_leaking_groups"], summary["grouped_leaking_groups"]]
    bars = ax.bar(
        ["По отдельным строкам", "По связанным группам"], totals, color=["#b35806", "#2166ac"]
    )
    ax.bar_label(bars, padding=3)
    ax.set(
        ylabel="Группы в нескольких частях",
        ylim=(0, max(1, max(totals) * 1.18)),
        title="Пересечения при разбиении 60/20/20",
    )
    save_plot(fig, "split_leakage.png")


def run() -> dict:
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    lock = read_lock()
    frame = prepare_frame(load_tasks(lock))
    metadata = frame.drop(columns=["prompt", "target", "passage_hash"])
    coverage = pd.crosstab(frame.domain, frame.language).reindex(
        index=["general", "math", "python"], columns=LANGUAGES, fill_value=0
    )
    lengths = frame.groupby(["source", "language"]).tokens_proxy.agg(
        count="count", minimum="min", median="median", p95=lambda x: x.quantile(0.95), maximum="max"
    )
    answers = frame.loc[frame.source.eq("Belebele")].groupby(["language", "target"]).size()
    split_table = frame.groupby(["source", "language", "grouped_split"]).size()
    pairs = translation_pairs(frame, "tokens_proxy").dropna()
    summary = summarize(frame, coverage, pairs, lock["snapshot_date"], len(metadata.columns))

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(TABLES / "task_metadata.csv", index=False, lineterminator="\n")
    coverage.to_csv(TABLES / "coverage.csv", lineterminator="\n")
    lengths.to_csv(TABLES / "lengths.csv", lineterminator="\n")
    answers.to_csv(TABLES / "answer_positions.csv", header=["count"], lineterminator="\n")
    split_table.to_csv(TABLES / "illustrative_splits.csv", header=["count"], lineterminator="\n")

    plot_coverage(coverage)
    plot_lengths(frame, pairs)
    plot_answers(answers.unstack(0))
    plot_groups(frame, summary)
    rendered_summary = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    (ROOT / "reports/eda_summary.json").write_text(
        rendered_summary + "\n", encoding="utf-8", newline="\n"
    )
    print(rendered_summary)
    return summary


if __name__ == "__main__":
    run()
