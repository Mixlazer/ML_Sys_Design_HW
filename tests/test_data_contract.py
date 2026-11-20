"""Guard consequential ingestion and leakage assumptions of the pinned corpus."""

import unittest
from collections import Counter, defaultdict
from decimal import Decimal

from scripts.eda import (
    load_tasks,
    merge_groups,
    mgsm_target,
    normalize,
    partition,
    python_reference,
)


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks()

    def test_snapshot_schema_and_reference_coverage(self):
        self.assertEqual(len(self.tasks), 2464)
        self.assertEqual(len({r["task_id"] for r in self.tasks}), 2464)
        self.assertTrue(all(r["prompt_present"] for r in self.tasks))
        self.assertTrue(all(r["valid_target"] and r["reference_present"] for r in self.tasks))
        self.assertEqual(sum(r.get("numeric_separator", False) for r in self.tasks), 8)
        self.assertEqual(sum(r.get("duplicate_options", False) for r in self.tasks), 4)
        self.assertEqual(sum(r.get("duplicate_correct_option", False) for r in self.tasks), 2)

    def test_mgsm_revisions_are_mapped_to_original_rows(self):
        revised = [row for row in self.tasks if row.get("upstream_revised", False)]
        self.assertEqual(Counter(row["language"] for row in revised), {"en": 22, "ru": 32})
        self.assertEqual(len({row["pair_id"] for row in revised}), 32)

    def test_translation_pairs_and_gold_alignment(self):
        pairs = defaultdict(list)
        for row in self.tasks:
            if row["source"] != "HumanEval":
                pairs[row["pair_id"]].append(row)
        self.assertEqual(len(pairs), 1150)
        for pair in pairs.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual({r["language"] for r in pair}, {"en", "ru"})
            self.assertEqual(len({r["target"] for r in pair}), 1)

    def test_related_prompts_never_cross_grouped_partitions(self):
        tasks = [dict(row) for row in self.tasks]
        merge_groups(tasks)
        original_links = defaultdict(set)
        for original, merged in zip(self.tasks, tasks, strict=True):
            group_id = merged["group_id"]
            for key in ("group_id", "pair_id", "passage_hash"):
                if key in original:
                    original_links[key, original[key]].add((group_id, partition(group_id)))
            original_links["prompt", normalize(original["prompt"])].add(
                (group_id, partition(group_id))
            )
        self.assertEqual(len({row["group_id"] for row in tasks}), 902)
        self.assertTrue(all(len(group) == 1 for group in original_links.values()))

    def test_mgsm_number_requires_valid_thousands_groups(self):
        self.assertEqual(mgsm_target(" -1,205.50 "), Decimal("-1205.50"))
        for invalid in ("1,2", "1,,200", "1,000,2", "2,50.0"):
            with self.subTest(invalid=invalid):
                self.assertIsNone(mgsm_target(invalid))

    def test_humaneval_invalid_test_or_entry_point(self):
        prompt = "def func():\n"
        solution = "    return 1\n"
        check = "def check(candidate):\n    assert candidate() == 1\n"
        self.assertEqual(python_reference(prompt, solution, check, "func"), (True, 1))
        self.assertEqual(python_reference(prompt, solution, "def", "func"), (False, None))
        self.assertEqual(python_reference(prompt, solution, "pass", "func"), (False, 0))
        self.assertEqual(python_reference(prompt, solution, check, "missing"), (False, 1))

    def test_duplicate_union_is_transitive_and_order_independent(self):
        tasks = [
            {"task_id": "a", "group_id": "g1", "prompt": "Same  prompt"},
            {"task_id": "b", "group_id": "g2", "prompt": "SAME prompt"},
            {"task_id": "c", "group_id": "g2", "prompt": "Different"},
            {"task_id": "d", "group_id": "g3", "prompt": "Different"},
            {"task_id": "e", "group_id": "g3", "pair_id": "pair", "prompt": "Question"},
            {
                "task_id": "f",
                "group_id": "g4",
                "pair_id": "pair",
                "prompt": "Вопрос",
                "passage_hash": "shared passage",
            },
            {
                "task_id": "g",
                "group_id": "g5",
                "prompt": "Другой вопрос",
                "passage_hash": "shared passage",
            },
            {"task_id": "h", "group_id": "g6", "prompt": "Unrelated"},
        ]
        reverse = [dict(r) for r in reversed(tasks)]
        merge_groups(tasks)
        merge_groups(reverse)
        expected = dict.fromkeys("abcdefg", "g1") | {"h": "g6"}
        self.assertEqual({r["task_id"]: r["group_id"] for r in tasks}, expected)
        self.assertEqual({r["task_id"]: r["group_id"] for r in reverse}, expected)
        self.assertEqual(normalize(" Ａ\nB "), "a b")


if __name__ == "__main__":
    unittest.main()
