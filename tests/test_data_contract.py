"""Guard consequential ingestion and leakage assumptions of the pinned corpus."""

import unittest
from collections import defaultdict

from scripts.eda import load_tasks, merge_groups, normalize, partition


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks()

    def test_snapshot_schema_and_reference_coverage(self):
        self.assertEqual(len(self.tasks), 2464)
        self.assertEqual(len({r["task_id"] for r in self.tasks}), 2464)
        self.assertTrue(all(r["prompt"].strip() for r in self.tasks))
        self.assertTrue(all(r["valid_target"] and r["reference_present"] for r in self.tasks))
        self.assertEqual(sum(r.get("numeric_separator", False) for r in self.tasks), 8)
        self.assertEqual(sum(r.get("duplicate_options", False) for r in self.tasks), 4)

    def test_translation_pairs_and_gold_alignment(self):
        pairs = defaultdict(list)
        for row in self.tasks:
            if row["source"] != "HumanEval":
                pairs[row["pair_id"]].append(row)
        self.assertEqual(len(pairs), 1150)
        for pair in pairs.values():
            self.assertEqual({r["language"] for r in pair}, {"en", "ru"})
            self.assertEqual(len({r["target"] for r in pair}), 1)

    def test_related_prompts_never_cross_grouped_partitions(self):
        tasks = [dict(row) for row in self.tasks]
        merge_groups(tasks)
        groups = defaultdict(set)
        for row in tasks:
            groups[row["group_id"]].add(partition(row["group_id"]))
        self.assertEqual(len(groups), 902)
        self.assertTrue(all(len(parts) == 1 for parts in groups.values()))

    def test_duplicate_union_is_transitive_and_order_independent(self):
        tasks = [
            {"task_id": "a", "group_id": "g1", "prompt": "Same  prompt"},
            {"task_id": "b", "group_id": "g2", "prompt": "SAME prompt"},
            {"task_id": "c", "group_id": "g2", "prompt": "Different"},
            {"task_id": "d", "group_id": "g3", "prompt": "Different"},
        ]
        reverse = [dict(r) for r in reversed(tasks)]
        merge_groups(tasks)
        merge_groups(reverse)
        self.assertEqual({r["group_id"] for r in tasks}, {"g1"})
        self.assertEqual({r["group_id"] for r in reverse}, {"g1"})
        self.assertEqual(normalize(" Ａ\nB "), "a b")


if __name__ == "__main__":
    unittest.main()
