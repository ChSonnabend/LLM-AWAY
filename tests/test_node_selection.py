from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock
import re
import time
import sys

SOURCE = (Path(__file__).resolve().parents[1] / 'scripts/remote/llm-epn-serverctl').read_text()
FUNCTION = SOURCE[SOURCE.index('def choose_node(node_class):'):SOURCE.index('\ndef submit():')]

class NodeSelectionTests(unittest.TestCase):
    def choose(self, listing, code=0, fallback=False):
        capture = Mock(return_value=subprocess.CompletedProcess([], code, listing, 'scheduler unavailable'))
        namespace = dict(payload={'partition': 'prod', 'mi50_fallback': fallback}, llama={}, run_capture=capture, re=re, time=time, sys=sys)
        exec(FUNCTION, namespace)
        return namespace['choose_node']('mi100')

    def test_busy_nodes_are_used_when_none_idle(self):
        self.assertIn(self.choose('epn282 alloc\nepn349 mix\nepn280 drain*\nepn000 idle\n'), {'epn282', 'epn349'})

    def test_idle_preferred(self):
        self.assertEqual(self.choose('epn282 alloc\nepn303 idle\n'), 'epn303')

    def test_missing_class_fails_without_inventing_host(self):
        with self.assertRaisesRegex(SystemExit, 'No usable mi100'):
            self.choose('epn000 idle\nepn280 drain*\nepn331 down*\n')

    def test_discovery_failure_does_not_submit(self):
        with self.assertRaisesRegex(SystemExit, 'Cannot discover Slurm nodes'):
            self.choose('', 1)

    def test_fallback_prefers_idle_mi100(self):
        self.assertEqual(self.choose('epn282 idle\nepn001 idle\n', fallback=True), 'epn282')

    def test_fallback_uses_idle_mi50(self):
        self.assertEqual(self.choose('epn282 alloc\nepn001 idle\n', fallback=True), 'epn001')

    def test_both_busy_queue_on_mi100(self):
        self.assertEqual(self.choose('epn282 alloc\nepn001 alloc\n', fallback=True), 'epn282')
