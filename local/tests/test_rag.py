import tempfile
import unittest
from pathlib import Path

from llm_away.rag import roots_for


class RagPathTests(unittest.TestCase):
    def test_accepts_files_and_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'src';folder.mkdir()
            source=Path(temporary)/'README.md';source.write_text('project')
            self.assertEqual(roots_for([str(folder),str(source)]),[source.resolve(),folder.resolve()])

    def test_parent_directory_deduplicates_selected_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);source=folder/'README.md';source.write_text('project')
            self.assertEqual(roots_for([str(folder),str(source)]),[folder.resolve()])

    def test_rejects_missing_path(self):
        with self.assertRaisesRegex(ValueError,'files or directories'):
            roots_for(['/definitely/not/a/real/rag/path'])


if __name__=='__main__':unittest.main()
