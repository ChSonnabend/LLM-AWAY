import tempfile
import unittest
from pathlib import Path

from unittest.mock import Mock, patch

from llm_away.rag import Index, FolderIndexes, roots_for
from llm_away.rag import index_directory, reusable_roots, candidates
import sqlite3
import json


class RagPathTests(unittest.TestCase):
    def test_parent_refresh_embeds_only_uncovered_files(self):
        try:import fastembed
        except ImportError:self.skipTest('requires RAG runtime')
        with tempfile.TemporaryDirectory() as temporary, patch('llm_away.rag.CACHE',Path(temporary)/'cache'):
            root=Path(temporary)/'project';child=root/'child';child.mkdir(parents=True)
            (child/'code.py').write_text('print("child")')
            (root/'extra.py').write_text('print("parent")')
            embedder=Mock()
            embedder.passage_embed.side_effect=lambda passages,**kw:iter([[1.,0.] for _ in passages])
            index=Index([child],embedder=embedder)
            index.refresh();index.db.close();embedder.reset_mock()
            with patch('fastembed.TextEmbedding',return_value=embedder):
                folders=FolderIndexes([root]);folders.refresh()
            embedder.passage_embed.assert_called_once()
            self.assertTrue(embedder.passage_embed.call_args.args[0][0].startswith('extra.py'))
            remainder=folders.indexes[root]
            self.assertEqual(remainder.db.execute('SELECT path FROM files').fetchall(),[(str(root/'extra.py'),)])
            for index in folders.indexes.values():index.db.close()

    def test_discovers_legacy_cache_and_partitions_parent(self):
        with tempfile.TemporaryDirectory() as temporary, patch('llm_away.rag.CACHE',Path(temporary)/'cache'):
            root=Path(temporary)/'project';child=root/'child';child.mkdir(parents=True)
            directory=index_directory([child]);directory.mkdir(parents=True)
            with sqlite3.connect(directory/'index.sqlite') as db:
                db.execute('CREATE TABLE files(path TEXT PRIMARY KEY,digest TEXT)')
                db.execute('INSERT INTO files VALUES(?,?)',(str(child/'file.py'),'digest'))
            self.assertEqual(reusable_roots(),{child})
            folders=FolderIndexes([root])
            self.assertEqual(folders.roots,[child,root])
            self.assertEqual(folders.excluded[root],(child,))
            self.assertNotEqual(index_directory([root]),index_directory([root],[child]))
            self.assertEqual(FolderIndexes([child]).roots,[child])
            with sqlite3.connect(directory/'index.sqlite') as db:
                db.execute('CREATE TABLE rag_metadata(key TEXT PRIMARY KEY,value TEXT)')
                db.execute('INSERT INTO rag_metadata VALUES(?,?)',('manifest',json.dumps(
                    {'roots':[str(child)],'model':'incompatible','format':1,'excluded':[]})))
            self.assertEqual(reusable_roots(),set())

    def test_prefers_existing_parent_and_nonoverlapping_children(self):
        root=Path('/project');child=root/'child';nested=child/'nested'
        with patch('llm_away.rag.reusable_roots',return_value={child,nested}):
            self.assertEqual(FolderIndexes([root]).roots,[child,root])
        with patch('llm_away.rag.reusable_roots',return_value={root,child,nested}):
            self.assertEqual(FolderIndexes([root]).roots,[root])

    def test_remainder_does_not_scan_reused_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);child=root/'child';child.mkdir()
            (child/'code.py').write_text('reused')
            extra=root/'extra.py';extra.write_text('new')
            try:import pathspec
            except ImportError:self.skipTest('pathspec requires RAG runtime')
            self.assertEqual(list(candidates(root,[child])),[extra])

    def test_builder_stops_after_owner_exits(self):
        import os,subprocess,sys,signal
        code='from llm_away.rag import watch_owner; import time; watch_owner(-1); time.sleep(20)'
        child=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,env=dict(os.environ))
        try:self.assertEqual(child.wait(timeout=5),-signal.SIGTERM)
        finally:
            if child.poll() is None:child.kill();child.wait()

    def test_federation_merges_folders_and_embeds_query_once(self):
        folders=FolderIndexes([Path('/a'),Path('/b')])
        folders.embedder=Mock()
        folders.embedder.query_embed.return_value=iter([[1,0]])
        first=Mock();second=Mock()
        first.search.return_value=[(.02,'/a/file',1,2,'first folder')]
        second.search.return_value=[(.03,'/b/file',1,2,'second folder')]
        with patch.object(folders,'get',side_effect=[first,second]):
            result=folders.search('question',2)
        self.assertLess(result.index('second folder'),result.index('first folder'))
        folders.embedder.query_embed.assert_called_once_with('question')
        self.assertEqual(first.search.call_args.kwargs['query_vector'],[1,0])

    def test_busy_folder_does_not_hide_ready_folder(self):
        folders=FolderIndexes([Path('/a'),Path('/b')]);folders.embedder=Mock()
        folders.embedder.query_embed.return_value=iter([[1]])
        ready=Mock();ready.search.return_value=[(.02,'/b/file',1,2,'available excerpt')]
        with patch.object(folders,'get',side_effect=[ValueError('RAG index is still building: /a'),ready]):
            result=folders.search('question')
        self.assertIn('still building: /a',result)
        self.assertIn('available excerpt',result)

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

    def test_search_logs_the_exact_query_before_retrieval(self):
        index=Index.__new__(Index);index.report=Mock()
        index.refresh=Mock(side_effect=RuntimeError('stop after log'))
        with self.assertRaisesRegex(RuntimeError,'stop after log'):
            index.search('first line\nsecond line')
        index.report.assert_called_once_with('QUERY | message "first line\\nsecond line"')


if __name__=='__main__':unittest.main()
