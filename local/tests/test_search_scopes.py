import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llm_away.rag import Index, FolderIndexes, search_paths, path_matches
from llm_away.resources import timestamp_log_lines


class ScopeTests(unittest.TestCase):
    def test_paths(self):
        roots=[Path('/project')]
        for query in ['find in /project/src', 'find in src/', 'find in project/src']:
            self.assertEqual(search_paths(query,roots),(Path('/project/src'),))
        scopes=search_paths('find in `/project/my src/`',roots)
        self.assertTrue(path_matches('/project/my src/code.py',scopes))
        self.assertFalse(path_matches('/project/my src-extra/code.py',scopes))
        self.assertFalse(path_matches('/project/src/code.py',search_paths('/elsewhere/',roots)))
        self.assertFalse(path_matches('/project/src/other.py',search_paths('/project/src/one.py',roots)))

    def test_parent_scope_searches_reused_children(self):
        with patch('llm_away.rag.reusable_roots',return_value={Path('/project/src')}):
            folders=FolderIndexes([Path('/project')])
        folders.embedder=Mock();folders.embedder.query_embed.return_value=iter([[1]])
        index=Mock();index.search.return_value=[]
        with patch.object(folders,'get',return_value=index) as get:
            folders.search('find bug in /project')
        self.assertEqual([call.args[0] for call in get.call_args_list],folders.roots)

    def test_scope_filters_before_top_candidates(self):
        try:import numpy as np
        except ImportError:self.skipTest("requires RAG runtime")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'src';target.mkdir()
            wanted=target/'code.py';wanted.write_text('needle')
            index=object.__new__(Index);index.np=np;index.report=Mock();index.refresh=Mock()
            index.db=sqlite3.connect(':memory:')
            self.addCleanup(index.db.close)
            index.db.executescript('CREATE TABLE chunks(id INTEGER PRIMARY KEY,path,start,end,text,vector); CREATE TABLE files(path,digest); CREATE VIRTUAL TABLE lexical USING fts5(text,path);')
            for n in range(50):
                path=str(root/f'other{n}.py') if n<49 else str(wanted)
                vector=np.asarray([1.,0.] if n<49 else [0.,1.],dtype='float32').tobytes()
                index.db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?)',(n+1,path,1,1,'needle',vector))
                index.db.execute('INSERT INTO lexical(rowid,text,path) VALUES(?,?,?)',(n+1,'needle',path))
            index.db.execute('INSERT INTO files VALUES(?,?)',(str(wanted),hashlib.sha256(wanted.read_bytes()).hexdigest()))
            result=index.search('needle',ranked=True,query_vector=[1.,0.],search_scopes=(target,))
            self.assertEqual([row[1] for row in result],[str(wanted)])

    def test_timestamp_multiline_and_partial_chunks(self):
        with patch('llm_away.resources.time.strftime',return_value='TIME | '):
            output,pending=timestamp_log_lines('first\nsec')
            self.assertEqual(output,'TIME | first\n')
            output,pending=timestamp_log_lines(pending+'ond\nlast')
            self.assertEqual(output,'TIME | second\n')
            self.assertEqual(timestamp_log_lines(pending,final=True),('TIME | last\n',''))
