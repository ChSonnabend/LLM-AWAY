import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile
from llm_away.helper_deep import investigate_deep,load,extract_private
from llm_away.helper_extract import extract
from llm_away.helper_system import access_checker


def complete(messages,timeout):
    return {'choices':[{'message':{'content':'The supplied files describe detector research. Source paths are in the evidence.'}}],
            'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}

class DeepTests(unittest.TestCase):
    def test_recursive_resume_and_no_reprocessing_unchanged_files(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            root=Path(tmp);nested=root/'one'/'two'/'three';nested.mkdir(parents=True)
            (nested/'chapter.tex').write_text('Research findings '*3000)
            (root/'README.md').write_text('Project overview')
            (root/'video.mp4').write_bytes(b'\0')
            (root/'token').mkdir();(root/'token'/'secret.txt').write_text('DO NOT READ')
            check=access_checker([tmp]);options=dict(max_model_calls=1,timeout_seconds=30)
            with patch('llm_away.helper_deep.extract_private',side_effect=lambda p,d,t,o:extract(p,o)) as parser:
                result=investigate_deep(tmp,'Overview',check,complete,cache,**options)
                identifier=result['investigation_id']
                for _ in range(15):
                    if result['status']=='complete':break
                    result=investigate_deep('','Overview',check,complete,cache,investigation_id=identifier,**options)
                self.assertEqual(result['status'],'complete')
                self.assertEqual(result['coverage']['processed'],2)
                self.assertEqual(result['coverage']['skipped'],1)
                self.assertEqual(result['coverage']['excluded_directories'],1)
                self.assertEqual(parser.call_count,2)
                result=investigate_deep('','Overview',check,complete,cache,investigation_id=identifier,**options)
                self.assertEqual(result['usage']['calls_this_request'],0)
                (root/'README.md').write_text('Changed project overview')
                result=investigate_deep('','Overview',check,complete,cache,investigation_id=identifier,max_model_calls=10)
                self.assertEqual(parser.call_count,3)
                self.assertEqual(result['status'],'complete')

    def test_refresh_discovers_added_files_and_missing_dependencies_reported(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            root=Path(tmp);(root/'initial.txt').write_text('First')
            check=access_checker([tmp])
            with patch('llm_away.helper_deep.extract_private',side_effect=lambda p,d,t,o:extract(p,o)):
                result=investigate_deep(tmp,'Overview',check,complete,cache)
                (root/'added.tex').write_text('Added')
                result=investigate_deep('','Overview',check,complete,cache,investigation_id=result['investigation_id'],refresh=True)
                self.assertEqual(result['coverage']['processed'],2)
            (root/'scanned.pdf').write_bytes(b'%PDF')
            with patch('llm_away.helper_deep.extract_private',side_effect=ValueError('Missing OCR dependency')):
                result=investigate_deep('','Overview',check,complete,cache,investigation_id=result['investigation_id'],refresh=True)
                self.assertEqual(result['coverage']['failed'],1)
                self.assertFalse(result['coverage']['all_discovered_files_fully_processed'])

    def test_interrupted_model_and_revocation(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            (Path(tmp)/'README.md').write_text('Private research')
            check=access_checker([tmp])
            with patch('llm_away.helper_deep.extract_private',side_effect=lambda p,d,t,o:extract(p,o)):
                def fail(*args):raise OSError('interrupted')
                result=investigate_deep(tmp,'Overview',check,fail,cache)
                self.assertEqual(result['coverage']['pending'],1)
                resumed=investigate_deep('','Overview',check,complete,cache,investigation_id=result['investigation_id'])
                self.assertEqual(resumed['status'],'complete')
                with self.assertRaises(ValueError):load(cache,result['investigation_id'],access_checker([]))

    def test_inventory_bound_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            for i in range(3):(Path(tmp)/f'{i}.txt').write_text('Document')
            check=access_checker([tmp])
            with patch('llm_away.helper_deep.extract_private',side_effect=lambda p,d,t,o:extract(p,o)):
                result=investigate_deep(tmp,'Overview',check,complete,cache,max_entries=1)
                self.assertEqual(result['status'],'in_progress')
                result=investigate_deep('','Overview',check,complete,cache,max_entries=10,investigation_id=result['investigation_id'])
                self.assertEqual(result['coverage']['processed'],3)

    def test_office_and_notebook_extractors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for suffix,member,xml in [('.docx','word/document.xml','<document><p>Hello document</p></document>'),
                                     ('.pptx','ppt/slides/slide1.xml','<slide><p>Slide research</p></slide>'),
                                     ('.odt','content.xml','<document><p>Open document</p></document>')]:
                path=root/('test'+suffix)
                with zipfile.ZipFile(path,'w') as archive:archive.writestr(member,xml)
                self.assertIn('document' if suffix!='.pptx' else 'research',extract(path)['text'])
            book=root/'test.ipynb';book.write_text(json.dumps({'cells':[{'cell_type':'markdown','source':['Research notebook']}]}))
            self.assertIn('Research notebook',extract(book)['text'])
            file=root/'test.tex';file.write_text('abcdef')
            result=extract(file,max_chars=3)
            self.assertTrue(result['partial']);self.assertEqual(result['text'],'abc')

    def test_actual_extraction_worker_and_reduction_tree(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cache:
            path=Path(tmp)/'space ; file.tex';path.write_text('Research chapter from isolated worker')
            result=extract_private(path,cache,10,False)
            self.assertIn('isolated worker',result['text'])
            for i in range(10):(Path(tmp)/f'{i}.md').write_text(f'Project {i}')
            check=access_checker([tmp])
            with patch('llm_away.helper_deep.extract_private',side_effect=lambda p,d,t,o:extract(p,o)):
                result=investigate_deep(tmp,'Overview',check,complete,cache,max_model_calls=30)
                self.assertEqual(result['coverage']['processed'],11)
                self.assertEqual(result['status'],'complete')
                self.assertEqual(result['usage']['calls_this_request'],13)

    def test_xlsx_shared_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.xlsx'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('xl/sharedStrings.xml','<sst><si><t>Research cell</t></si></sst>')
                archive.writestr('xl/worksheets/sheet1.xml','<worksheet><row><c r="A1" t="s"><v>0</v></c></row></worksheet>')
            self.assertIn('A1: Research cell',extract(path)['text'])

    def test_pdf_extraction_with_poppler(self):
        import shutil
        if not shutil.which('pdftotext'):self.skipTest('Poppler is optional')
        # Small self-contained fixture; no external document or user folder is read.
        objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
                 b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
                 b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
        stream=b'BT /F1 12 Tf 30 250 Td (Research PDF fixture) Tj ET'
        objects.append(b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream')
        data=b'%PDF-1.4\n';offsets=[0]
        for i,obj in enumerate(objects,1):
            offsets.append(len(data));data+=str(i).encode()+b' 0 obj\n'+obj+b'\nendobj\n'
        start=len(data)
        data+=b'xref\n0 6\n0000000000 65535 f \n'+b''.join(f'{n:010d} 00000 n \n'.encode() for n in offsets[1:])
        data+=b'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n'+str(start).encode()+b'\n%%EOF'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.pdf';path.write_bytes(data)
            result=extract_private(path,tmp,10,False)
            self.assertIn('Research PDF fixture',result['text'])
            self.assertFalse(result['partial'])
