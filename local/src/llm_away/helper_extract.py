"""Read-only document-to-text worker; invoked in an isolated, resource-limited process."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile
from xml.etree import ElementTree as ET
from .document_tools import find_tool

TEXT={'.tex','.bib','.sty','.cls','.md','.rst','.txt','.csv','.tsv','.log','.py','.sh','.toml','.json','.yaml','.yml','.ini','.cfg','.c','.cpp','.cxx','.h','.js','.ts','.html','.css','.xml','.sql','.r'}
OFFICE={'.docx','.pptx','.xlsx','.odt','.odp','.ods'}
IMAGES={'.png','.jpg','.jpeg','.tif','.tiff'}
SUPPORTED=TEXT|OFFICE|{'.ipynb','.pdf'}


def run(command, timeout=25):
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if result.returncode:raise ValueError('Extractor failed: '+result.stderr.decode(errors='replace')[-300:])
    return result.stdout.decode('utf-8',errors='replace')


def office(path):
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist())>128*1024*1024:raise ValueError('Office archive exceeds 128 MiB expanded limit')
        names=archive.namelist();suffix=path.suffix.lower()
        if suffix=='.docx':names=[n for n in names if re.match(r'word/(document|footnotes|endnotes|header\d+|footer\d+)\.xml$',n)]
        elif suffix=='.pptx':names=[n for n in names if re.match(r'ppt/(slides/slide\d+|notesSlides/notesSlide\d+)\.xml$',n)]
        elif suffix=='.xlsx':names=[n for n in names if n=='xl/sharedStrings.xml' or re.match(r'xl/worksheets/sheet\d+\.xml$',n)]
        else:names=[n for n in names if n=='content.xml']
        names.sort(key=lambda n:[int(s) if s.isdigit() else s for s in re.split(r'(\d+)',n)])
        texts=[];strings=[]
        if suffix=='.xlsx' and 'xl/sharedStrings.xml' in names:
            root=ET.fromstring(archive.read('xl/sharedStrings.xml'))
            strings=[''.join(n.itertext()) for n in root]
        for name in names:
            if name=='xl/sharedStrings.xml':continue
            root=ET.fromstring(archive.read(name));parts=[]
            if suffix=='.xlsx':
                for cell in root.iter():
                    if cell.tag.split('}')[-1]!='c':continue
                    value=''.join(n.text or '' for n in cell.iter() if n.tag.split('}')[-1] in ('v','t'))
                    if cell.attrib.get('t')=='s':
                        try:value=strings[int(value)]
                        except (ValueError,IndexError):pass
                    parts.append(cell.attrib.get('r','')+': '+value)
            else:
                for node in root.iter():
                    if node.tag.split('}')[-1] in ('p','h'):
                        text=''.join(node.itertext()).strip()
                        if text:parts.append(text)
            texts.append('['+name+']\n'+'\n'.join(parts))
        return '\n\n'.join(texts)


def extract(path,enable_ocr=False,max_chars=2000000):
    path=Path(path);suffix=path.suffix.lower();notes=[];partial=False
    if path.stat().st_size>64*1024*1024:raise ValueError('Source exceeds 64 MiB extraction limit')
    if suffix in TEXT or path.name in ('Makefile','Dockerfile'):
        raw=path.read_bytes()
        if b'\0' in raw:raise ValueError('Binary content in text file')
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text=raw.decode('utf-8',errors='replace');partial=True;notes.append('Invalid UTF-8 replaced')
    elif suffix=='.ipynb':
        book=json.loads(path.read_text())
        text='\n\n'.join(f'[cell {i+1}: {cell.get("cell_type")}]\n'+(''.join(cell.get('source',[]))) for i,cell in enumerate(book.get('cells',[])))
        notes.append('Notebook source and markdown only; outputs/attachments excluded')
    elif suffix in OFFICE:
        text=office(path);notes.append('Document text only; charts, images, formatting and embedded objects excluded')
    elif suffix=='.pdf':
        if not find_tool('pdftotext'):raise ValueError('Missing dependency: pdftotext (Poppler)')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'text.txt';run([find_tool('pdftotext'),'-layout',str(path),str(out)])
            text=out.read_text(errors='replace');pages=text.split('\f')
            if pages and not pages[-1].strip():pages.pop()
            empty=[i for i,page in enumerate(pages) if not page.strip()]
            if not text.strip() and not pages:empty=[0];pages=['']
            if empty and enable_ocr:
                if not find_tool('pdftoppm') or not find_tool('tesseract'):
                    notes.append('OCR required but pdftoppm/tesseract unavailable');partial=True
                else:
                    for i in empty[:20]:
                        prefix=str(Path(tmp)/f'page-{i+1}')
                        run([find_tool('pdftoppm'),'-f',str(i+1),'-l',str(i+1),'-scale-to','1800','-png','-singlefile',str(path),prefix])
                        pages[i]=run([find_tool('tesseract'),prefix+'.png','stdout'])
                    if len(empty)>20:notes.append('OCR limited to 20 image-only pages');partial=True
            elif empty:
                notes.append('Image-only pages require enable_ocr=true');partial=True
            if any(not p.strip() for p in pages):partial=True
            text='\n\n'.join(f'[page {i+1}]\n{page}' for i,page in enumerate(pages))
            if not any(p.strip() for p in pages):raise ValueError('; '.join(notes) or 'No extractable PDF text')
            notes.append('PDF text extraction; figures and layout not interpreted; OCR may be inaccurate')
    elif suffix in IMAGES and enable_ocr:
        if not find_tool('tesseract'):raise ValueError('Missing dependency: tesseract for OCR')
        text=run([find_tool('tesseract'),str(path),'stdout']);notes.append('OCR text only, not image interpretation')
        if not text.strip():raise ValueError('OCR produced no text')
    else:raise ValueError('Unsupported format')
    if len(text)>max_chars:partial=True;notes.append(f'Extracted text capped at {max_chars} characters')
    return {'text':text[:max_chars],'partial':partial,'notes':notes}


def configure_limits():
    # RLIMIT_AS is not a reliable address-space bound on macOS. CPU, output size,
    # and the parent's elapsed-time/process-group limits still apply there.
    try:import resource
    except ImportError:return
    limits=[(resource.RLIMIT_CPU,40),(resource.RLIMIT_FSIZE,32*1024*1024)]
    if sys.platform.startswith('linux'):limits.append((resource.RLIMIT_AS,768*1024*1024))
    for kind,value in limits:
        try:
            _,hard=resource.getrlimit(kind)
            bound=value if hard==resource.RLIM_INFINITY else min(value,hard)
            resource.setrlimit(kind,(bound,bound))
        except (ValueError,OSError):continue


def main():
    configure_limits()
    try:result=extract(sys.argv[1],sys.argv[3]=='1')
    except Exception as exc:result={'error':str(exc)[:500]}
    Path(sys.argv[2]).write_text(json.dumps(result))

if __name__=='__main__':main()
