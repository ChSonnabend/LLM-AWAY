"""Recursive document ingestion and explicit question -> answer-file supervision."""
import csv
import json
from pathlib import Path

MANIFEST='QandA_finetuning.txt'
TEXT={'.txt','.csv','.tsv','.md','.rst','.json','.jsonl','.yaml','.yml','.xml','.html','.htm','.py','.js','.ts','.c','.cpp','.h','.tex','.log','.sh','.toml'}

def inside(path, roots):
    return any(path==root or (root.is_dir() and root in path.parents) for root in roots)

def discover(paths, exclude=None):
    roots=[Path(p).expanduser().resolve() for p in paths]
    if not roots:raise ValueError('Provide at least one source file or folder')
    found=set();excluded=Path(exclude).resolve() if exclude else None
    for root in roots:
        if not root.exists():raise ValueError('Source does not exist: '+str(root))
        for path in ([root] if root.is_file() else root.rglob('*')):
            if not path.is_file():continue
            real=path.resolve()
            if excluded and (real==excluded or excluded in real.parents):continue
            if inside(real,roots):found.add(real)
    return roots,sorted(found)

def labels(manifests,roots):
    pairs=[]
    for manifest in manifests:
        questions=[]
        for line in manifest.read_text(encoding='utf-8-sig').splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            if line.startswith('[') and line.endswith(']'):
                target=Path(line[1:-1].strip()).expanduser()
                target=(manifest.parent/target).resolve() if not target.is_absolute() else target.resolve()
                if not questions:raise ValueError(str(manifest)+': answer path has no preceding questions')
                if not target.is_file() or not inside(target,roots):
                    raise ValueError('Answer file must exist within the supplied sources: '+str(target))
                if target.name==MANIFEST:raise ValueError('A Q&A manifest cannot be its own training answer')
                pairs.extend((q,target) for q in questions);questions=[]
            else:
                questions.extend(q.strip() for q in next(csv.reader([line])) if q.strip())
        if questions:raise ValueError(str(manifest)+': questions have no answer-file path')
    return pairs

def extract(path):
    if path.stat().st_size>128*1024*1024:raise ValueError('File exceeds 128 MiB extraction limit; split it into smaller files')
    if path.suffix.lower() in TEXT or not path.suffix:
        raw=path.read_bytes()
        if b'\x00' in raw[:8192]:raise ValueError('Binary data in text file')
        return raw.decode('utf-8-sig')
    if path.suffix.lower()=='.pdf':
        from pypdf import PdfReader
        return '\n\n'.join(page.extract_text() or '' for page in PdfReader(str(path)).pages)
    from markitdown import MarkItDown
    return MarkItDown().convert(str(path)).text_content

def prepare(paths, output, progress=lambda **kw:None, stopped=lambda:False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    roots,files=discover(paths,output)
    manifests=[p for p in files if p.name==MANIFEST]
    pairs=labels(manifests,roots);by_file={}
    for question,path in pairs:by_file.setdefault(path,[]).append(question)
    documents=sorted(set(files)-set(manifests)|set(by_file))
    skipped=[];examples=0;accepted=0
    target=output/'dataset.jsonl'
    with target.open('w',encoding='utf-8') as stream:
        for i,path in enumerate(documents):
            if stopped():raise InterruptedError('Stopped during dataset preparation')
            try:
                text=extract(path).strip()
                if not text:raise ValueError('No extractable text; scanned media needs OCR/transcription first')
            except Exception as exc:
                if path in by_file:raise ValueError('Could not read labeled answer '+str(path)+': '+str(exc)) from exc
                skipped.append({'path':str(path),'reason':str(exc)});text=''
            if text:
                accepted+=1
                for question in by_file.get(path,['']):
                    stream.write(json.dumps({'file_id':str(path),'question':question,'answer':text},ensure_ascii=False)+'\n');examples+=1
            progress(phase='PREPARING',files_total=len(documents),files_prepared=i+1,files_accepted=accepted,skipped=len(skipped))
    report={'files_total':len(documents),'files_accepted':accepted,'examples':examples,'skipped':skipped,'manifests':[str(p) for p in manifests]}
    (output/'dataset-report.json').write_text(json.dumps(report,indent=2))
    if not examples:raise ValueError('No trainable text found; see dataset-report.json')
    return target,report
