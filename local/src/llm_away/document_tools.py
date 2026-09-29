"""Discover and explicitly install optional system document/OCR dependencies."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

MANIFEST=Path(__file__).resolve().parents[2]/'requirements'/'documents-system.json'
MAC_BINS=('/opt/homebrew/bin','/usr/local/bin','/opt/local/bin')


def find_tool(name):
    found=shutil.which(name)
    if found:return found
    # Desktop-launched MCP hosts often lack Homebrew in PATH.
    if platform.system()=='Darwin':
        for directory in MAC_BINS:
            path=Path(directory)/name
            if path.is_file() and os.access(path,os.X_OK):return str(path)
    return None


def requirements():return json.loads(MANIFEST.read_text())


def status():
    spec=requirements()
    tools={name:find_tool(name) for name in spec['executables']}
    missing=[name for name,value in tools.items() if not value]
    languages=[];error=None
    if tools['tesseract']:
        try:
            result=subprocess.run([tools['tesseract'],'--list-langs'],capture_output=True,text=True,timeout=15)
            if result.returncode:error='Tesseract could not list installed languages'
            else:languages=[line.strip() for line in result.stdout.splitlines()[1:] if line.strip()]
        except (OSError,subprocess.SubprocessError) as exc:error=str(exc)
        if error or spec['ocr_language'] not in languages:missing.append('Tesseract English language data (eng)')
    return {'available':not missing,'executables':tools,'ocr_languages':languages,'missing':missing,'error':error}


def install_command(system=None):
    system=system or platform.system();spec=requirements()['platforms'].get(system)
    if not spec:raise ValueError('Automatic setup supports Homebrew macOS and apt-based Linux; install Poppler and Tesseract manually on this platform')
    manager=find_tool(spec['manager'])
    if not manager:raise ValueError(f"Install {spec['manager']} first, or install Poppler and Tesseract manually")
    command=[manager,'install',*spec['packages']]
    if system=='Linux' and os.geteuid()!=0:
        sudo=find_tool('sudo')
        if not sudo:raise ValueError('Install system packages as an administrator; sudo is unavailable')
        command.insert(0,sudo)
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true',help='Install the declared system packages (may require administrator authentication)')
    parser.add_argument('--check',action='store_true',help='Report availability without installing anything (default)')
    args=parser.parse_args()
    if args.check and args.install:parser.error('Choose --check or --install')
    current=status()
    if args.install and not current['available']:
        try:command=install_command()
        except ValueError as exc:parser.exit(1,str(exc)+'\n')
        print('Installing document requirements: '+' '.join(command),flush=True)
        subprocess.run(command,check=True)
        current=status()
    print(json.dumps(current,indent=2))
    if not current['available']:parser.exit(1,'Install with local/bin/setup-documents --install (or local/scripts/init --documents).\n')

if __name__=='__main__':main()
