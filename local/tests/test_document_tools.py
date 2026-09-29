import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_away import document_tools as tools
from llm_away.helper_extract import configure_limits

class DocumentToolsTests(unittest.TestCase):
    def test_mac_homebrew_discovery_without_shell_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'tesseract';binary.write_text('#!/bin/sh\n');binary.chmod(0o755)
            with patch.object(tools.platform,'system',return_value='Darwin'),patch.object(tools.shutil,'which',return_value=None),patch.object(tools,'MAC_BINS',(tmp,)):
                self.assertEqual(tools.find_tool('tesseract'),str(binary))
                self.assertIsNone(tools.find_tool('pdftotext'))

    def test_install_commands(self):
        with patch.object(tools,'find_tool',side_effect=lambda name:'/bin/'+name):
            self.assertEqual(tools.install_command('Darwin'),['/bin/brew','install','poppler','tesseract'])
            with patch.object(tools.os,'geteuid',return_value=1000):
                self.assertEqual(tools.install_command('Linux'),['/bin/sudo','/bin/apt-get','install','poppler-utils','tesseract-ocr','tesseract-ocr-eng'])
        with patch.object(tools,'find_tool',return_value=None):
            with self.assertRaisesRegex(ValueError,'brew'):tools.install_command('Darwin')

    def test_ocr_language_check(self):
        import subprocess
        with patch.object(tools,'find_tool',side_effect=lambda name:'/bin/'+name),patch.object(tools.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'List of available languages (2):\neng\nosd\n','')):
            self.assertTrue(tools.status()['available'])
        with patch.object(tools,'find_tool',side_effect=lambda name:'/bin/'+name),patch.object(tools.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'List of available languages (1):\nosd\n','')):
            self.assertIn('Tesseract English language data (eng)',tools.status()['missing'])

    def test_mac_limits_skip_address_space_and_tolerate_unsupported_limit(self):
        import resource
        with patch('llm_away.helper_extract.sys.platform','darwin'),patch.object(resource,'getrlimit',return_value=(-1,-1)),patch.object(resource,'setrlimit',side_effect=[OSError('unsupported'),None]) as setter:
            configure_limits()
            self.assertEqual([call.args[0] for call in setter.call_args_list],[resource.RLIMIT_CPU,resource.RLIMIT_FSIZE])
