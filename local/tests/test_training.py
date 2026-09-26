import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

TRAINING=Path(__file__).resolve().parents[2]/'remote/training'
def module(name):
    spec=importlib.util.spec_from_file_location(name,TRAINING/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
dataset=module('dataset');control=module('control')
with patch.dict(sys.modules,{'dataset':dataset}):runner=module('runner')

class DatasetTests(unittest.TestCase):
    def test_nested_labels_and_unlabelled_without_answer_duplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'nested').mkdir();(root/'nested/answer.txt').write_text('The complete answer')
            (root/'corpus.csv').write_text('a,b\n1,2')
            (root/dataset.MANIFEST).write_text('"Question, with comma?", Second question?\n[nested/answer.txt]\n')
            out,report=dataset.prepare([str(root),str(root/'nested')],root/'output')
            rows=[json.loads(s) for s in out.read_text().splitlines()]
            self.assertEqual(report['files_accepted'],2);self.assertEqual(len(rows),3)
            self.assertEqual(sorted(r['question'] for r in rows),['','Question, with comma?','Second question?'])
            self.assertEqual(sum(r['answer']=='The complete answer' for r in rows),2)
    def test_answer_escape_and_missing_answer_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'sources';folder.mkdir();(root/'secret.txt').write_text('outside')
            manifest=folder/dataset.MANIFEST;manifest.write_text('Question\n[../secret.txt]')
            with self.assertRaisesRegex(ValueError,'within the supplied'):dataset.prepare([folder],root/'out')
            manifest.write_text('Question')
            with self.assertRaisesRegex(ValueError,'no answer'):dataset.prepare([folder],root/'out')
    def test_bad_unlabelled_file_reported_but_labelled_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'good.txt').write_text('good');(root/'bad.txt').write_bytes(b'\x00bad')
            _,report=dataset.prepare([root],root/'out')
            self.assertEqual(report['files_accepted'],1);self.assertEqual(len(report['skipped']),1)
            (root/dataset.MANIFEST).write_text('Question\n[bad.txt]')
            with self.assertRaisesRegex(ValueError,'labeled answer'):dataset.prepare([root],root/'out')
    def test_cancel_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'text.txt').write_text('text')
            with self.assertRaises(InterruptedError):dataset.prepare([root],root/'out',stopped=lambda:True)
    def test_chunks_cover_full_answer_with_question_mask_and_eos_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'rows';source.write_text(json.dumps(dict(file_id='file',question='Q',answer='long'))+'\n')
            tokenizer=Mock(eos_token_id=0);tokenizer.apply_chat_template.return_value=[90,91];tokenizer.encode.return_value=list(range(1,68))
            offsets,counts=runner.token_examples(source,tokenizer,32,root/'tokens',lambda:False)
            rows=[json.loads(s) for s in (root/'tokens').read_text().splitlines()]
            self.assertEqual(counts['file'],3);self.assertEqual(len(offsets),3)
            self.assertEqual([t for row in rows for t in row['labels'][2:]],list(range(1,68))+[0])
            self.assertTrue(all(row['labels'][:2]==[-100,-100] for row in rows))

class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.state=self.root/'allocation';self.state.mkdir();self.source=self.root/'input.txt';self.source.write_text('train')
        self.current={'active':True,'generation':'inference'}
        self.payload={'confirmed':True,'expected_generation':'inference','settings':{'python':sys.executable,'model':'base/model','paths':[str(self.source)]}}
        self.probe=patch.object(control.subprocess,'run',return_value=Mock(returncode=0));self.probe.start();self.addCleanup(self.probe.stop)
    def call(self,action,p=None):return control.dispatch(action,p or self.payload,self.state,self.root,self.current,lambda path,s:path.write_text(s))
    def test_preflight_does_not_touch_allocation(self):
        self.payload['operation']='start';self.call('training-check');self.assertEqual(list(self.state.iterdir()),[])
    def test_start_stop_and_concurrent_start_fence(self):
        status=self.call('training-start');run=status['run_id']
        self.assertEqual((self.state/'desired').read_text(),run)
        self.assertTrue((self.state/(run+'.training.json')).exists())
        with self.assertRaisesRegex(ValueError,'already occupies'):self.call('training-start')
        with self.assertRaisesRegex(ValueError,'run changed'):self.call('training-stop',{'run_id':'other'})
        self.call('training-stop',{'run_id':run});self.assertTrue((Path(status['output'])/'stop').exists())
    def test_bad_runtime_or_generation_never_unloads(self):
        (self.state/'desired').write_text('inference');self.payload['settings']['python']='/no/such/python'
        with self.assertRaisesRegex(ValueError,'environment missing'):self.call('training-start')
        self.payload['expected_generation']='stale'
        with self.assertRaisesRegex(ValueError,'Model changed'):self.call('training-start')
        self.assertEqual((self.state/'desired').read_text(),'inference');self.assertFalse((self.state/'training-status.json').exists())
    def test_teacher_secret_not_in_request_or_status(self):
        self.payload['settings'].update(teacher={'base_url':'http://teacher'},teacher_key='secret-token')
        status=self.call('training-start');output=Path(status['output'])
        self.assertNotIn('secret-token',(output/'request.json').read_text());self.assertNotIn('secret-token',json.dumps(status))
        self.assertEqual((output/'teacher-api-key').stat().st_mode & 0o777,0o600)
    def test_invalid_export_does_not_unload(self):
        self.payload['settings']['adapter']=str(self.root)
        with self.assertRaisesRegex(ValueError,'adapter_config'):self.call('training-export')
        self.assertFalse((self.state/'desired').exists())

class ContainerTests(ControlTests):
    def test_container_preflight_and_worker_use_same_mounts_and_interpreter(self):
        import shlex
        image=self.root/'unsloth image.sif';image.touch()
        self.payload['settings'].update(runtime='apptainer',container=str(image),python='/opt/unsloth/bin/python')
        with patch.object(control.shutil,'which',return_value='/usr/bin/apptainer'):
            status=self.call('training-start')
        probe=control.subprocess.run.call_args.args[0]
        self.assertEqual(probe[:2],['/usr/bin/apptainer','exec'])
        self.assertNotIn('--nv',probe)
        self.assertIn(str(image.resolve()),probe);self.assertIn('/opt/unsloth/bin/python',probe)
        script=(self.state/(status['run_id']+'.sh')).read_text()
        self.assertIn('APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"',script)
        command=shlex.split(script.split('exec ',1)[1].split(' >> ',1)[0])
        self.assertIn('--nv',command);self.assertIn(str(image.resolve()),command)
        self.assertIn(str(self.state)+':'+str(self.state),command)
        self.assertIn(str(self.root)+':'+str(self.root),command)
        self.assertFalse((self.state/'venvs').exists())
    def test_missing_image_and_runtime_leave_inference_untouched(self):
        (self.state/'desired').write_text('inference')
        self.payload['settings'].update(runtime='singularity',container='/missing/image.sif')
        with patch.object(control.shutil,'which',return_value='/usr/bin/singularity'):
            with self.assertRaisesRegex(ValueError,'container image'):self.call('training-start')
        with patch.object(control.shutil,'which',return_value=None):
            with self.assertRaisesRegex(ValueError,'not installed'):self.call('training-start')
        self.assertEqual((self.state/'desired').read_text(),'inference')
    def test_export_uses_container(self):
        image=self.root/'image.sif';image.touch();adapter=self.root/'adapter';adapter.mkdir();(adapter/'adapter_config.json').write_text('{}')
        self.payload['settings'].update(runtime='singularity',container=str(image),python='',adapter=str(adapter),quantization='q8_0')
        with patch.object(control.shutil,'which',return_value='/usr/bin/singularity'):
            status=self.call('training-export')
        self.assertIn('python3',control.subprocess.run.call_args.args[0])
        request=json.loads((Path(status['output'])/'request.json').read_text())
        self.assertEqual(request['action'],'export');self.assertEqual(request['quantization'],'q8_0')

class LocalTests(unittest.TestCase):
    def test_preflight_failure_does_not_claim(self):
        from llm_away import training,resources,shared_sessions
        with patch.object(training,'state',return_value=(Path('/session'),{'token':'t','remote_port':12},Mock())),patch.object(shared_sessions,'current',return_value={'owner':{'id':'other'},'generation':'g'}),patch.object(shared_sessions,'claim') as claim,patch.object(resources,'remote',side_effect=ValueError('Runtime missing')):
            with self.assertRaisesRegex(ValueError,'Runtime missing'):training.operate(1,'start',{'paths':'/data','model':'base'},True)
            claim.assert_not_called()
    def test_f7_stop_passes_positional_arguments(self):
        from llm_away import training,monitor_ui
        with patch.object(monitor_ui,'dropdown_win',return_value='Stop and save'),patch.object(monitor_ui,'terminal_operation',side_effect=lambda func,*args:func(*args)),patch.object(training,'operate',side_effect=[{'run_id':'run'},'stopped']) as operate:
            self.assertEqual(training.menu(1),'stopped');operate.assert_called_with(1,'stop',None,False,'run')

class SnapshotTests(unittest.TestCase):
    def test_cross_root_answers_and_preserved_rag_snapshot(self):
        from llm_away import training,rag_transfer
        from llm_away.config import AppConfig
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);questions=root/'questions';answers=root/'answers'
            questions.mkdir();answers.mkdir();answer=answers/'answer.txt';answer.write_text('answer content')
            manifest=questions/dataset.MANIFEST;original='Question?\n['+str(answer)+']\n';manifest.write_text(original)
            cfg=AppConfig();cfg=replace(cfg,backend_type='direct',ssh=replace(cfg.ssh,connection='local'),remote=replace(cfg.remote,resource_state_dir=str(root/'remote')))
            old=Path(rag_transfer.remote_state(cfg,'token'));old.mkdir(parents=True);(old/'keep').write_text('RAG data')
            targets=training.local_sources(cfg,{},'token',[str(questions),str(answers),str(answer)],root/'local')
            self.assertEqual(len(targets),2);self.assertEqual(manifest.read_text(),original)
            self.assertEqual((old/'keep').read_text(),'RAG data')
            output,report=dataset.prepare(targets,root/'prepared')
            rows=[json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report['files_accepted'],1);self.assertEqual(rows[0]['answer'],'answer content')

class CheckpointTests(unittest.TestCase):
    def test_stop_and_training_exception_both_save_adapter(self):
        from types import SimpleNamespace
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'input.txt';source.write_text('a useful training answer')
                spec=dict(run_id='run',output=str(root/'out'),status_path=str(root/'status.json'),paths=[str(source)],
                          model='base',sequence_length=128,precision='qlora',lora_rank=16,epochs=1,
                          gradient_accumulation=1,learning_rate=.0002,save_steps=50)
                job=runner.Job(spec)
                tokenizer=Mock(eos_token_id=0,pad_token_id=0);tokenizer.encode.return_value=[1,2,3]
                # Avoid Mock's automatic `.tokenizer` processor attribute.
                del tokenizer.tokenizer
                model=Mock(hf_device_map={'a':0,'b':1})
                loader=Mock();loader.from_pretrained.return_value=(model,tokenizer);loader.get_peft_model.return_value=model
                torch=Mock();torch.cuda.is_available.return_value=True;torch.cuda.is_bf16_supported.return_value=True
                torch.cuda.device_count.return_value=2;torch.cuda.mem_get_info.return_value=(10000,12000)
                saved=[]
                class Trainer:
                    def __init__(self,**kwargs):self.kwargs=kwargs
                    def compute_loss(self,*args,**kwargs):return 1
                    def train(self):
                        if fail:raise RuntimeError('training failure')
                        row=self.kwargs['train_dataset'][0]
                        self.compute_loss(model,{'_records':[(row['example_id'],row['file_id'])]})
                        job.stop_path.touch();ctl=SimpleNamespace(should_save=False,should_training_stop=False)
                        self.kwargs['callbacks'][0].on_step_end(None,SimpleNamespace(global_step=1,max_steps=1,epoch=1),ctl)
                        assert ctl.should_save and ctl.should_training_stop
                    def save_state(self):saved.append(True)
                modules={'unsloth':SimpleNamespace(FastModel=loader),'torch':torch,
                         'transformers':SimpleNamespace(Trainer=Trainer,TrainingArguments=lambda **kw:kw,TrainerCallback=object)}
                with patch.dict(sys.modules,modules):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'training failure'):runner.train(job)
                    else:runner.train(job)
                model.save_pretrained.assert_called_once_with(str(root/'out/adapter'))
                tokenizer.save_pretrained.assert_called_once();self.assertTrue(saved)
                self.assertTrue(job.status['checkpoint_saved'])
                self.assertEqual(loader.from_pretrained.call_args.kwargs['device_map'],'balanced')
                if not fail:
                    self.assertEqual(job.status['phase'],'STOPPED');self.assertEqual(job.status['files_trained'],1)

if __name__=='__main__':unittest.main()
