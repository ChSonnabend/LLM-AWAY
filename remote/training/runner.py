#!/usr/bin/env python3
"""Allocation-bound Unsloth LoRA training, document distillation, and GGUF export."""
import collections
import json
import os
from pathlib import Path
import signal
import sys
import time
from urllib.request import Request, build_opener, ProxyHandler
from dataset import prepare

ACTIVE={'QUEUED','PREPARING','DISTILLING','LOADING','TRAINING','SAVING','EXPORTING','STOPPING'}

def write(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False));os.replace(temp,path)

class Job:
    def __init__(self,spec):
        self.spec=spec;self.output=Path(spec['output']);self.output.mkdir(parents=True,exist_ok=True)
        self.status_path=Path(spec['status_path']);self.stop_path=self.output/'stop'
        self.status=dict(run_id=spec['run_id'],output=str(self.output),phase='QUEUED',files_trained=0,files_total=0,percent=0)
        self.interrupted=False
    def update(self,**kw):
        self.status.update(kw,time=time.time());write(self.status_path,self.status)
    def stopped(self):return self.interrupted or self.stop_path.exists()
    def signal(self,*unused):self.interrupted=True

def distill(job,source):
    teacher=job.spec['teacher'];key_path=Path(job.spec['teacher_key_file'])
    key=key_path.read_text().strip() if key_path.exists() else ''
    target=job.output/'teacher-dataset.jsonl';opener=build_opener(ProxyHandler({}));files=set()
    with target.open('w') as out, source.open() as source_stream:
        for line in source_stream:
            row=json.loads(line);answer=row['answer']
            for offset in range(0,len(answer),12000):
                if job.stopped():raise InterruptedError('Stopped during teacher generation')
                context=answer[offset:offset+12000]
                prompt=('Answer this question using the reference: '+row['question'] if row['question'] else
                        'Create one useful question and its answer grounded in this reference. Return JSON with question and answer strings.')
                payload={'model':teacher['model'],'messages':[{'role':'user','content':prompt+'\n\nREFERENCE:\n'+context}],
                         'temperature':0.3,'max_tokens':int(teacher.get('max_tokens',2048))}
                headers={'Content-Type':'application/json'}
                if key:headers['Authorization']='Bearer '+key
                request=Request(teacher['base_url'].rstrip('/')+'/v1/chat/completions',data=json.dumps(payload).encode(),headers=headers)
                with opener.open(request,timeout=180) as response:body=json.load(response)
                choice=body['choices'][0]
                if choice.get('finish_reason')=='length':raise ValueError('Teacher response truncated; increase its output budget')
                text=choice['message']['content']
                if row['question']:result=dict(row,answer=text)
                else:
                    clean=text.strip()
                    if clean.startswith('```'):clean='\n'.join(clean.splitlines()[1:-1])
                    result=dict(json.loads(clean),file_id=row['file_id'])
                    if not isinstance(result.get('question'),str) or not isinstance(result.get('answer'),str):raise ValueError('Teacher did not return question/answer text')
                if not result['answer'].strip():raise ValueError('Teacher returned an empty answer')
                out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
            files.add(row['file_id']);job.update(phase='DISTILLING',teacher_files=len(files))
    return target

def token_examples(source,tokenizer,length,target,stopped):
    counts=collections.Counter();offsets=[]
    with target.open('wb') as stream, source.open() as source_stream:
        for line in source_stream:
            if stopped():raise InterruptedError('Stopped during tokenization')
            row=json.loads(line)
            prefix=[]
            if row['question']:
                prefix=tokenizer.apply_chat_template([{'role':'user','content':row['question']}],tokenize=True,add_generation_prompt=True)
                if len(prefix)>=length-16:raise ValueError('Question exceeds training sequence length: '+row['file_id'])
            tokens=tokenizer.encode(row['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
            size=length-len(prefix)
            for start in range(0,len(tokens),size):
                answer=tokens[start:start+size];ids=prefix+answer
                offsets.append(stream.tell());counts[row['file_id']]+=1
                record={'input_ids':ids,'labels':[-100]*len(prefix)+answer,'file_id':row['file_id'],'example_id':len(offsets)-1}
                stream.write((json.dumps(record)+'\n').encode())
    return offsets,counts

def train(job):
    # Unsloth must be imported before transformers/peft to install its patches.
    from unsloth import FastModel
    import torch
    from transformers import Trainer, TrainingArguments, TrainerCallback
    if not torch.cuda.is_available():raise ValueError('Training needs a CUDA-compatible GPU environment inside the allocation')
    spec=job.spec
    source,report=prepare(spec['paths'],job.output,job.update,job.stopped)
    job.update(files_total=report['files_accepted'],files_registered=report['files_total'],skipped=len(report['skipped']))
    if spec.get('teacher'):source=distill(job,source)
    if job.stopped():raise InterruptedError('Stopped before loading weights')
    job.update(phase='LOADING')
    memory={i:int(torch.cuda.mem_get_info(i)[0]*0.8) for i in range(torch.cuda.device_count())}
    model,tokenizer=FastModel.from_pretrained(model_name=spec['model'],max_seq_length=spec['sequence_length'],
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=spec['precision']=='qlora',device_map='balanced' if len(memory)>1 else {'':0},max_memory=memory,
        trust_remote_code=False)
    processor=tokenizer
    tokenizer=getattr(processor,'tokenizer',processor)
    # Do not pretend that CPU/disk inference dispatch is sharded GPU training.
    if any(str(v) in ('cpu','disk') for v in getattr(model,'hf_device_map',{}).values()):
        raise ValueError('Model exceeds allocated GPU memory even when sharded. Use QLoRA or allocate more GPUs; CPU/disk offload is not enabled for training')
    if tokenizer.eos_token_id is None:raise ValueError('Tokenizer needs an EOS token')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=FastModel.get_peft_model(model,r=spec['lora_rank'],lora_alpha=spec['lora_rank']*2,
        finetune_vision_layers=False,finetune_language_layers=True,
        finetune_attention_modules=True,finetune_mlp_modules=True,lora_dropout=0,
        bias='none',use_gradient_checkpointing='unsloth',random_state=3407)
    rows_path=job.output/'tokenized.jsonl'
    offsets,required=token_examples(source,tokenizer,spec['sequence_length'],rows_path,job.stopped)
    class Examples:
        def __len__(self):return len(offsets)
        def __getitem__(self,index):
            with rows_path.open('rb') as stream:stream.seek(offsets[index]);return json.loads(stream.readline())
    def collate(rows):
        width=max(len(r['input_ids']) for r in rows)
        return {'input_ids':torch.tensor([r['input_ids']+[tokenizer.pad_token_id]*(width-len(r['input_ids'])) for r in rows]),
                'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in rows]),
                'labels':torch.tensor([r['labels']+[-100]*(width-len(r['labels'])) for r in rows]),
                '_records':[(r['example_id'],r['file_id']) for r in rows]}
    seen=set();completed=collections.Counter();pending=[]
    class Progress(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            for identifier,file_id in pending:
                if identifier not in seen:seen.add(identifier);completed[file_id]+=1
            pending.clear();done=sum(completed[k]>=n for k,n in required.items())
            job.update(phase='TRAINING',step=state.global_step,steps_total=state.max_steps,epoch=state.epoch,
                       files_trained=done,files_total=len(required),percent=round(100*done/max(1,len(required)),2))
            if job.stopped():control.should_save=True;control.should_training_stop=True
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs:job.update(metrics={k:v for k,v in logs.items() if isinstance(v,(float,int))})
    class TrackingTrainer(Trainer):
        def compute_loss(self,model,inputs,*args,**kwargs):
            records=inputs.pop('_records');result=super().compute_loss(model,inputs,*args,**kwargs)
            pending.extend(records);return result
    args=TrainingArguments(output_dir=str(job.output/'checkpoints'),num_train_epochs=spec['epochs'],per_device_train_batch_size=1,
        gradient_accumulation_steps=spec['gradient_accumulation'],learning_rate=spec['learning_rate'],logging_steps=1,
        save_steps=spec['save_steps'],save_total_limit=2,bf16=torch.cuda.is_bf16_supported(),fp16=not torch.cuda.is_bf16_supported(),
        optim='adamw_8bit',report_to='none',remove_unused_columns=False,dataloader_num_workers=0,seed=3407)
    trainer=TrackingTrainer(model=model,args=args,train_dataset=Examples(),data_collator=collate,callbacks=[Progress()])
    job.update(phase='TRAINING',examples_total=len(offsets),gpus=len(memory),device_map={str(k):str(v) for k,v in getattr(model,'hf_device_map',{}).items()})
    failure=None
    try:
        if not job.stopped():trainer.train()
    except Exception as exc:failure=exc
    finally:
        job.update(phase='SAVING')
        model.save_pretrained(str(job.output/'adapter'));processor.save_pretrained(str(job.output/'adapter'));trainer.save_state()
        write(job.output/'base-model.json',{'model':spec['model'],'precision':spec['precision'],'sequence_length':spec['sequence_length']})
        job.update(adapter=str(job.output/'adapter'),checkpoint_saved=True)
    if failure:raise failure
    job.update(phase='STOPPED' if job.stopped() else 'COMPLETE')

def export(job):
    from unsloth import FastModel
    import torch
    spec=job.spec;job.update(phase='LOADING')
    model,tokenizer=FastModel.from_pretrained(model_name=spec['adapter'],max_seq_length=spec['sequence_length'],
        load_in_4bit=True,device_map='balanced' if torch.cuda.device_count()>1 else {'':0},trust_remote_code=False)
    if job.stopped():raise InterruptedError('Export stopped before conversion')
    job.update(phase='EXPORTING')
    model.save_pretrained_gguf(str(job.output/'gguf'),tokenizer,quantization_method=spec['quantization'])
    job.update(phase='COMPLETE',export_path=str(job.output/'gguf'))

def main():
    os.umask(0o077);job=Job(json.loads(Path(sys.argv[1]).read_text()))
    signal.signal(signal.SIGTERM,job.signal);signal.signal(signal.SIGINT,job.signal)
    try:
        if job.spec.get('action')=='export':export(job)
        else:train(job)
    except InterruptedError as exc:job.update(phase='STOPPED',message=str(exc),checkpoint_saved=job.status.get('checkpoint_saved',False))
    except BaseException as exc:
        job.update(phase='FAILED',error=type(exc).__name__+': '+str(exc));raise
if __name__=='__main__':main()
