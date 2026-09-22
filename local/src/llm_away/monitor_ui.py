"""Full-screen resource monitor. Reads daemon snapshots without extra SSH polling."""
import curses
import fcntl
import json
import re
import shlex
import textwrap
import threading
import time
from pathlib import Path


_active_window = None
_shell_mode = None

def _terminal_screen(screen):
    """Restore the actual entry TTY mode, including output newline translation.

    Repeated curses sessions may retain an earlier shell-mode snapshot. Never
    let that snapshot become the mode inherited by tmux or the next prompt.
    """
    import sys
    import termios
    global _active_window, _shell_mode
    if _active_window is not None:
        return screen(_active_window)
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    def enter(win):
        global _active_window, _shell_mode
        _active_window, _shell_mode = win, previous
        return screen(win)
    try:
        return curses.wrapper(enter)
    finally:
        _active_window = _shell_mode = None
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def terminal_operation(func, *args):
    """Run non-UI work without exposing the shell or inheriting curses TTY modes."""
    if _active_window is None:
        return func(*args)
    import os
    import sys
    import tempfile
    import termios
    fd = sys.stdin.fileno()
    current = termios.tcgetattr(fd)
    sys.stdout.flush();sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    try:
        with tempfile.TemporaryFile() as output:
            os.dup2(output.fileno(), 1);os.dup2(output.fileno(), 2)
            termios.tcsetattr(fd, termios.TCSADRAIN, _shell_mode)
            try:
                return func(*args)
            finally:
                sys.stdout.flush();sys.stderr.flush()
    finally:
        for target, source in zip((1, 2), saved):
            os.dup2(source, target);os.close(source)
        termios.tcsetattr(fd, termios.TCSADRAIN, current)


def busy(path):
    lease=path/'client.lock'
    if not lease.exists():return False
    try:
        with lease.open('a') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
            fcntl.flock(stream,fcntl.LOCK_UN)
    except OSError:pass
    return False


def snapshots(store):
    rows=[]
    for path in store.glob('*/session.json'):
        try:
            data=json.loads(path.read_text())
            if data.get('phase')=='RELEASED':continue
            if data.get('native'):
                from .native_sessions import status
                data=status(path.parent,data)
            try:
                from .session_guard import ensure
                data['ps']=ensure(path.parent,data)
            except Exception:data['ps']=None
            data['_busy']=busy(path.parent)
            selection=path.parent/'agent-selection.json'
            data['agent_location']=json.loads(selection.read_text()).get('location','local') if selection.exists() else 'local'
            if selection.exists():
                try:
                    loading=json.loads(selection.read_text()).get('loading')
                    if loading:
                        cfg=loading.get('config',{}).get('llamacpp',{})
                        model=loading.get('model',{})
                        data['loading']={
                            'model':model.get('alias') or model.get('name'),
                            'mtp':loading.get('mtp'),
                            'context_size':cfg.get('context_size'),
                            'server_extra_args':cfg.get('server_extra_args',[]),
                        }
                except (OSError,ValueError):pass
            from .terminals import engine
            terminal=(engine()['capture'](path.parent) if data['agent_location']=='local' and engine()['alive'](path.parent)
                      else data.get('allocation',{}).get('terminal',{}))
            data['_terminal']=terminal.get('status')=='TERMINAL'
            if data['_terminal']:data.setdefault('allocation',{})['prompt']=terminal
            result_path=path.parent/'agent-result.json'
            if not data['_terminal'] and data['agent_location']=='local' and result_path.exists():
                data.setdefault('allocation',{})['prompt']=json.loads(result_path.read_text())
            if not data.get('native') and not data.get('allocation_cleaned') and (not (path.parent/'control.sock').exists() or time.time()-path.stat().st_mtime>45):
                data['_stale']=True
            rows.append(data)
        except (OSError,ValueError):continue
    from .helper_relations import annotate
    return sorted(annotate(store,rows),key=lambda d:int(d['id']))


def detail(data):
    relations=['PS: '+str(data.get('ps') or '—'), 'IS MASTER: '+data.get('is_master','—'), 'IS HELPER: '+data.get('is_slave','—')]
    if data.get('native'):
        return relations+['Native CLI: '+data['native_cli'], 'Host: '+data['host'],
                'State: '+data['phase'], 'No GPU allocation or model server.']+([data['error']] if data.get('error') else [])
    cfg=data.get('config',{});slurm=cfg.get('slurm',{});kind=cfg.get('backend_type','?')
    llama=cfg.get('llamacpp',{})
    if kind=='slurm_server':
        options=['--nodes='+str(slurm.get('nodes',1)),'--gres=gpu:'+str(slurm.get('gpus',0))]
        if slurm.get('partition'):options+=['--partition='+slurm['partition']]
        if slurm.get('exclusive'):options+=['--exclusive']
        options+=slurm.get('custom_options',[])
        class_name=str(slurm.get('node_class') or 'any')
        if slurm.get('mi50_fallback') and class_name=='mi100':
            class_name+=' (MI50 fallback enabled)'
        elif class_name=='any':
            class_name+=' (no node-class filter)'
        lines=['Slurm: '+shlex.join(options),
               'Node class: '+class_name,
               'GPU backend: '+str(llama.get('backend') or 'unknown')+
               (' | ROCm arch: '+str(llama['rocm_arch']) if llama.get('rocm_arch') and llama['rocm_arch']!='auto' else '')]
    elif kind=='kubernetes':
        lines=['Kubernetes: '+json.dumps(cfg.get('kubernetes',{}),ensure_ascii=False)]
    else:lines=['Direct devices: '+str(cfg.get('llamacpp',{}).get('visible_devices') or 'auto')]
    allocation=data.get('allocation',{})
    lines+=['Remote path: '+str(cfg.get('remote',{}).get('workdir','')),
            'Model: '+str(data.get('model') or 'none')+' | backend: '+str(allocation.get('model_state','unknown')),
            'Agent: '+('in use' if data.get('_busy') else 'not attached')+' | PID: '+str(data.get('client_pid') or '—')]
    job=allocation.get('prompt',{})
    if job:lines+=['Agent prompt: '+job.get('status','?')+(' | '+job['error'] if job.get('error') else '')]
    if data.get('_stale'):lines+=['Status may be stale: background daemon is offline or not updating.']
    if data.get('error'):lines+=['Error: '+str(data['error'])]
    return relations+lines


class AllocateSelected(Exception):pass


class AttachSelected(Exception):pass


class SelectModelThenAttach(Exception):pass


class RefreshSelected(Exception):pass


class HelperSelected(Exception):pass


class RestartSelected(Exception):pass


class ReloadSelected(Exception):pass


class MaintenanceSelected(Exception):pass


class ReconnectSelected(Exception):pass


def model_select_win(models,loader,number,current=''):
    """Curses dropdown for allocating a model (or none) to one resource."""
    from .models import mtp_label
    labels=[f"{m['name']} — {m['size_bytes'] / 1024**3:.1f} GiB; {mtp_label(m)}" for m in models]
    labels.append('No model — keep resource allocation only')
    default=next((i for i,m in enumerate(models) if m.get('name')==current),0)
    def screen(win):
        try:curses.curs_set(0)
        except curses.error:pass
        win.keypad(True);win.timeout(200)
        index=default
        while True:
            h,w=win.getmaxyx();win.erase()
            def put(y,text,attr=0):
                if 0<=y<h:
                    try:win.addnstr(y,0,text[:max(0,w-1)].ljust(w-1),max(0,w-1),attr)
                    except curses.error:pass
            put(0,f' RESOURCE {number} — ALLOCATE MODEL ',curses.A_BOLD)
            count=min(len(labels),max(1,h-4))
            top=max(0,min(index-count//2,len(labels)-count))
            for i in range(top,top+count):
                put(i-top+2,('▸ ' if i==index else '  ')+labels[i],curses.A_REVERSE if i==index else curses.A_NORMAL)
            put(h-1,'↑↓ choose; Enter allocates; q/Esc keeps resources only',curses.A_REVERSE)
            win.refresh();key=win.getch()
            if key in (curses.KEY_UP,ord('k')):index=(index-1)%len(labels)
            elif key in (curses.KEY_DOWN,ord('j')):index=(index+1)%len(labels)
            elif key in (10,13,32):return models[index] if index<len(models) else None
            elif 49<=key<=48+len(labels):return models[key-49] if key<=48+len(models) else None
            elif key in (27,ord('q'),3):return None
    return _terminal_screen(screen)


def mtp_select_win(models,number,current=''):
    """Model dropdown plus MTP choice; returns (model|None, mtp) with no terminal prompts."""
    from .models import mtp_label
    model=model_select_win(models,None,number,current)
    if model is None:return None,'auto'
    mtp=model.get('mtp',{})
    if mtp.get('available') and mtp.get('toggle_supported'):
        choice=dropdown_win('MTP — '+model['name'],['MTP on','MTP off'],'MTP on')
        if choice is None:return model,'auto'
        mtp='on' if choice=='MTP on' else 'off'
    else:mtp='auto'
    return model,mtp


class Canceled(Exception):pass


def _form_screen(title,rows,values):
    """rows: list of (kind,label,options,default); returns values list, or None on Esc."""
    def screen(win):
        try:curses.curs_set(1)
        except curses.error:pass
        win.keypad(True);win.timeout(200)
        index=0;editing=False;buffer=str(values[0])
        while True:
            h,w=win.getmaxyx();win.erase()
            def put(y,text,attr=0):
                if 0<=y<h:
                    try:win.addnstr(y,0,text[:max(0,w-1)].ljust(w-1),max(0,w-1),attr)
                    except curses.error:pass
            put(0,f' {title} ',curses.A_BOLD|curses.A_REVERSE)
            put(1,'↑↓ move; Enter edit/choose; Esc cancel',curses.A_DIM)
            for i,(kind,label,options,default) in enumerate(rows):
                value='['+buffer+']' if editing and i==index else str(values[i])
                attr=curses.A_REVERSE if i==index else curses.A_NORMAL
                put(3+i,f'{label}: {value}',attr)
            confirm=len(rows)
            attr=curses.A_REVERSE|curses.A_BOLD if index==confirm else curses.A_NORMAL
            put(3+confirm,'  ▶ Start allocation  ',attr)
            put(h-1,'Enter: select/edit   Esc: cancel',curses.A_REVERSE)
            win.refresh();key=win.getch()
            if editing:
                if key in (10,13):
                    values[index]=buffer if buffer.strip() else str(rows[index][3]);editing=False;buffer=''
                elif key in (27,3):editing=False;buffer=''
                elif key in (curses.KEY_BACKSPACE,127,8):buffer=buffer[:-1]
                elif 32<=key<127:buffer+=chr(key)
                continue
            if key in (curses.KEY_UP,ord('k')):index=(index-1)%(len(rows)+1)
            elif key in (curses.KEY_DOWN,ord('j')):index=(index+1)%(len(rows)+1)
            elif key in (10,13):
                if index==confirm:return values
                kind,label,options,default=rows[index]
                if kind=='text':
                    editing=True;buffer=str(values[index])
                else:
                    pick=_sub_screen(lambda win:dropdown_win(label,options,values[index] if values[index] in options else 0,win))
                    if pick is not None:values[index]=pick
            elif key==27:return None
    return _terminal_screen(screen)


def _sub_screen(func):
    """Run an inner screen on a fresh window without ending the outer curses session."""
    win=curses.newwin(0,0)
    try:return func(win)
    finally:del win


def dropdown_win(title,options,default=0,win=None):
    """Full-screen dropdown; returns selected option string or None on Esc."""
    if not options:return None
    if win is None:
        return _terminal_screen(lambda w:dropdown_win(title,options,default,w))
    index=default if isinstance(default,int) else max(0,options.index(default) if default in options else 0)
    def screen(win):
        nonlocal index
        try:curses.curs_set(0)
        except curses.error:pass
        win.keypad(True);win.timeout(200)
        while True:
            h,w=win.getmaxyx();win.erase()
            def put(y,text,attr=0):
                if 0<=y<h:
                    try:win.addnstr(y,0,text[:max(0,w-1)].ljust(w-1),max(0,w-1),attr)
                    except curses.error:pass
            put(0,f' {title} — ↑↓ or number; Enter selects; Esc cancels ',curses.A_BOLD|curses.A_REVERSE)
            count=min(len(options),max(1,h-3));top=max(0,min(index-count//2,len(options)-count))
            for i in range(top,top+count):
                put(i-top+2,('▸ ' if i==index else '  ')+f'{i+1}. {options[i]}',curses.A_REVERSE if i==index else curses.A_NORMAL)
            win.refresh();key=win.getch()
            if key in (curses.KEY_UP,ord('k')):index=(index-1)%len(options)
            elif key in (curses.KEY_DOWN,ord('j')):index=(index+1)%len(options)
            elif key in (10,13):return options[index]
            elif 49<=key<=48+len(options):return options[key-49]
            elif key in (27,ord('q'),3):return None
    return screen(win) if win else _terminal_screen(screen)


def allocation_wizard():
    """Full-screen allocation form. Returns settings dict or None if canceled."""
    import os as _os
    from .config import load_config
    cfg=load_config(_os.environ.get('LLM_REMOTE_CONFIG',str(Path(__file__).resolve().parents[2]/'config/model.toml')))
    from .onboarding import ssh_hosts
    aliases,_=ssh_hosts()
    host_options=aliases or []
    rows=[
        ('choice','Location',['local','ssh'],'ssh'),
        ('choice','Session type',['Custom model','Native CLI (own model/account)'],'Custom model'),
        ('choice','SSH host',host_options,host_options[0] if host_options else ''),
        ('text','GPUs',[str(cfg.slurm.gpus or 1)],str(cfg.slurm.gpus or 1)),
        ('text','Additional Slurm options',None,shlex.join(cfg.slurm.custom_options)),
    ]
    values=[rows[0][3],rows[1][3],rows[2][3],rows[3][3],rows[4][3]]
    result=_form_screen('ALLOCATE RESOURCES',rows,values)
    if result is None:return None
    connection,mode,host,gpus,slurm_options=result
    settings=dict(connection=connection,native=mode.startswith('Native'),host=host,gpus=gpus,
                  slurm_options=slurm_options)
    if settings['native']:
        cli=dropdown_win('Native CLI',['codex','claude'],'codex')
        if cli is None:return None
        settings['cli']=cli
    return settings


def show(store,release,attach,submit=None,refresh=None,allocate=None,set_helper=None,restart=None,unload=None,refresh_monitor=None,cleanup_monitor=None,reconnect=None):
    """Release runs off the UI thread; terminal state is restored on every exit."""
    def screen(win):
        try:curses.curs_set(0)
        except curses.error:pass
        win.keypad(True);win.timeout(200)
        if curses.has_colors():
            curses.start_color()
            try:curses.use_default_colors()
            except curses.error:pass
            for i,color in enumerate((curses.COLOR_CYAN,curses.COLOR_GREEN,curses.COLOR_YELLOW,curses.COLOR_RED),1):
                curses.init_pair(i,color,-1)
        color=lambda n:curses.color_pair(n) if curses.has_colors() else 0
        def clean(value):
            value=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',str(value))
            return ''.join(c if c.isprintable() else ' ' for c in value)
        def put(y,text,attr=0):
            h,w=win.getmaxyx()
            if 0<=y<h:
                try:win.addnstr(y,0,clean(text).ljust(w-1),max(0,w-1),attr)
                except curses.error:pass
        def separator(y):
            h,w=win.getmaxyx()
            if 0<=y<h:
                try:win.hline(y,0,curses.ACS_HLINE,w,color(1))
                except curses.error:pass
        rows=[];selected=None;last=0;message='';pending=None;confirm=None
        log_id=None;log_top=None;log_lines=[];detail_offset=0;log_name='session.log';monitor_mode='Agent reply terminal'
        preview_id=None;preview_scroll=0;preview_lines=[];preview_lock=threading.Lock();preview_loading=False;last_refresh=0
        result=[]
        prompt_text=None
        tools_open=False
        menu=None;menu_index=0
        menus={
            'logs':['Session telemetry log','Helper log'],
            'release':['Unload model — keep resources','Reload a different model','Release resources — kill job'],
            'monitor':['Agent reply terminal','Telemetry logs','Helper log','Resource monitor','Model loading options'],
            'allocate':['Run res-alloc (allocate new resources)','Allocate a model to selected resource'],
        }
        def close_menu():nonlocal menu,menu_index;menu=None;menu_index=0
        def draw_menu(title,options,index):
            h,w=win.getmaxyx();height=len(options)+2
            top=max(0,h-height-2)
            for y in range(top,top+height):
                put(y,' '*(w-1),curses.A_REVERSE)
            put(top,f' {title} — ↑↓ choose; Enter selects; Esc cancels ',curses.A_REVERSE|curses.A_BOLD)
            for i,label in enumerate(options):
                attr=curses.A_REVERSE if i==index else curses.A_NORMAL
                put(top+i+1,f'  {i+1}. {label}  ',attr)
        def menu_pick(index):
            nonlocal log_name,log_id,log_top,last,message,confirm,pending,monitor_mode,preview_id,preview_scroll
            current=menu
            close_menu()
            if current=='logs':
                if selected is None:message='No allocation selected.'
                else:
                    log_name='helper.log' if index==1 else 'session.log'
                    log_id=selected;log_top=None;last=0
            elif current=='monitor':
                monitor_mode=menus['monitor'][index];preview_id=None;preview_scroll=0;last=0
            elif current=='release':
                if index==2:confirm=selected
                elif unload is None:message='Unload unavailable.'
                elif index==1:raise ReloadSelected(selected)
                else:
                    pending=threading.Thread(target=unload_worker,args=(selected,),daemon=False);pending.start()
            elif current=='allocate':
                if index==0:
                    if not allocate:message='Allocator unavailable.'
                    else:raise AllocateSelected()
                elif selected is None:message='No allocation selected.'
                else:raise AttachSelected(selected)
        def prompt_worker(number,text):
            try:
                answer=submit(number,text);result.append('Agent task '+answer['id'][:8]+' queued; safe to close this terminal.')
            except Exception as exc:result.append('Prompt failed: '+str(exc))
        def unload_worker(number):
            try:unload(number);result.append(f'Model unloaded; allocation {number} retained.')
            except Exception as exc:result.append('Unload failed: '+str(exc))
        def release_worker(number):
            try:release(number);result.append(f'Allocation {number} released.')
            except Exception as exc:result.append('Release failed: '+str(exc))
        while True:
            now=time.monotonic()
            if now-last>=1:
                rows=snapshots(store);last=now
                if selected not in [d['id'] for d in rows]:selected=rows[0]['id'] if rows else None
                if log_id is not None:
                    try:
                        with (store/str(log_id)/log_name).open('rb') as f:
                            size=f.seek(0,2);offset=max(0,size-262144);f.seek(offset)
                            if offset:f.readline()
                            log_lines=f.read().decode('utf-8','replace').splitlines()
                    except FileNotFoundError:
                        log_lines=(['No helper log yet. Helper logging may be disabled, or no query has run.']
                                   if log_name=='helper.log' else ['No session log yet.'])
                    except OSError as exc:log_lines=[str(exc)]
                def load_preview(number):
                    nonlocal preview_loading,preview_lines
                    preview_name='helper.log' if monitor_mode=='Helper log' else 'session.log'
                    try:
                        with (store/str(number)/preview_name).open('rb') as f:
                            size=f.seek(0,2);offset=max(0,size-4194304);f.seek(offset)
                            if offset:f.readline()
                            data=f.read().decode('utf-8','replace').splitlines()
                        with preview_lock:preview_lines=data
                    except FileNotFoundError:
                        message=('No helper log yet. Helper logging is enabled after the first helper request.'
                                 if preview_name=='helper.log' else 'No session log yet.')
                        with preview_lock:preview_lines=[message]
                    except OSError as exc:
                        with preview_lock:preview_lines=[str(exc)]
                    finally:preview_loading=False
                if selected is not None and not preview_loading and (preview_id!=selected or now-last_refresh>=1):
                    if preview_id!=selected:
                        preview_scroll=0;preview_id=selected
                        with preview_lock:preview_lines=[]
                    preview_loading=True;last_refresh=now
                    threading.Thread(target=load_preview,args=(selected,),daemon=True).start()
            if pending and not pending.is_alive():
                message=result.pop() if result else 'Operation finished.';pending=None;last=0
            h,w=win.getmaxyx();win.erase()
            def wrapped_log_lines():
                result=[]
                for line in log_lines:
                    result.extend(textwrap.wrap(clean(line),max(1,w-1),replace_whitespace=False,
                                               drop_whitespace=False) or [''])
                return result
            if h<10 or w<45:
                put(0,'Enlarge terminal (minimum 45 columns × 10 rows).')
                put(h-1,'q / Esc: Exit' if not pending else 'Release in progress…')
            elif log_id is not None:
                log_title='helper log' if log_name=='helper.log' else 'allocation / provider traffic log'
                put(0,f' SESSION {log_id} — live {log_title}',curses.A_BOLD|color(1))
                put(1,str(store/str(log_id)/log_name),color(3))
                visible=wrapped_log_lines()
                count=h-4;top=max(0,len(visible)-count) if log_top is None else min(log_top,max(0,len(visible)-count))
                for i,line in enumerate(visible[top:top+count],2):put(i,line)
                put(h-2,'Following' if log_top is None else 'Scrollback paused',color(3))
                put(h-1,'Esc / q: Back   ↑↓ / PgUp/PgDn: Scroll   ←→: Session   End: Follow',curses.A_REVERSE)
            elif menu is not None:
                draw_menu({'logs':'Choose log to view','allocate':'Allocator','release':'Release / unload','monitor':'Change monitor'}[menu],menus[menu],menu_index)
            else:
                pos=next((i for i,d in enumerate(rows) if d['id']==selected),0)
                put(0,f' RESOURCE MONITOR  |  {len(rows)} allocations  |  refreshed every second',curses.A_BOLD|color(1))
                if w>=95:
                    fixed=[5, max(10,w//9),5,10,14,9,11,8]
                    widths=fixed+[max(8,w-1-sum(fixed))]
                    headers=['ID','HOST','GPUS','SCHEDULER','STATE','AGENT','JOB','PS','MODEL / NODE']
                    def cells(d):
                        cfg=d.get('config',{});a=d.get('allocation',{})
                        return [d['id'],d.get('host','?'),d.get('gpus',0),'native' if d.get('native') else cfg.get('backend_type','?').replace('_server',''),
                                'STALE' if d.get('_stale') else d.get('phase','?'),'IN USE' if d['_busy'] else 'idle',a.get('job_id','—'),d.get('ps') or '—',
                                str(d.get('model') or d.get('native_cli') or 'no model')+' / '+str(a.get('host') or 'pending')]
                else:
                    widths=[5,10,7,8,max(10,w-31)];headers=['ID','STATE','AGENT','PS','HOST / MODEL']
                    def cells(d):return [d['id'],d.get('phase','?'),'BUSY' if d['_busy'] else 'idle',d.get('ps') or '—',str(d.get('host','?'))+' / '+str(d.get('model') or d.get('native_cli') or 'none')]
                if w>=125:
                    relation_width=max(11,min(20,(w-100)//2))
                    widths[-1]=max(8,widths[-1]-2*relation_width)
                    widths += [relation_width,relation_width]
                    headers += ['IS MASTER','IS HELPER']
                    original_cells=cells
                    def cells(d):return original_cells(d)+[d.get('is_master','—'),d.get('is_slave','—')]
                def formatted(values):return ''.join(clean(v)[:max(1,n-1)].ljust(n) for v,n in zip(values,widths))
                put(2,formatted(headers),curses.A_BOLD)
                count=max(1,(h-8)//3);top=max(0,min(pos-count+1,max(0,len(rows)-count)))
                for i,d in enumerate(rows[top:top+count],3):
                    attr=curses.A_REVERSE if d['id']==selected else color(2 if d['_busy'] else 1)
                    put(i,formatted(cells(d)),attr)
                if not rows:put(3,'No allocations. Create one with res-alloc.')
                separator(3+count)
                y=4+count
                put(y,' SELECTED ALLOCATION ',curses.A_BOLD|color(1));y+=1
                details=[part for line in (detail(rows[pos]) if rows else []) for part in (textwrap.wrap(clean(line),max(1,w-2)) or [''])]
                preview_y=max(y+1,(h-5)*2//3)
                detail_offset=min(detail_offset,max(0,len(details)-(preview_y-y)))
                for wrapped in details[detail_offset:]:
                    if y>=preview_y:break
                    put(y,wrapped);y+=1
                separator(preview_y)
                job=rows[pos].get('allocation',{}).get('prompt',{}) if rows else {}
                title=' '+monitor_mode.upper()+' — '+(job.get('status','') if monitor_mode=='Agent reply terminal' else '')
                put(preview_y+1,title,curses.A_BOLD|color(1))
                pcount=h-4-(preview_y+2)
                shown=([part for line in (job.get('text','') or job.get('error','') or 'Waiting for model…').splitlines() for part in (textwrap.wrap(line,max(1,w-2)) or [''])] if job and monitor_mode=='Agent reply terminal' else preview_lines)
                if monitor_mode=='Agent reply terminal' and not job:shown=['No agent reply yet.']
                if monitor_mode=='Resource monitor':shown=gpu_lines(rows[pos] if rows else {})
                elif monitor_mode=='Model loading options':shown=loading_options(rows[pos] if rows else {})
                ptop=max(0,len(shown)-pcount-preview_scroll)
                if monitor_mode in ('Telemetry logs','Helper log') and preview_id==selected and not preview_lines:
                    put(preview_y+2,'Loading recent log…',color(3))
                for i,line in enumerate(shown[ptop:ptop+pcount],preview_y+2):
                    put(i,line)
                status=f'Release allocation {confirm} and stop its agent? Enter/y confirms; Esc cancels.' if confirm is not None else ('Submitting operation…' if pending else message)
                separator(h-3)
                put(h-2,status,color(3))
                put(h-1,'q/Esc Exit   1/F1 Tools   2/F2 Attach   3/F3 Release   4/F4 Allocator   5/F5 Logs   6/F6 Change monitor   Space Prompt   ←→ Select   ↑↓ Log   PgUp/Dn Details',curses.A_REVERSE)
            if tools_open:
                put(h-1,'q/Esc Back   1/F1 Refresh   2/F2 Set helper   3/F3 Restart   4/F4 Refresh res-mon   5/F5 Cleanup   6/F6 Reconnect',curses.A_REVERSE)
            if prompt_text is not None:
                put(h-3,' AGENT PROMPT — Enter submits; Esc cancels',curses.A_REVERSE)
                put(h-2,'> '+prompt_text[-max(1,w-4):],curses.A_BOLD)
                put(h-1,'Agent has file, shell and network access on its configured host.')
            win.refresh();key=win.getch()
            if key==-1:continue
            if key==curses.KEY_RESIZE:last=0;continue
            if prompt_text is not None:
                if key in (27,3):prompt_text=None
                elif key in (10,13) and prompt_text.strip():
                    pending=threading.Thread(target=prompt_worker,args=(selected,prompt_text),daemon=False)
                    pending.start();prompt_text=None
                elif key in (curses.KEY_BACKSPACE,127,8):prompt_text=prompt_text[:-1]
                elif 32<=key<127 and len(prompt_text)<100000:prompt_text+=chr(key)
                continue
            if pending:
                # Do not abandon an in-flight release when exiting the UI.
                continue
            if menu is not None:
                count=len(menus[menu])
                if key in (27,ord('q'),3):close_menu()
                elif key in (curses.KEY_UP,ord('k')):menu_index=(menu_index-1)%count
                elif key in (curses.KEY_DOWN,ord('j')):menu_index=(menu_index+1)%count
                elif key in (10,13):menu_pick(menu_index)
                elif 49<=key<=48+count:menu_pick(key-49)
                continue
            if log_id is not None:
                count=max(1,h-4)
                if key in (27,ord('q'),3):log_id=None;last=0
                elif key in (curses.KEY_LEFT,curses.KEY_RIGHT) and rows:
                    pos=next((i for i,d in enumerate(rows) if d['id']==log_id),0)
                    pos=max(0,min(len(rows)-1,pos+(-1 if key==curses.KEY_LEFT else 1)))
                    selected=log_id=rows[pos]['id'];log_top=None;last=0
                elif key==curses.KEY_END:log_top=None
                elif key==curses.KEY_HOME:log_top=0
                elif key in (curses.KEY_UP,curses.KEY_PPAGE,curses.KEY_DOWN,curses.KEY_NPAGE):
                    visible=wrapped_log_lines()
                    if log_top is None:log_top=max(0,len(visible)-count)
                    delta=(-1 if key in (curses.KEY_UP,curses.KEY_PPAGE) else 1)*(count if key in (curses.KEY_PPAGE,curses.KEY_NPAGE) else 1)
                    log_top=max(0,min(log_top+delta,max(0,len(visible)-count)))
                continue
            if confirm is not None:
                if key in (10,13,ord('y'),ord('Y')):
                    pending=threading.Thread(target=release_worker,args=(confirm,),daemon=False);pending.start();confirm=None
                elif key in (27,ord('n'),ord('q')):confirm=None
                continue
            if tools_open:
                if key in (ord('q'),27,3):tools_open=False
                elif key in (ord('4'),curses.KEY_F4):raise MaintenanceSelected('refresh')
                elif key in (ord('5'),curses.KEY_F5):raise MaintenanceSelected('cleanup')
                elif key in (ord('6'),curses.KEY_F6):
                    if selected is None:message='No allocation selected.'
                    elif not reconnect:message='Reconnect unavailable.'
                    else:raise ReconnectSelected(selected)
                elif key in (ord('1'),curses.KEY_F1,ord('2'),curses.KEY_F2,ord('3'),curses.KEY_F3):
                    d=next((d for d in rows if d['id']==selected),{})
                    if selected is None:message='No allocation selected.'
                    elif d.get('native'):message='Native sessions have no helper model; exit and reopen the CLI to refresh.'
                    elif key in (ord('3'),curses.KEY_F3):
                        if restart:raise RestartSelected(selected)
                        else:message='Restart unavailable.'
                    elif key in (ord('2'),curses.KEY_F2):
                        if set_helper:raise HelperSelected(selected)
                        else:message='Helper registration unavailable.'
                    elif not refresh:message='Refresh unavailable.'
                    elif d.get('_busy') and not d.get('_terminal'):message='Wait for the current agent task before refreshing.'
                    else:raise RefreshSelected(selected)
                continue
            if key in (ord('q'),27,3):return
            if key in (ord('4'),curses.KEY_F4):
                menu='allocate';menu_index=0
                continue
            if key in (ord('1'),curses.KEY_F1):
                tools_open=True
                continue
            if rows:
                pos=next((i for i,d in enumerate(rows) if d['id']==selected),0)
                if key in (curses.KEY_LEFT,ord('h')):selected=rows[max(0,pos-1)]['id'];detail_offset=0;preview_id=None
                elif key in (curses.KEY_RIGHT,ord('l')):selected=rows[min(len(rows)-1,pos+1)]['id'];detail_offset=0;preview_id=None
                elif key==curses.KEY_PPAGE:detail_offset=max(0,detail_offset-3)
                elif key==curses.KEY_NPAGE:detail_offset+=3
                elif key in (curses.KEY_UP,curses.KEY_DOWN,ord('j'),ord('k')):
                    preview_scroll=max(0,preview_scroll+(1 if key in (curses.KEY_UP,ord('k')) else -1))
                elif key in (ord('2'),curses.KEY_F2):
                    d=next((d for d in rows if d['id']==selected),{})
                    if d.get('_terminal'):return selected
                    elif d.get('allocation',{}).get('prompt',{}).get('status') in ('QUEUED','RUNNING'):
                        message='An agent task is running; wait before attaching.'
                    elif not d.get('_busy'):
                        if not d.get('native') and (not d.get('model') or d.get('provider_exit') is not None or d.get('allocation',{}).get('model_state') in ('EXITED','IDLE')):
                            if callable(allocate):raise SelectModelThenAttach(selected)
                            message='Model selection unavailable.'
                        else:return selected
                    else:message='Another run command owns this session.'
                elif key in (ord('3'),curses.KEY_F3):
                    if selected is None:message='No allocation selected.'
                    else:menu='release';menu_index=0
                elif key in (ord('4'),curses.KEY_F4):
                    menu='allocate';menu_index=0
                elif key in (ord('5'),curses.KEY_F5):
                    if selected is None:message='No allocation selected.'
                    else:menu='logs';menu_index=0
                elif key in (ord('6'),curses.KEY_F6):menu='monitor';menu_index=0
                elif key==ord(' '):
                    d=rows[pos]
                    if not submit:message='Prompt submission unavailable.'
                    elif d.get('_busy') and not d.get('_terminal'):message='A legacy background task is still running.'
                    elif not d.get('model') and not d.get('native'):message='Load a model first with 4/F4.'
                    elif d.get('native') and not d.get('_terminal'):message='Open the native CLI first with 2/F2.'
                    elif d.get('allocation',{}).get('prompt',{}).get('status') in ('QUEUED','RUNNING'):message='A remote prompt is already running.'
                    else:prompt_text=''
                elif key in (ord('r'),ord('R')):
                    d=rows[pos]
                    if not refresh:message='Refresh unavailable.'
                    elif d.get('native'):message='Native sessions refresh by exiting and reopening with F2.'
                    elif d.get('_busy') and not d.get('_terminal'):message='Wait for the current agent task before refreshing.'
                    else:raise RefreshSelected(selected)
    def navigate(win):
        while True:
            try:return screen(win)
            except MaintenanceSelected as exc:
                try:
                    operation=refresh_monitor if exc.args[0]=='refresh' else cleanup_monitor
                    result=(terminal_operation(operation) if exc.args[0]=='refresh' else operation()) if operation else 'Unavailable'
                    dropdown_win(result or 'Done',['Back to monitor'])
                except Exception as error:dropdown_win(str(error),['Back to monitor'])
            except ReconnectSelected as exc:
                try:dropdown_win(terminal_operation(reconnect,exc.args[0]),['Back to monitor'])
                except Exception as error:dropdown_win('Reconnect failed: '+str(error),['Back to monitor'])
            except ReloadSelected as exc:
                try:
                    terminal_operation(unload,exc.args[0])
                    if callable(allocate):allocate('model',exc.args[0])
                except (OSError,ValueError,RuntimeError) as error:
                    dropdown_win('Reload failed: '+str(error),['Back to monitor'])
            except RestartSelected as exc:
                try:
                    terminal_operation(restart,exc.args[0])
                    dropdown_win('Allocation restarting — use F4 to load a model when ready',['Back to monitor'])
                except (OSError,ValueError,RuntimeError) as error:
                    dropdown_win('Restart failed: '+str(error),['Back to monitor'])
            except HelperSelected as exc:
                try:
                    terminal_operation(set_helper,exc.args[0])
                    dropdown_win('Helper registered — refresh consuming sessions to load it',['Back to monitor'])
                except (OSError,ValueError,RuntimeError) as error:
                    dropdown_win('Helper registration failed: '+str(error),['Back to monitor'])
            except RefreshSelected as exc:
                if callable(refresh):terminal_operation(refresh,exc.args[0])
            except AllocateSelected:
                if callable(allocate):allocate('allocate')
            except SelectModelThenAttach as exc:
                chosen=allocate('model',exc.args[0])
                if chosen is not None:return chosen
            except AttachSelected as exc:
                if callable(allocate):allocate('model',exc.args[0] if exc.args else None)
    try:return _terminal_screen(navigate)
    except KeyboardInterrupt:pass


def gpu_lines(data):
    telemetry=data.get('allocation',{}).get('gpu_telemetry',{})
    if not telemetry:return ['GPU telemetry unavailable — start the updated telemetry worker on this allocation.']
    age=max(0,int(time.time()-telemetry.get('timestamp',0)))
    lines=[f'GPU VRAM / utilization — sampled {age}s ago'+(' (STALE)' if age>20 else '')]
    raw=telemetry.get('lines',[])
    if raw and raw[0].startswith('index'):
        import csv
        for fields in csv.reader(raw[1:]):
            if len(fields)==6:
                index,uuid,name,used,total,util=[x.strip() for x in fields]
                lines.append(f'GPU {index}  {name}  VRAM {used} / {total}  GPU utilization {util}')
    else:lines.extend(raw)
    if telemetry.get('error'):lines.append(telemetry['error'])
    return lines


def loading_options(data):
    if data.get('native'):
        return ['Native CLI session — no model server or loading options.']
    loading=data.get('loading')
    if not loading:
        return ['No saved loading options. Load a model with F2/F4 first.']
    lines=['Model: '+str(loading.get('model') or 'unknown'),
           'MTP: '+str(loading.get('mtp') or 'auto')]
    if loading.get('context_size'):
        lines.append('Context size: '+str(loading['context_size']))
    args=loading.get('server_extra_args') or []
    lines.append('Server flags: '+(shlex.join(args) if args else '(none — preset defaults only)'))
    return lines
