"""Full-screen resource monitor. Reads daemon snapshots without extra SSH polling."""
import curses
import fcntl
import json
import re
import shlex
import textwrap
import threading
import time


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
            data['_busy']=busy(path.parent)
            if not (path.parent/'control.sock').exists() or time.time()-path.stat().st_mtime>45:
                data['_stale']=True
            rows.append(data)
        except (OSError,ValueError):continue
    return sorted(rows,key=lambda d:int(d['id']))


def detail(data):
    cfg=data.get('config',{});slurm=cfg.get('slurm',{});kind=cfg.get('backend_type','?')
    if kind=='slurm_server':
        options=['--nodes='+str(slurm.get('nodes',1)),'--gres=gpu:'+str(slurm.get('gpus',0))]
        if slurm.get('partition'):options+=['--partition='+slurm['partition']]
        if slurm.get('exclusive'):options+=['--exclusive']
        options+=slurm.get('custom_options',[])
        lines=['Slurm: '+shlex.join(options),'Node class: '+str(slurm.get('node_class') or 'any')]
    elif kind=='kubernetes':
        lines=['Kubernetes: '+json.dumps(cfg.get('kubernetes',{}),ensure_ascii=False)]
    else:lines=['Direct devices: '+str(cfg.get('llamacpp',{}).get('visible_devices') or 'auto')]
    allocation=data.get('allocation',{})
    lines+=['Remote path: '+str(cfg.get('remote',{}).get('workdir','')),
            'Model: '+str(data.get('model') or 'none')+' | backend: '+str(allocation.get('model_state','unknown')),
            'Agent: '+('in use' if data.get('_busy') else 'not attached')+' | PID: '+str(data.get('client_pid') or '—')]
    if data.get('_stale'):lines+=['Status may be stale: background daemon is offline or not updating.']
    if data.get('error'):lines+=['Error: '+str(data['error'])]
    return lines


def show(store,release,attach):
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
        log_id=None;log_top=None;log_lines=[];detail_offset=0
        preview_id=None;preview_scroll=0;preview_lines=[];preview_lock=threading.Lock();preview_loading=False;last_refresh=0
        result=[]
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
                        with (store/str(log_id)/'session.log').open('rb') as f:
                            size=f.seek(0,2);offset=max(0,size-262144);f.seek(offset)
                            if offset:f.readline()
                            log_lines=f.read().decode('utf-8','replace').splitlines()
                    except OSError as exc:log_lines=[str(exc)]
                def load_preview(number):
                    nonlocal preview_loading,preview_lines
                    try:
                        with (store/str(number)/'session.log').open('rb') as f:
                            size=f.seek(0,2);offset=max(0,size-4194304);f.seek(offset)
                            if offset:f.readline()
                            data=f.read().decode('utf-8','replace').splitlines()
                        with preview_lock:preview_lines=data
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
            if h<10 or w<45:
                put(0,'Enlarge terminal (minimum 45 columns × 10 rows).')
                put(h-1,'1 / F1 / q: Exit' if not pending else 'Release in progress…')
            elif log_id is not None:
                put(0,f' SESSION {log_id} — live allocation / provider traffic log',curses.A_BOLD|color(1))
                count=h-3;top=max(0,len(log_lines)-count) if log_top is None else min(log_top,max(0,len(log_lines)-count))
                for i,line in enumerate(log_lines[top:top+count],1):put(i,line)
                put(h-2,'Following' if log_top is None else 'Scrollback paused',color(3))
                put(h-1,'Esc / 1 / F1: Back   ↑↓ / PgUp/PgDn: Scroll   End: Follow',curses.A_REVERSE)
            else:
                pos=next((i for i,d in enumerate(rows) if d['id']==selected),0)
                put(0,f' RESOURCE MONITOR  |  {len(rows)} allocations  |  refreshed every second',curses.A_BOLD|color(1))
                if w>=95:
                    fixed=[6, max(12,w//8),6,12,15,12,14]
                    widths=fixed+[max(8,w-1-sum(fixed))]
                    headers=['ID','HOST','GPUS','SCHEDULER','STATE','AGENT','JOB','MODEL / NODE']
                    def cells(d):
                        cfg=d.get('config',{});a=d.get('allocation',{})
                        return [d['id'],d.get('host','?'),d.get('gpus',0),cfg.get('backend_type','?').replace('_server',''),
                                'STALE' if d.get('_stale') else d.get('phase','?'),'IN USE' if d['_busy'] else 'idle',a.get('job_id','—'),
                                str(d.get('model') or 'no model')+' / '+str(a.get('host') or 'pending')]
                else:
                    widths=[6,12,7,max(10,w-26)];headers=['ID','STATE','AGENT','HOST / MODEL']
                    def cells(d):return [d['id'],d.get('phase','?'),'BUSY' if d['_busy'] else 'idle',str(d.get('host','?'))+' / '+str(d.get('model') or 'none')]
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
                put(preview_y+1,' LOG PREVIEW  ↑/↓ scroll'+('' if preview_scroll else '  (following)'),curses.A_BOLD|color(1))
                pcount=h-4-(preview_y+2)
                ptop=max(0,len(preview_lines)-pcount-preview_scroll)
                if preview_id==selected and not preview_lines:
                    put(preview_y+2,'Loading recent log…',color(3))
                for i,line in enumerate(preview_lines[ptop:ptop+pcount],preview_y+2):
                    put(i,line)
                status=f'Release allocation {confirm} and stop its agent? Enter/y confirms; Esc cancels.' if confirm is not None else ('Stopping agent and releasing allocation…' if pending else message)
                separator(h-3)
                put(h-2,status,color(3))
                put(h-1,'1/F1 Exit   2/F2 Release   3/F3 Logs   4/F4 Attach   ←→ Select   ↑↓ Log   PgUp/Dn Details',curses.A_REVERSE)
            win.refresh();key=win.getch()
            if key==-1:continue
            if key==curses.KEY_RESIZE:last=0;continue
            if pending:
                # Do not abandon an in-flight release when exiting the UI.
                continue
            if log_id is not None:
                count=max(1,h-3)
                if key in (27,ord('q'),ord('1'),curses.KEY_F1,3):log_id=None;last=0
                elif key==curses.KEY_END:log_top=None
                elif key==curses.KEY_HOME:log_top=0
                elif key in (curses.KEY_UP,curses.KEY_PPAGE,curses.KEY_DOWN,curses.KEY_NPAGE):
                    if log_top is None:log_top=max(0,len(log_lines)-count)
                    delta=(-1 if key in (curses.KEY_UP,curses.KEY_PPAGE) else 1)*(count if key in (curses.KEY_PPAGE,curses.KEY_NPAGE) else 1)
                    log_top=max(0,min(log_top+delta,max(0,len(log_lines)-count)))
                continue
            if confirm is not None:
                if key in (10,13,ord('y'),ord('Y')):
                    pending=threading.Thread(target=release_worker,args=(confirm,),daemon=False);pending.start();confirm=None
                elif key in (27,ord('n'),ord('q')):confirm=None
                continue
            if key in (ord('1'),curses.KEY_F1,ord('q'),27,3):return
            if rows:
                pos=next((i for i,d in enumerate(rows) if d['id']==selected),0)
                if key in (curses.KEY_LEFT,ord('h')):selected=rows[max(0,pos-1)]['id'];detail_offset=0;preview_id=None
                elif key in (curses.KEY_RIGHT,ord('l')):selected=rows[min(len(rows)-1,pos+1)]['id'];detail_offset=0;preview_id=None
                elif key==curses.KEY_PPAGE:detail_offset=max(0,detail_offset-3)
                elif key==curses.KEY_NPAGE:detail_offset+=3
                elif key in (curses.KEY_UP,curses.KEY_DOWN):
                    preview_scroll=max(0,preview_scroll+(1 if key==curses.KEY_UP else -1))
                elif key in (ord('2'),curses.KEY_F2):confirm=selected
                elif key in (ord('3'),curses.KEY_F3):log_id=selected;log_top=None;last=0
                elif key in (ord('4'),curses.KEY_F4):
                    d=next((d for d in rows if d['id']==selected),{})
                    if not d.get('_busy') and d.get('phase')!='DAEMON OFFLINE':
                        return selected
                    if d.get('phase')=='DAEMON OFFLINE':message='Session daemon is offline; reconnect unavailable.'
    try:curses.wrapper(screen)
    except KeyboardInterrupt:pass
