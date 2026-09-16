"""Self-contained, explicitly labelled replay from actual saved camera images.
No network, model, simulator, or fabricated frames. Browser export is WebM.
"""
from pathlib import Path
import base64
import html
import json
from tg12_common import read_json,inside


def build_replay(run):
    run=Path(run);r=read_json(run/'rollout_report.json')
    entries=read_json(run/'replay_frames.json') if (run/'replay_frames.json').is_file() else []
    frames=[]
    for row in entries:
        if not row.get('images'):continue
        pictures={cam:'data:image/jpeg;base64,'+base64.b64encode(inside(run,name).read_bytes()).decode()
                  for cam,name in row['images'].items()}
        frames.append({'t':row['time_s'],'images':pictures})
    if not frames:return None
    evidence={'frames':frames,'status':r.get('status'),'success':r.get('task_success'),
        'backend':r.get('backend'),'run':run.name,'error':r.get('error'),
        'wall':r.get('worker_wall_time_s'),'filtered':r.get('command_filter',{}).get('modified_step_fraction')}
    payload=json.dumps(evidence,allow_nan=False).replace('<','\\u003c')
    page=HTML.replace('/*PAYLOAD*/',payload)
    p=run/'replay.html';p.write_text(page,encoding='utf-8')
    try:
        make_gif_preview(run, entries, r)
    except Exception as exc:
        (run/'replay_preview_error.txt').write_text(type(exc).__name__+': '+str(exc),encoding='utf-8')
    return p


def make_gif_preview(run, entries, report):
    """Actual saved frames, reduced resolution/frame rate; recorded sim-time pace."""
    import math
    from PIL import Image, ImageDraw
    valid=[r for r in entries if r.get('images',{}).get('top')]
    if not valid:return None
    stride=max(1,math.ceil(len(valid)/160))
    selected=list(range(0,len(valid),stride))
    if selected[-1]!=len(valid)-1:selected.append(len(valid)-1)
    frames=[];durations=[]
    for n,i in enumerate(selected):
        row=valid[i]
        with Image.open(inside(run,row['images']['top'])) as im:
            im=im.convert('RGB').resize((400,300))
        canvas=Image.new('RGB',(400,330),'black');canvas.paste(im,(0,0))
        draw=ImageDraw.Draw(canvas)
        draw.text((5,302),'RECORDED LEARNED POLICY | CUP-ONLY TEST',fill='white')
        label='PASS' if report.get('task_success') is True else 'FAILED/PARTIAL'
        draw.text((5,315),f"{label} | simulation t={row['time_s']:.3f}s | not real-time",fill='white')
        frames.append(canvas.convert('P',palette=Image.Palette.ADAPTIVE))
        dt=(valid[selected[n+1]]['time_s']-row['time_s']) if n+1<len(selected) else .1
        durations.append(max(10,round(1000*dt/10)*10))
    path=run/'replay_preview.gif'
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2)
    return path


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TableGuard 12 — recorded learned-policy attempt</title>
<style>body{margin:0;background:#101827;color:#edf2f7;font:16px system-ui}main{max-width:1000px;margin:auto;padding:25px}h1{font-size:26px}p{line-height:1.5}.note{color:#b9c9df}canvas{width:100%;max-width:800px;background:#000;border-radius:6px}button,select{padding:11px;margin:5px;border-radius:5px;border:1px solid #a6b4c7}input{width:95%}#state{font-weight:700}#save{color:#acf}pre{white-space:pre-wrap;word-break:break-word}footer{font-size:13px;color:#b9c9df}</style>
<main><h1>TableGuard • learned-policy execution replay</h1><p id="state"></p>
<p class="note">RECORDED REPLAY — not live execution. Cup-only development attempt; not complete two-arm task or selective repair.</p>
<canvas id="screen" width="640" height="528"></canvas><p id="clock"></p>
<div><button id="play">Play</button><button id="rewind">Restart replay</button>
<select id="camera"><option value="top">Overhead camera</option><option value="right_wrist_cam">Right-wrist camera</option></select>
<button id="export">Export replay as WebM</button><a id="save" hidden>Save WebM video</a></div>
<input id="seek" type="range" min="0" step="1" value="0"><pre id="details"></pre>
<footer>Playback follows recorded simulation timestamps. The simulator paused while waiting for each policy prediction; replay speed is not evidence of wall-clock real-time control. Only actual recorded frames are used. Joint-range/slew filtering is part of this controller and was logged separately from raw VLA actions.</footer></main>
<script>
const data=/*PAYLOAD*/;
const $=id=>document.getElementById(id),canvas=$('screen'),ctx=canvas.getContext('2d');
let current=0,playing=false,timer=null,recorder=null,blobURL=null,paintID=0,exporting=false;
const outcome=data.success===true?'CUP ENGINEERING CHECKS PASSED':data.success===false?'FAILED / PARTIAL CUP ATTEMPT':'OUTCOME NOT ESTABLISHED';
$('state').textContent=outcome+' | '+data.status;
$('details').textContent='Run: '+data.run+'\nBackend: '+data.backend+'\nRecorded worker wall time: '+data.wall+' s\nCommand-filter modified-step fraction: '+data.filtered+'\nStop/error: '+(data.error||'none reported');
$('seek').max=data.frames.length-1;
async function paint(){const id=++paintID;const f=data.frames[current];let im=new Image();im.src=f.images[$('camera').value];await im.decode();if(id!==paintID)return;
ctx.fillStyle='#000';ctx.fillRect(0,0,640,528);ctx.drawImage(im,0,0,640,480);ctx.fillStyle='#fff';ctx.font='12px sans-serif';
ctx.fillText('RECORDED LEARNED POLICY + BOUNDED COMMANDS | '+outcome,8,496);
ctx.fillText('sim t='+f.t.toFixed(3)+' s | '+data.backend+' | not full two-arm task',8,516);
$('clock').textContent='Recorded simulation time: '+f.t.toFixed(3)+' s | frame '+(current+1)+' / '+data.frames.length;$('seek').value=current;}
function stop(){playing=false;clearTimeout(timer);$('play').textContent='Play';if(recorder&&recorder.state==='recording')recorder.stop();}
async function next(){if(!playing)return;await paint();if(current>=data.frames.length-1){timer=setTimeout(stop,120);return;}
let ms=Math.max(1,1000*(data.frames[current+1].t-data.frames[current].t));timer=setTimeout(()=>{current++;next().catch(showError)},ms);}
function showError(e){stop();$('details').textContent+='\nReplay error: '+e.message;}
$('play').onclick=()=>{if(exporting)return;if(playing){stop();return;}if(current===data.frames.length-1)current=0;playing=true;$('play').textContent='Pause';next().catch(showError)};
$('rewind').onclick=()=>{if(exporting)return;stop();current=0;paint().catch(showError)};
$('seek').oninput=()=>{if(exporting)return;stop();current=Number($('seek').value);paint().catch(showError)};
$('camera').onchange=()=>{if(!exporting)paint().catch(showError)};
$('export').onclick=async()=>{try{if(exporting)return;if(!window.MediaRecorder||!canvas.captureStream)throw Error('WebM recording unavailable in this browser; use the replay controls.');
stop();current=0;await paint();exporting=true;$('camera').disabled=true;$('seek').disabled=true;$('export').disabled=true;
const types=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'];const mime=types.find(t=>MediaRecorder.isTypeSupported(t));if(!mime)throw Error('No supported WebM recorder.');
const stream=canvas.captureStream(30),chunks=[];recorder=new MediaRecorder(stream,{mimeType:mime});recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
recorder.onstop=()=>{stream.getTracks().forEach(t=>t.stop());if(blobURL)URL.revokeObjectURL(blobURL);blobURL=URL.createObjectURL(new Blob(chunks,{type:mime}));
$('save').href=blobURL;$('save').download='TableGuard12_'+data.run+'.webm';$('save').hidden=false;exporting=false;$('camera').disabled=false;$('seek').disabled=false;$('export').disabled=false;recorder=null;};
recorder.start();playing=true;$('play').textContent='Exporting — keep tab visible';next().catch(showError);
}catch(e){exporting=false;$('camera').disabled=false;$('seek').disabled=false;$('export').disabled=false;showError(e)}};
paint().catch(showError);
</script></html>'''
