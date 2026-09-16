"""Fresh MuJoCo execution driven only by the remote-in-process-family VLA worker.
No IK/planner imports, demonstration actions, trajectory replay, contact-based
steering, position assignment, artificial attachment, or scene modification.
Uses existing tableguard Python; inference uses the separate existing vla312.
"""
from pathlib import Path
import csv
import json
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tg12_common import read_json,write_json,sha,event,inside,OBS_KEYS
from tg12_control import DT,CHUNK,CONTACT_N,BoundedCommands,CupEvaluator,validate_contract
from tg12_mapping import validate_mapping


class GuardStop(RuntimeError): pass


def scene_asset_hashes(scene):
    tree=ET.parse(scene).getroot()
    if tree.find('.//include') is not None or tree.find('.//plugin') is not None or tree.find('equality') is not None:
        raise ValueError('Only the existing self-contained, unattached scene is accepted')
    compiler=tree.find('compiler');values={str(Path(scene).resolve()):sha(scene)}
    if compiler is None or compiler.get('angle')!='radian':raise ValueError('Unexpected scene units')
    for node in tree.findall('asset/*'):
        name=node.get('file')
        if not name:continue
        base=compiler.get('meshdir' if node.tag=='mesh' else 'texturedir',compiler.get('assetdir',''))
        p=(Path(scene).parent/base/name).resolve()
        if not p.is_file():raise FileNotFoundError(p)
        values[str(p)]=sha(p)
    return values


class Simulator:
    def __init__(self,cfg):
        self.cfg=cfg;self.out=Path(cfg['output']);self.ipc=self.out/'ipc'
        self.model=None;self.data=None;self.renderer=None;self.eval=None;self.filter=None
        self.camera_rows=[];self.steps=0;self.started=time.monotonic();self.setup_steps=0
        self.trace=None;self.policy_calls=0;self.policy_wait_s=0;self.assets={}
        self.report=dict(revision=cfg['revision'],status='initializing',error=None,stops=[],
            task_scope='learned cup-transfer integration diagnostic; NOT complete bimanual project',
            task_success=None,two_arm_task_success=None,camera_goal_verifier_implemented=False,
            selective_repair_implemented=False,official_challenge_score=None,
            checkpoint=cfg['checkpoint'],training_run=cfg['training_run'],
            source_training_episodes=1,checkpoint_total_updates=cfg['checkpoint_total_updates'],
            evaluation_environment_seen_in_training=True,held_out_generalization=False,
            backend=cfg['backend'],openvino_policy_inference=False,policy_calls=0,
            inverse_kinematics_calls=0,recorded_actions_replayed=False,execution_positions_assigned=False,
            external_lifting_force=False,weld_or_attachment=False,scene_modified=False,
            privileged_evaluation_and_safety_monitor=True,
            privileged_state_used_to_steer_policy=False,simulator_paused_during_inference=True,
            continuous_wall_clock_realtime=False,action_dt_s=DT,action_chunk_steps=CHUNK,
            initial_settle_s=.75,max_rollout_simulation_s=cfg['max_sim_s'],
            no_automatic_retry=True,policy_observations=['top RGB','right wrist RGB','12 joint positions','instruction'])
    def ident(self,kind,name):
        i=self.mj.mj_name2id(self.model,kind,name)
        if i<0:raise ValueError('Model element missing: '+name)
        return i
    def gname(self,g):return self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_GEOM,g) or 'unnamed_'+str(g)
    def load(self):
        import mujoco as mj
        self.mj=mj
        event(self.out,'simulator','load_original_passed_07c_scene')
        scene=Path(self.cfg['scene']);expected=self.cfg['scene_sha256']
        if sha(scene)!=expected:raise ValueError('Original 07C scene changed')
        self.assets=scene_asset_hashes(scene);write_json(self.out/'source_scene_hashes.json',self.assets)
        self.model=mj.MjModel.from_xml_path(str(scene));self.data=mj.MjData(self.model)
        m,d=self.model,self.data;mj.mj_forward(m,d)
        if not math.isclose(float(m.opt.timestep),DT,abs_tol=1e-10):raise ValueError('Timestep mismatch')
        if int(m.neq)!=0 or np.any(m.body_gravcomp>0):raise ValueError('Unexpected attachment/compensation')
        self.mapping=validate_mapping(m,d,mj);by={r['name']:r for r in self.mapping}
        c=read_json(Path(self.cfg['checkpoint'])/'contract.json')
        stats=read_json(Path(self.cfg['checkpoint'])/'normalization.json')
        self.order,low,high=validate_contract(c,stats);self.rows_map=[by[n] for n in self.order]
        if any(not np.isclose(r['low'],low[i],rtol=0,atol=1e-8) or not np.isclose(r['high'],high[i],rtol=0,atol=1e-8) for i,r in enumerate(self.rows_map)):
            raise ValueError('Compiled joint limits differ from training contract')
        self.cup=self.ident(mj.mjtObj.mjOBJ_BODY,'development_cup')
        self.gripper=self.ident(mj.mjtObj.mjOBJ_BODY,'right_gripper')
        self.moving=self.ident(mj.mjtObj.mjOBJ_BODY,'right_moving_jaw_so101_v1')
        self.table=self.ident(mj.mjtObj.mjOBJ_GEOM,'development_table')
        self.tip1=self.ident(mj.mjtObj.mjOBJ_GEOM,'right_fixed_jaw_sph_tip1')
        self.tip2=self.ident(mj.mjtObj.mjOBJ_GEOM,'right_moving_jaw_sph_tip1')
        self.props={n:self.ident(mj.mjtObj.mjOBJ_BODY,'development_'+n) for n in ('plate','fork')}
        self.cup_geoms={g for g in range(m.ngeom) if int(m.geom_bodyid[g])==self.cup}
        self.robot_geoms=set();self.jaw_group={}
        for g in range(m.ngeom):
            bid=int(m.geom_bodyid[g]);cur=bid
            while cur:
                bn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,cur) or ''
                if bn.startswith(('left_','right_')):self.robot_geoms.add(g);break
                cur=int(m.body_parentid[cur])
            gn=self.gname(g);mesh=''
            if int(m.geom_type[g])==int(mj.mjtGeom.mjGEOM_MESH):
                mesh=mj.mj_id2name(m,mj.mjtObj.mjOBJ_MESH,int(m.geom_dataid[g])) or ''
            if bid==self.gripper and (gn.startswith('right_fixed_jaw_sph_') or gn in
                {'right_fixed_jaw_box3','right_fixed_jaw_box4','right_fixed_jaw_box5','right_fixed_jaw_box6','right_fixed_jaw_box7'}
                or mesh=='wrist_roll_follower_so101_gripper_part0_v1'):
                self.jaw_group[g]='fixed'
            elif bid==self.moving:self.jaw_group[g]='moving'
        g=by['right_gripper']
        if not m.actuator_forcelimited[g['aid']] or not np.allclose(m.actuator_forcerange[g['aid']],[-.25,.25],atol=1e-8,rtol=0):
            raise ValueError('Right-gripper cap differs from recorded 07C model')
        for r in self.rows_map:d.ctrl[r['cid']]=r['q0']
        self.initial_q=np.array([r['q0'] for r in self.rows_map])
        self.warning0=np.array(d.warning.number,copy=True)
        self.prop_reference={n:d.xpos[b].copy() for n,b in self.props.items()}
        self.renderer=mj.Renderer(m,height=480,width=640)
        self.options=mj.MjvOption();self.options.geomgroup[3:5]=0;self.options.sitegroup[:]=0
        for n in ('top','right_wrist_cam'):self.ident(mj.mjtObj.mjOBJ_CAMERA,n)
        self.report.update(mujoco_version=mj.__version__,source_scene=str(scene),source_scene_sha256=expected,
                           native_actuator_mapping=self.mapping,action_order=self.order)
        # Same 0.75-second actuator hold used to settle the source demonstration.
        # This is a declared reset/setup step, NOT a task trajectory or a policy call.
        event(self.out,'simulator','initial_setup_hold',simulated_seconds=.75)
        for _ in range(150):
            self.check_cancel();mj.mj_step(m,d);mj.mj_forward(m,d);self.setup_steps+=1
            row,issues=self.measure()
            if issues:raise GuardStop('Setup: '+'; '.join(issues[:3]))
        self.initial_cup=d.xpos[self.cup].copy()
        self.prop_reference={n:d.xpos[b].copy() for n,b in self.props.items()}
        self.eval=CupEvaluator(self.initial_cup)
        prev=np.array([d.ctrl[r['cid']] for r in self.rows_map])
        self.filter=BoundedCommands(prev,low,high,self.cfg['bounded_commands'])
        self.rollout_start_sim=float(d.time)
        self.report['initial_cup_position_m']=self.initial_cup.tolist()
        self.report['evaluator_thresholds']={'lift_m':.025,'placement_m':.008,'cup_tilt_deg':30,
            'jaw_contact_N':.02,'support_fraction':.8,'contact_penetration_m':.003,
            'command_rate_rad_s':.35,'joint_speed_rad_s':3,'inactive_arm_displacement_rad':.15,
            'other_object_displacement_m':.010,'cup_lateral_distance_m':.035,
            'lift_and_release_window_s':.5,'destination_window_s':.25,
            'added_hand_clearance_above_cup_center_m':.04,'official_thresholds':False}
    def check_cancel(self):
        if (self.out/'cancel.request').exists():raise GuardStop('Cancelled by user or launcher')
        if time.monotonic()-self.started>self.cfg['wall_timeout_s']:raise TimeoutError('Rollout wall-clock budget exceeded')
    def measure(self):
        d,m,mj=self.data,self.model,self.mj;forces={'fixed':0.,'moving':0.};support=0.;issues=[]
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise GuardStop('Nonfinite physics state')
        if np.any(np.asarray(d.warning.number)>self.warning0):raise GuardStop('MuJoCo numerical/capacity warning')
        for i in range(d.ncon):
            c=d.contact[i];a,b=int(c.geom1),int(c.geom2);dist=float(c.dist)
            if a<0 or b<0 or not math.isfinite(dist):raise GuardStop('Unsupported contact')
            ca,cb=a in self.cup_geoms,b in self.cup_geoms;other=b if ca else a
            jaw=self.jaw_group.get(other) if ca or cb else None
            intended=bool(jaw and (ca or cb));robot=a in self.robot_geoms or b in self.robot_geoms
            f=np.zeros(6)
            if c.efc_address>=0:mj.mj_contactForce(m,d,i,f)
            if not np.isfinite(f).all():raise GuardStop('Nonfinite contact force')
            normal=max(0.,float(f[0]))
            if intended and dist<=.0005:forces[jaw]+=normal
            if (ca or cb) and other==self.table:support+=normal
            if dist<-.003:issues.append('Excess penetration')
            if robot and dist<=0 and not intended:issues.append('Unexpected robot contact: '+self.gname(a)+' / '+self.gname(b))
            if intended and normal>25:issues.append('Jaw contact force exceeds original 25 N guard')
        xyz=d.xpos[self.cup].copy();tilt=math.degrees(math.acos(float(np.clip(d.xmat[self.cup].reshape(3,3)[2,2],-1,1))))
        row=dict(step=self.steps,time_s=float(d.time),cup_x=float(xyz[0]),cup_y=float(xyz[1]),cup_z=float(xyz[2]),cup_tilt_deg=tilt,
            fixed_normal_n=forces['fixed'],moving_normal_n=forces['moving'],table_normal_n=support,
            two_jaw_contact=forces['fixed']>=CONTACT_N and forces['moving']>=CONTACT_N,
            jaw_above_cup_center_m=float(min(d.geom_xpos[self.tip1,2],d.geom_xpos[self.tip2,2])-xyz[2]))
        for i,r in enumerate(self.rows_map):
            q=float(d.qpos[r['qadr']]);v=float(d.qvel[r['vadr']]);cmd=float(d.ctrl[r['cid']])
            row[r['name']+'_position_rad']=q;row[r['name']+'_velocity_rad_s']=v
            row[r['name']+'_applied_command_rad']=cmd
            if not r['low']-.03<=q<=r['high']+.03:issues.append('Joint position range: '+r['name'])
            if abs(v)>3:issues.append('Joint velocity: '+r['name'])
            limit=.65 if r['name']=='right_gripper' else .12
            if abs(q-cmd)>limit:issues.append('Actuator tracking: '+r['name'])
            if r['name'].startswith('left_') and abs(q-r['q0'])>.15:issues.append('Inactive arm drift')
        for n,b in self.props.items():
            dis=float(np.linalg.norm(d.xpos[b]-self.prop_reference[n]));row[n+'_displacement_m']=dis
            if dis>.010:issues.append('Other object moved: '+n)
        if tilt>30:issues.append('Cup tilted above original 30 degree guard')
        if hasattr(self,'initial_cup') and np.linalg.norm(xyz[:2]-self.initial_cup[:2])>.035:
            issues.append('Cup lateral motion exceeds original 35 mm workspace')
        if np.any(d.xfrc_applied!=0) or np.any(d.qfrc_applied!=0):issues.append('Unexpected applied external force')
        return row,issues
    def capture(self,frame):
        obsdir=self.out/'observations';obsdir.mkdir(exist_ok=True)
        paths={};images=[]
        for camera in ('top','right_wrist_cam'):
            self.renderer.update_scene(self.data,camera=camera,scene_option=self.options)
            pix=self.renderer.render().copy()
            p=obsdir/f'{frame:05d}_{camera}.jpg'
            Image.fromarray(pix).save(p,quality=85)
            # Match training's JPEG85 decode path; exact policy pixels are these saved files.
            with Image.open(p) as im:images.append(np.asarray(im.convert('RGB')).copy())
            paths[camera]=p.relative_to(self.out).as_posix()
            # Windows-safe: inference receives fresh RGB through request NPZ files,
            # and replay uses immutable frame JPEGs. Avoid replacing a shared
            # latest_*.jpg that VS Code/Chrome/antivirus may temporarily hold open.
        return images,paths
    def step_loop(self):
        self.load();self.report['status']='running'
        write_json(self.out/'rollout_report.json',self.report)
        maxchunks=int(round(self.cfg['max_sim_s']/.1))
        self.trace=(self.out/'rollout_trace.csv').open('x',newline='',encoding='utf-8')
        writer=None;success=False
        with (self.out/'rollout_observations.jsonl').open('x',encoding='utf-8') as observations:
            for frame in range(maxchunks):
                self.check_cancel()
                images,paths=self.capture(frame)
                st=np.array([self.data.qpos[r['qadr']] for r in self.rows_map],dtype=np.float64)
                input_file=self.ipc/f'request_{frame:05d}.npz';tmp=input_file.with_suffix('.tmp')
                with tmp.open('xb') as f:np.savez_compressed(f,image_top=images[0],image_right_wrist=images[1],state=st)
                os.replace(tmp,input_file)
                req=dict(request_id=frame,simulation_time_s=float(self.data.time),instruction=self.cfg['instruction'],
                         input_file=input_file.name,input_sha256=sha(input_file))
                reqfile=self.ipc/f'request_{frame:05d}.json';write_json(reqfile,req)
                record=dict(frame_id=frame,step=self.steps,time_s=float(self.data.time),images=paths,
                    state=st.tolist(),instruction=self.cfg['instruction'],policy_request_id=frame,
                    action_order=self.order,policy_source='Notebook 15A multi-episode checkpoint; fresh closed-loop inference; fixed diffusion noise across requests',
                    inference_completed=False,applied_steps=0,episode_outcome='in_progress')
                self.camera_rows.append(record)
                write_json(self.out/'live_status.json',{'frame_id':frame,'simulation_time_s':float(self.data.time),
                    'stage':'waiting_for_policy','mode':'LIVE SIMULATION / LEARNED POLICY + BOUNDED COMMANDS',
                    'latest_cameras':paths,'evaluator_only':self.eval.report()})
                wait=time.monotonic();answer=self.ipc/f'response_{frame:05d}.json'
                while not answer.is_file():
                    self.check_cancel()
                    if (self.ipc/'policy_failed.json').exists():raise RuntimeError(read_json(self.ipc/'policy_failed.json')['error'])
                    if time.monotonic()-wait>90:raise TimeoutError('No policy response in 90 seconds')
                    time.sleep(.01)
                self.policy_wait_s+=time.monotonic()-wait
                response=read_json(answer)
                if response.get('request_id')!=frame or response.get('request_sha256')!=sha(reqfile):raise ValueError('Stale or mismatched action response')
                rp=inside(self.ipc,response['output_file'])
                if sha(rp)!=response['output_sha256']:raise ValueError('Policy output changed in transit')
                with np.load(rp,allow_pickle=False) as z:raw=np.array(z['requested_targets_rad'],copy=True)
                if raw.shape!=(20,12) or not np.isfinite(raw).all():raise ValueError('Invalid policy chunk')
                self.policy_calls+=1
                record.update(inference_completed=True,inference_s=response['inference_s'],noise_seed=response['seed'],
                              raw_action_chunk=raw.tolist(),action_chunk_dt_s=DT,applied_action_chunk=[])
                try:
                    for j in range(20):
                        self.check_cancel();cmd=self.filter.apply(raw[j])
                        for r,q in zip(self.rows_map,cmd):self.data.ctrl[r['cid']]=q
                        t=float(self.data.time)
                        self.mj.mj_step(self.model,self.data);self.steps+=1;self.mj.mj_forward(self.model,self.data)
                        if not math.isclose(float(self.data.time),t+DT,abs_tol=1e-8):raise GuardStop('Unexpected simulation reset/time jump')
                        record['applied_action_chunk'].append(cmd.tolist());record['applied_steps']+=1
                        row,issues=self.measure()
                        row.update(policy_frame_id=frame,chunk_step=j)
                        for r,q in zip(self.rows_map,raw[j]):row[r['name']+'_raw_policy_target_rad']=float(q)
                        if writer is None:writer=csv.DictWriter(self.trace,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row)
                        success=self.eval.observe(row)
                        if issues:raise GuardStop('; '.join(issues[:4]))
                        if success:break
                finally:
                    observations.write(json.dumps(record,allow_nan=False)+'\n');observations.flush();self.trace.flush()
                if frame%10==0 or success:
                    self.report.update(policy_calls=self.policy_calls,physics_steps=self.steps,simulation_time_s=float(self.data.time),
                                       evaluator=self.eval.report(),command_filter=self.filter.report())
                    write_json(self.out/'rollout_report.json',self.report)
                    event(self.out,'simulator','learned_execution_progress',policy_calls=self.policy_calls,simulation_time_s=float(self.data.time),
                          grasp=self.eval.flags['grasp_verified'],lift=self.eval.flags['lift_verified'],placement=self.eval.flags['placement_verified'])
                if success:break
        self.report.update(status='rollout_completed',task_success=bool(success),
            end_reason='cup_engineering_gates_passed' if success else 'simulation_time_budget_reached_without_complete_task')
    def finish(self):
        if self.trace:self.trace.close()
        if self.renderer:
            try:
                _,paths=self.capture(len(self.camera_rows))
                self.camera_rows.append(dict(frame_id=len(self.camera_rows),time_s=float(self.data.time),images=paths,final_snapshot=True))
                self.report['final_images']=paths
            except Exception as exc:self.report['final_render_error']=str(exc)
            finally:self.renderer.close()
        unchanged=all(Path(p).is_file() and sha(p)==h for p,h in self.assets.items()) if self.assets else None
        if unchanged is False:self.report.update(status='failed',task_success=False,error='Original scene or asset changed')
        self.report.update(source_scene_and_assets_unchanged=unchanged,
            physics_steps=self.steps,setup_physics_steps=self.setup_steps,policy_calls=self.policy_calls,
            simulation_time_s=float(self.data.time) if self.data else 0.,
            rollout_simulation_s=float(self.data.time)-getattr(self,'rollout_start_sim',float(self.data.time)) if self.data else 0.,
            worker_wall_time_s=time.monotonic()-self.started,total_policy_wait_s=self.policy_wait_s,
            recorded_observations=sum(not x.get('final_snapshot') for x in self.camera_rows))
        if self.eval:self.report['evaluator']=self.eval.report()
        if self.filter:self.report['command_filter']=self.filter.report()
        outcome=('passed_learned_cup_development_trial' if self.report.get('task_success') is True
                 else 'failed_or_partial_learned_trial' if self.report.get('task_success') is False
                 else 'no_task_outcome_established')
        with (self.out/'rollout_observations.jsonl').open('w',encoding='utf-8') as f:
            for record in self.camera_rows:
                if not record.get('final_snapshot'):
                    record['episode_outcome']=outcome
                    f.write(json.dumps(record,allow_nan=False)+'\n')
        write_json(self.out/'live_status.json',{'stage':self.report['status'],
            'task_success':self.report.get('task_success'),'latest_cameras':self.report.get('final_images',{}),
            'evaluator_only':self.eval.report() if self.eval else None,
            'mode':'FINISHED LEARNED-POLICY CUP ATTEMPT; NOT FULL PROJECT'})
        write_json(self.out/'replay_frames.json',self.camera_rows)
        write_json(self.out/'rollout_report.json',self.report)
        write_json(self.ipc/'simulator_done.json',{'status':self.report['status'],'task_success':self.report.get('task_success')})
        event(self.out,'simulator','finished',status=self.report['status'],task_success=self.report.get('task_success'),policy_calls=self.policy_calls)


def main(path):
    sim=Simulator(read_json(path))
    try:sim.step_loop()
    except GuardStop as exc:
        sim.report.update(status='rollout_stopped_by_guard',task_success=False if sim.steps else None,error=str(exc),stops=[str(exc)],end_reason='guard')
        event(sim.out,'simulator','guard_stop',reason=str(exc))
    except BaseException as exc:
        sim.report.update(status='failed',task_success=False if sim.steps else None,error=type(exc).__name__+': '+str(exc),stops=[str(exc)])
        (sim.out/'simulator_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        event(sim.out,'simulator','failed',error=sim.report['error'])
        raise
    finally:sim.finish()


if __name__=='__main__': main(sys.argv[1])
