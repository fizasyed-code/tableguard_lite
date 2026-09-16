"""Execution constraints and evaluator for one learned cup diagnostic.
Ground-truth task scores are NOT returned to the policy. No IK or scripted action.
"""
from collections import deque
import math
import numpy as np
from tg12_common import JOINTS

DT=0.005
CHUNK=20
COMMAND_RATE=0.35
CONTACT_N=0.02


def vector(x, shape, label):
    a=np.asarray(x,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all(): raise ValueError('Invalid '+label)
    return a


def validate_contract(c, stats):
    if c.get('cameras')!=['top','right_wrist_cam']: raise ValueError('Expected the two recorded cup cameras')
    if c.get('episodes')!=1 or c.get('action_dim')!=12 or c.get('state_dim')!=12:
        raise ValueError('Expected the existing one-episode 12-axis diagnostic')
    names=c.get('action_order')
    if not isinstance(names,list) or len(names)!=12 or set(names)!=set(JOINTS) or c.get('state_order')!=names:
        raise ValueError('State/action joint order is invalid')
    if c.get('action_chunk_length')!=CHUNK or not math.isclose(c.get('native_action_dt_s',0),DT,abs_tol=1e-10):
        raise ValueError('Native action contract changed; no implicit resampling')
    if not math.isclose(c.get('observation_interval_s',0),.1,abs_tol=1e-10): raise ValueError('Camera timing changed')
    if c.get('action_units')!='radians' or c.get('action_representation')!='absolute joint position targets':
        raise ValueError('Unexpected action semantics')
    for name in ('observation.state','action'):
        vector(stats[name]['mean'],(12,),name+' mean')
        if np.any(vector(stats[name]['std'],(12,),name+' std')<=0): raise ValueError('Invalid scale')
    lo=vector(c['joint_lower_rad'],(12,),'lower limits')
    hi=vector(c['joint_upper_rad'],(12,),'upper limits')
    if np.any(lo>=hi): raise ValueError('Invalid joint limits')
    for cam in c['cameras']:
        if c['image_sizes_wh'][cam]!=[640,480]:
            raise ValueError('Expected recorded 640x480 images; no implicit resizing')
    return names,lo,hi


class BoundedCommands:
    """Transparent joint-range projection and per-step slew limiting on ALL axes.

    This is explicitly constrained learned control. Every raw and applied action
    is recorded. No old trajectory, object pose, target waypoint, or phase enters
    the filter. Gross out-of-range requests stop instead of being silently fixed.
    """
    def __init__(self, previous, low, high, enabled=True):
        self.previous=vector(previous,(12,),'previous command').copy()
        self.low=vector(low,(12,),'lower limits');self.high=vector(high,(12,),'upper limits')
        self.enabled=bool(enabled);self.steps=0;self.modified_steps=0;self.clipped_values=0;self.max_change=0.
    def apply(self, requested):
        raw=vector(requested,(12,),'model command')
        if np.any(raw<self.low-.10) or np.any(raw>self.high+.10):
            raise RuntimeError('Model target more than 0.10 rad outside a joint range; no command applied')
        bounded=np.clip(raw,self.low,self.high)
        if not self.enabled:
            if np.any(bounded!=raw) or np.any(np.abs(raw-self.previous)>COMMAND_RATE*DT*1.06+1e-8):
                raise RuntimeError('Raw policy command violates retained range/slew constraint')
            applied=raw.copy()
        else:
            applied=self.previous+np.clip(bounded-self.previous,-COMMAND_RATE*DT,COMMAND_RATE*DT)
        err=float(np.max(np.abs(applied-raw)))
        self.steps+=1;self.modified_steps+=int(err>1e-8)
        self.clipped_values+=int(np.count_nonzero(bounded!=raw));self.max_change=max(self.max_change,err)
        self.previous=applied.copy()
        return applied
    def report(self):
        return dict(enabled=self.enabled,mode='joint bounds plus 0.35 rad/s slew limiter; no IK',
            commanded_steps=self.steps,modified_steps=self.modified_steps,
            modified_step_fraction=self.modified_steps/self.steps if self.steps else 0.,
            range_clipped_values=self.clipped_values,maximum_raw_applied_difference_rad=self.max_change,
            raw_unfiltered_policy_task_success_established=False)


class CupEvaluator:
    """Independent privileged-state scorer. Never provides waypoints or corrections.

    Thresholds follow the scripted cup engineering test: 25 mm lift; 8 mm new
    placement error; .8 contact/support fraction; 0.5 s lift/final windows. The
    0.25 s destination hold follows the source's transfer assessment. This is NOT
    an official competition scorer. Gripper clearance 40 mm is an added,
    explicitly reported release/retreat test for this rollout.
    """
    def __init__(self,initial):
        self.initial=vector(initial,(3,),'initial cup').copy()
        self.target=self.initial+np.array([.025,0,0])
        self.history=deque(maxlen=150)
        self.flags=dict(grasp_verified=False,lift_verified=False,transfer_verified=False,
                        placement_verified=False,release_verified=False,retreat_verified=False)
        self.times={};self.measurements={};self.last_t=None
    def window(self,seconds):
        n=int(round(seconds/DT))+1
        if len(self.history)<n:return None
        a=list(self.history)[-n:]
        if a[-1]['time_s']-a[0]['time_s']<seconds-1e-8:return None
        return a
    def _set(self,flag,t,measurement):
        self.flags[flag]=True;self.times[flag]=t;self.measurements[flag]=measurement
    def observe(self,row):
        t=float(row['time_s'])
        if self.last_t is not None and not math.isclose(t-self.last_t,DT,abs_tol=1e-8):
            raise ValueError('Noncontiguous evaluator samples')
        self.last_t=t;self.history.append(dict(row))
        if not self.flags['grasp_verified']:
            w=self.window(.1)
            if w and all(r['fixed_normal_n']>=CONTACT_N and r['moving_normal_n']>=CONTACT_N for r in w):
                self._set('grasp_verified',t,{'sustained_two_jaw_contact_s':.1})
        if self.flags['grasp_verified'] and not self.flags['lift_verified']:
            w=self.window(.5)
            if w and w[0]['time_s']>=self.times['grasp_verified']:
                rise=min(r['cup_z']-self.initial[2] for r in w)
                frac=sum(r['two_jaw_contact'] for r in w)/len(w);support=max(r['table_normal_n'] for r in w)
                if rise>=.025 and frac>=.8 and support<CONTACT_N:
                    self._set('lift_verified',t,dict(minimum_rise_m=rise,contact_fraction=frac,maximum_table_normal_n=support))
        if self.flags['lift_verified'] and not self.flags['transfer_verified']:
            w=self.window(.25)
            if w and w[0]['time_s']>=self.times['lift_verified']:
                error=max(np.linalg.norm(np.array([r['cup_x'],r['cup_y']])-self.target[:2]) for r in w)
                rise=min(r['cup_z']-self.initial[2] for r in w)
                frac=sum(r['two_jaw_contact'] for r in w)/len(w);support=max(r['table_normal_n'] for r in w)
                if error<=.008 and rise>=.025 and frac>=.8 and support<CONTACT_N:
                    self._set('transfer_verified',t,dict(maximum_horizontal_error_m=float(error),minimum_rise_m=rise))
        if self.flags['transfer_verified'] and not self.flags['placement_verified']:
            w=self.window(.5)
            if w and w[0]['time_s']>=self.times['transfer_verified']:
                error=max(np.linalg.norm(np.array([r['cup_x'],r['cup_y'],r['cup_z']])-self.target) for r in w)
                dx=min(r['cup_x']-self.initial[0] for r in w)
                frac=sum(r['table_normal_n']>=CONTACT_N and r['fixed_normal_n']<CONTACT_N and r['moving_normal_n']<CONTACT_N for r in w)/len(w)
                if error<=.008 and dx>=.017 and frac>=.8:
                    m=dict(maximum_destination_error_m=float(error),supported_and_released_fraction=frac)
                    self._set('placement_verified',t,m);self._set('release_verified',t,m)
        if self.flags['release_verified']:
            w=self.window(.5)
            # Require the cup to remain correctly supported after release while hand retreats.
            if w and w[0]['time_s']>=self.times['release_verified']:
                valid=all(np.linalg.norm(np.array([r['cup_x'],r['cup_y'],r['cup_z']])-self.target)<=.008 for r in w)
                clear=min(r['jaw_above_cup_center_m'] for r in w)
                free=sum(r['table_normal_n']>=CONTACT_N and r['fixed_normal_n']<CONTACT_N and r['moving_normal_n']<CONTACT_N for r in w)/len(w)
                if valid and clear>=.04 and free>=.8:
                    self._set('retreat_verified',t,dict(minimum_fingertip_height_above_cup_center_m=clear))
        return all(self.flags.values())
    def report(self):
        return {**self.flags,'milestone_times_s':self.times,'milestone_measurements':self.measurements,
            'destination_position_m':self.target.tolist(),'initial_position_m':self.initial.tolist(),
            'source':'simulator ground truth for evaluation/stop only; not a camera verifier',
            'official_challenge_score':None}
