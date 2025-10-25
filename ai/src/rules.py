from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time

@dataclass
class Telemetry:
    t: float
    speed_mps: float
    speed_limit_mps: float
    throttle: float
    brake: float
    steer_deg: float
    lane_offset_m: Optional[float]=None
    tl_state: Optional[str]=None
    in_stop_zone: Optional[bool]=None
    collision: bool=False

@dataclass
class CuesConfig:
    # thresholds with hysteresis
    speed_margin_warn_mps: float = 2.0
    speed_margin_clear_mps: float = 1.0
    lane_offset_warn_m: float   = 0.35
    lane_offset_clear_m: float  = 0.25
    ttc_warn_s: float           = 1.6
    ttc_clear_s: float          = 1.9

    harsh_brake_thresh: float   = 0.35
    cue_cooldown_s: float       = 1.5

    # display behavior
    min_display_s: float        = 1.6
    sustain_after_clear_s: float= 0.8
    max_concurrent_cues: int    = 2
    level_ema_alpha: float      = 0.65

@dataclass
class ScoreWeights:
    speeding: float=0.25; lane: float=0.25; headway: float=0.20; smooth: float=0.15; compliance: float=0.15

@dataclass
class ScoringState:
    cfg: CuesConfig=field(default_factory=CuesConfig)
    weights: ScoreWeights=field(default_factory=ScoreWeights)

    # tallies
    total_time: float=0.0; over_speed_time: float=0.0; out_lane_time: float=0.0; ttc_bad_time: float=0.0
    harsh_events: int=0; red_violations: int=0; collisions: int=0

    # prevs
    last_t: Optional[float]=None; last_brake: Optional[float]=None

    # cue mgmt
    last_emit_ts: Dict[str,float]=field(default_factory=dict)
    active_cues: Dict[str,Dict[str,float]]=field(default_factory=dict)  # name -> {level, until}

    def step(self, tel: Telemetry, lead_ttc_s: Optional[float]) -> List[Dict[str,Any]]:
        inst: List[Dict[str,Any]]=[]
        if self.last_t is None:
            self.last_t=tel.t; self.last_brake=tel.brake
            return inst

        dt=max(0.0, tel.t-self.last_t); self.total_time+=dt

        # 1) Speeding (hysteresis)
        over_warn = tel.speed_mps > tel.speed_limit_mps + self.cfg.speed_margin_warn_mps
        over_clear= tel.speed_mps > tel.speed_limit_mps + self.cfg.speed_margin_clear_mps
        if over_warn:
            self.over_speed_time+=dt
            self._activate_cue("SLOW_DOWN", level=min(1.0,(tel.speed_mps - (tel.speed_limit_mps+self.cfg.speed_margin_warn_mps))/5.0))
        else:
            if over_clear:
                self._extend_if_active("SLOW_DOWN", self.cfg.sustain_after_clear_s)

        # 2) Lane keeping (hysteresis)
        if tel.lane_offset_m is not None:
            off = abs(tel.lane_offset_m)
            if off > self.cfg.lane_offset_warn_m:
                self.out_lane_time+=dt
                self._activate_cue("KEEP_LANE", level=min(1.0,(off - self.cfg.lane_offset_warn_m)/0.5))
            elif off > self.cfg.lane_offset_clear_m:
                self._extend_if_active("KEEP_LANE", self.cfg.sustain_after_clear_s)

        # 3) Headway / TTC (hysteresis)
        if lead_ttc_s is not None:
            if lead_ttc_s < self.cfg.ttc_warn_s:
                self.ttc_bad_time+=dt
                lvl=max(0.3,min(1.0,(self.cfg.ttc_warn_s-lead_ttc_s)/self.cfg.ttc_warn_s))
                self._activate_cue("INCREASE_HEADWAY", level=lvl)
            elif lead_ttc_s < self.cfg.ttc_clear_s:
                self._extend_if_active("INCREASE_HEADWAY", self.cfg.sustain_after_clear_s)

        # 4) Harsh braking
        if self.last_brake is not None and dt>0:
            db=tel.brake-self.last_brake
            if db/max(dt,1e-3) > self.cfg.harsh_brake_thresh*10:
                self.harsh_events+=1
                self._activate_cue("SMOOTHER_BRAKE", level=min(1.0, db))

        # 5) Compliance
        if tel.in_stop_zone and tel.tl_state=="red" and tel.speed_mps>0.5:
            self.red_violations+=1
            self._activate_cue("BRAKE_NOW", level=1.0)

        if tel.collision: self.collisions+=1

        self.last_t=tel.t; self.last_brake=tel.brake
        self._prune_expired()
        return inst

    def get_display_cues(self) -> List[Dict[str,Any]]:
        now=time.time()
        self._prune_expired()
        items=[{"cue":k,"level":v["level"],"t_emit":now} for k,v in self.active_cues.items() if v["until"]>now]
        items.sort(key=lambda x: x["level"], reverse=True)
        return items[:self.cfg.max_concurrent_cues]

    def finalize(self)->Dict[str,Any]:
        eps=1e-6
        speeding_pen=min(25.0, 100.0*(self.over_speed_time/max(self.total_time,eps)))
        lane_pen=min(25.0, 100.0*(self.out_lane_time/max(self.total_time,eps)))
        headway_pen=min(25.0, 100.0*(self.ttc_bad_time/max(self.total_time,eps)))
        smooth_pen=min(25.0, self.harsh_events*5.0)
        compliance_pen=min(25.0, self.red_violations*10.0 + self.collisions*10.0)
        subs={"speeding":max(0,100-speeding_pen),"lane":max(0,100-lane_pen),
              "headway":max(0,100-headway_pen),"smooth":max(0,100-smooth_pen),
              "compliance":max(0,100-compliance_pen)}
        final=(subs["speeding"]*self.weights.speeding + subs["lane"]*self.weights.lane +
               subs["headway"]*self.weights.headway + subs["smooth"]*self.weights.smooth +
               subs["compliance"]*self.weights.compliance)
        return {"subscores":subs,"final":round(final,1),
                "violations":{"red_light":self.red_violations,"collisions":self.collisions}}

    # ---- cue helpers ----
    def _activate_cue(self, name:str, level:float):
        now=time.time()
        last=self.last_emit_ts.get(name, 0.0)
        if now-last < self.cfg.cue_cooldown_s and name not in self.active_cues:
            return
        self.last_emit_ts[name]=now
        level=float(max(0.0,min(1.0,level)))
        if name in self.active_cues:
            a=self.cfg.level_ema_alpha
            old=self.active_cues[name]["level"]
            self.active_cues[name]["level"]=a*level + (1-a)*old
            self.active_cues[name]["until"]=max(self.active_cues[name]["until"], now + self.cfg.min_display_s)
        else:
            self.active_cues[name]={"level":level, "until": now + self.cfg.min_display_s}

    def _extend_if_active(self, name:str, extra_s:float):
        now=time.time()
        if name in self.active_cues:
            self.active_cues[name]["until"]=max(self.active_cues[name]["until"], now + extra_s)

    def _prune_expired(self):
        now=time.time()
        for k in [k for k,v in self.active_cues.items() if v["until"]<=now]:
            del self.active_cues[k]
