"""Generate a self-contained local HTML analysis dashboard."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any
import json
import math
import sqlite3

from .analytics import build_fitness_analytics_set
from .audit import build_audit

METERS_PER_MILE = 1609.344


def _json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _segment_halves(connection: sqlite3.Connection) -> dict[int, dict[str, float | None]]:
    rows = connection.execute(
        """
        SELECT activity_id,distance_m,moving_time_s,average_hr_bpm,distance_into_run_m
        FROM segments
        WHERE is_pathological=0 AND distance_m > 0 AND moving_time_s > 0
        ORDER BY activity_id,segment_index
        """
    ).fetchall()
    grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[int(row["activity_id"])].append(row)
    result: dict[int, dict[str, float | None]] = {}
    for activity_id, segments in grouped.items():
        total_distance = sum(float(row["distance_m"]) for row in segments)
        halves = []
        for first_half in (True, False):
            selected = [
                row
                for row in segments
                if ((float(row["distance_into_run_m"]) - float(row["distance_m"]) / 2) <= total_distance / 2)
                == first_half
            ]
            distance = sum(float(row["distance_m"]) for row in selected)
            moving_time = sum(float(row["moving_time_s"]) for row in selected)
            hr_rows = [row for row in selected if row["average_hr_bpm"] is not None]
            hr_time = sum(float(row["moving_time_s"]) for row in hr_rows)
            average_hr = (
                sum(float(row["average_hr_bpm"]) * float(row["moving_time_s"]) for row in hr_rows)
                / hr_time
                if hr_time
                else None
            )
            pace = moving_time / 60 / (distance / METERS_PER_MILE) if distance else None
            speed_mph = (distance / METERS_PER_MILE) / (moving_time / 3600) if moving_time else None
            halves.append((pace, average_hr, speed_mph))
        first, second = halves
        decoupling = None
        if first[1] and second[1] and first[2] and second[2]:
            first_efficiency = first[2] / first[1]
            second_efficiency = second[2] / second[1]
            decoupling = (first_efficiency - second_efficiency) / first_efficiency * 100
        result[activity_id] = {
            "first_half_pace": first[0],
            "second_half_pace": second[0],
            "first_half_hr": first[1],
            "second_half_hr": second[1],
            "decoupling_percent": decoupling,
        }
    return result


def _window_average(scored: list[dict], index: int, days: int) -> float | None:
    end = datetime.fromisoformat(scored[index]["start_time_utc"])
    start = end - timedelta(days=days)
    values = [
        item["standardized_pace"]
        for item in scored[: index + 1]
        if datetime.fromisoformat(item["start_time_utc"]) >= start
        and item["standardized_pace"] is not None
    ]
    return mean(values) if values else None


def _prior_value(scored: list[dict], index: int, values: list[float | None], days: int) -> float | None:
    target = datetime.fromisoformat(scored[index]["start_time_utc"]) - timedelta(days=days)
    candidate = None
    for prior_index in range(index):
        if datetime.fromisoformat(scored[prior_index]["start_time_utc"]) <= target:
            candidate = values[prior_index]
        else:
            break
    return candidate


def _add_trends(runs: list[dict]) -> None:
    scored = [row for row in runs if row["standardized_pace"] is not None]
    scored.sort(key=lambda row: row["start_time_utc"])
    seven = [_window_average(scored, index, 7) for index in range(len(scored))]
    twenty_eight = [_window_average(scored, index, 28) for index in range(len(scored))]
    for index, row in enumerate(scored):
        row["trend_7d"] = seven[index]
        row["trend_28d"] = twenty_eight[index]
        prior_28 = _prior_value(scored, index, twenty_eight, 28)
        prior_90 = _prior_value(scored, index, twenty_eight, 90)
        row["change_28d"] = twenty_eight[index] - prior_28 if prior_28 is not None else None
        row["change_90d"] = twenty_eight[index] - prior_90 if prior_90 is not None else None


def _load_runs(connection: sqlite3.Connection) -> list[dict]:
    halves = _segment_halves(connection)
    rows = connection.execute(
        """
        SELECT a.id,a.activity_id,a.start_time_utc,a.start_time_local,a.total_distance_m,
               a.gps_quality,a.hr_quality,a.distance_source,a.notes AS activity_notes,
               m.calculated_moving_time_s,m.moving_pace_min_mile,m.moving_average_hr_bpm,
               m.hr_zone_seconds_json,m.model_eligible,m.exclusion_reason,
               m.standardized_pace_at_target_hr_min_mile,m.standardized_pace_uncertainty_min_mile,
               m.raw_aerobic_efficiency_min_mile,m.environmental_adjustment_min_mile,
               m.selected_model_name,m.previous_7d_miles,m.previous_28d_miles,
               aw.temperature_f,aw.dewpoint_f,aw.relative_humidity_percent,aw.wind_speed_mph,
               o.workout_type,o.illness,o.notes AS override_notes,mr.result_json
        FROM activities a
        LEFT JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN activity_weather aw ON aw.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        LEFT JOIN model_runs mr ON mr.activity_id=a.id AND mr.model_name='standardized_pace_at_target_hr'
        ORDER BY a.start_time_utc_epoch
        """
    ).fetchall()
    output = []
    for row in rows:
        model_result = _json(row["result_json"], {})
        zones = _json(row["hr_zone_seconds_json"], {})
        item = {
            "database_id": int(row["id"]),
            "activity_id": row["activity_id"],
            "date": str(row["start_time_local"] or row["start_time_utc"])[:10],
            "start_time_utc": row["start_time_utc"],
            "distance_miles": _finite(row["total_distance_m"] / METERS_PER_MILE) if row["total_distance_m"] else None,
            "moving_minutes": _finite(row["calculated_moving_time_s"] / 60) if row["calculated_moving_time_s"] else None,
            "moving_pace": _finite(row["moving_pace_min_mile"]),
            "average_hr": _finite(row["moving_average_hr_bpm"]),
            "zones_minutes": {name: round(float(seconds) / 60, 1) for name, seconds in zones.items()},
            "temperature_f": _finite(row["temperature_f"]),
            "dewpoint_f": _finite(row["dewpoint_f"]),
            "humidity_percent": _finite(row["relative_humidity_percent"]),
            "wind_mph": _finite(row["wind_speed_mph"]),
            "standardized_pace": _finite(row["standardized_pace_at_target_hr_min_mile"]),
            "uncertainty_95": _finite(row["standardized_pace_uncertainty_min_mile"]),
            "raw_efficiency": _finite(row["raw_aerobic_efficiency_min_mile"]),
            "environmental_adjustment": _finite(row["environmental_adjustment_min_mile"]),
            "selected_model": row["selected_model_name"],
            "model_eligible": bool(row["model_eligible"]),
            "exclusion_reason": row["exclusion_reason"],
            "gps_quality": row["gps_quality"],
            "hr_quality": row["hr_quality"],
            "distance_source": row["distance_source"],
            "workout_type": row["workout_type"] or "run",
            "illness": bool(row["illness"]),
            "notes": row["override_notes"] or row["activity_notes"],
            "previous_7d_miles": _finite(row["previous_7d_miles"]),
            "previous_28d_miles": _finite(row["previous_28d_miles"]),
            "contributions": model_result.get("contributions_min_mile", {}),
            "adjustment_evidence": model_result.get("adjustment_evidence", {}),
            "raw_pace_at_target_hr": _finite(model_result.get("raw_pace_at_target_hr_min_mile")),
            "measurement_uncertainty_95": _finite(
                model_result.get("measurement_uncertainty_95_min_mile")
            ),
            "heat_coefficient_uncertainty_95": _finite(
                model_result.get("heat_coefficient_uncertainty_95_min_mile")
            ),
            "observed_segment_pace": model_result.get("observed_segment_pace_min_mile"),
        }
        item.update(halves.get(int(row["id"]), {}))
        output.append(item)
    _add_trends(output)
    return output


def build_report_data(connection: sqlite3.Connection, config: dict) -> dict:
    runs = _load_runs(connection)
    fitness_analytics = build_fitness_analytics_set(
        runs, target_hr_bpm=float(config["target_hr"])
    )
    metadata_row = connection.execute(
        "SELECT metadata_json,fitted_at_utc FROM model_metadata "
        "WHERE model_name='standardized_pace_at_target_hr' ORDER BY fitted_at_utc DESC LIMIT 1"
    ).fetchone()
    metadata = _json(metadata_row["metadata_json"], {}) if metadata_row else {}
    scored = [run for run in runs if run["standardized_pace"] is not None]
    latest = scored[-1] if scored else None
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target_hr": config["target_hr"],
        "reference_conditions": config["reference_conditions"],
        "audit": build_audit(connection),
        "model": metadata,
        "fitness_analytics": fitness_analytics,
        "runs": runs,
        "summary": {
            "activities": len(runs),
            "eligible_runs": sum(run["model_eligible"] for run in runs),
            "scored_runs": len(scored),
            "latest_standardized_pace": latest["standardized_pace"] if latest else None,
            "latest_trend_28d": latest.get("trend_28d") if latest else None,
            "latest_change_90d": latest.get("change_90d") if latest else None,
        },
    }


def write_report(connection: sqlite3.Connection, config: dict, path: str | Path) -> Path:
    data = build_report_data(connection, config)
    safe_data = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = _REPORT_TEMPLATE.replace("__REPORT_DATA__", safe_data)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


_REPORT_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Garmin Aerobic Performance</title>
<style>
:root{--bg:#091014;--panel:#111b20;--panel2:#16242a;--text:#eef6f3;--muted:#91a8a6;--line:#294047;--cyan:#56d6c9;--orange:#ffb45b;--lime:#a7d96f;--red:#ff7b72;--blue:#70a7ff;--shadow:0 20px 50px #0005}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#17303a 0,transparent 30%),var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}main{max-width:1500px;margin:auto;padding:34px 28px 70px}h1{font-size:clamp(32px,5vw,66px);line-height:.96;letter-spacing:-.055em;margin:12px 0 16px;max-width:850px}h2{font-size:21px;letter-spacing:-.02em;margin:0 0 4px}p{color:var(--muted)}.eyebrow{text-transform:uppercase;letter-spacing:.16em;color:var(--cyan);font-weight:750;font-size:11px}.hero{display:grid;grid-template-columns:1.5fr .8fr;gap:22px;align-items:end;margin-bottom:24px}.hero-copy{max-width:720px}.model-note{border:1px solid #6b552a;background:#251f14;padding:18px;border-radius:16px;color:#f7ddb7}.model-note.good{border-color:#28645d;background:#102923}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0 34px}.card,.panel{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}.card{padding:17px}.card .value{font-size:25px;font-weight:760;letter-spacing:-.04em}.card .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.1em}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:16px 0}.panel{padding:20px;min-width:0}.panel.wide{grid-column:1/-1}.panel-head{display:flex;justify-content:space-between;gap:18px;align-items:start}.tag{background:#1d353b;color:var(--cyan);padding:4px 9px;border-radius:999px;font-size:11px;white-space:nowrap}.chart{position:relative;height:330px;margin-top:12px}.chart svg{width:100%;height:100%;overflow:visible}.axis{stroke:#47616a;stroke-width:1}.gridline{stroke:#24383e;stroke-width:1}.tick{fill:#7f9898;font-size:10px}.legend{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:12px;margin-top:10px}.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.tooltip{position:fixed;pointer-events:none;display:none;z-index:20;background:#061014f2;border:1px solid #41616a;border-radius:10px;padding:9px 11px;box-shadow:var(--shadow);font-size:12px;max-width:260px}.warning{color:#ffd18c}.cv-table{width:100%;border-collapse:collapse;margin-top:14px}.cv-table th,.cv-table td{text-align:right;border-bottom:1px solid var(--line);padding:10px}.cv-table th:first-child,.cv-table td:first-child{text-align:left}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}.controls input,.controls select,.time-select{background:#0b1519;border:1px solid var(--line);color:var(--text);padding:10px 12px;border-radius:10px}.controls input{min-width:280px}.table-wrap{overflow:auto;max-height:720px;border:1px solid var(--line);border-radius:13px}table.runs{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}table.runs th{position:sticky;top:0;background:#17262c;z-index:2;text-align:left;color:#c6d8d5;cursor:pointer;white-space:nowrap}table.runs th,table.runs td{padding:10px 12px;border-bottom:1px solid #21353b;white-space:nowrap}table.runs tbody tr:hover{background:#193039;cursor:pointer}.pill{display:inline-block;border-radius:999px;padding:2px 7px;background:#223940;color:#b9cdca}.pill.ok{background:#173b35;color:#84e2d4}.pill.no{background:#39241f;color:#ffb3a9}.pace{font-variant-numeric:tabular-nums;font-weight:680}.method{font-size:12px;color:var(--muted);border-left:2px solid var(--cyan);padding-left:12px}.modal{position:fixed;inset:0;background:#000a;z-index:30;display:none;align-items:center;justify-content:center;padding:22px}.modal.open{display:flex}.modal-card{background:#101c21;border:1px solid #36525a;border-radius:18px;padding:24px;max-width:760px;width:100%;max-height:90vh;overflow:auto;box-shadow:var(--shadow)}.modal-head{display:flex;justify-content:space-between;gap:20px}.close{background:#263b42;color:white;border:0;border-radius:8px;padding:7px 10px;cursor:pointer}.detail-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.detail{padding:10px;background:#15262c;border-radius:10px}.detail b{display:block;font-size:16px}.steps{border-left:1px solid #3c5961;margin-left:8px;padding-left:20px}.step{padding:6px 0}.footer{margin-top:28px;color:#758c8b;font-size:12px}.fitness-readout{display:grid;grid-template-columns:1.3fr repeat(4,minmax(120px,.7fr));gap:12px;margin-top:16px}.fitness-story{font-size:17px;color:var(--text);padding:14px;background:#10252a;border-radius:12px}.metric{padding:12px;background:#14252b;border-radius:12px}.metric span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.metric b{display:block;font-size:20px;margin-top:4px}
@media(max-width:900px){.hero,.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}.panel.wide{grid-column:auto}.fitness-readout{grid-template-columns:1fr 1fr}}@media(max-width:560px){main{padding:24px 14px}.cards{grid-template-columns:1fr 1fr}.detail-grid,.fitness-readout{grid-template-columns:1fr}.controls input{min-width:100%;width:100%}}
</style>
</head>
<body><main>
<section class="hero"><div><div class="eyebrow">Prior-anchored corrections · personal fitness series</div><h1>Aerobic performance without fitting weather to your seasons.</h1><p class="hero-copy">Each run is normalized independently to <span id="target-copy"></span> bpm. Published evidence anchors environmental effects; calendar-local matched runs may update them without letting the multi-year fitness trend choose their weights.</p></div><div id="model-note" class="model-note"></div></section>
<section id="cards" class="cards"></section>
<section class="grid">
  <article class="panel wide"><div class="panel-head"><div><h2>What this says about your fitness</h2><p>The time horizon changes the balance between responsiveness and stability.</p></div><select id="fitness-window" class="time-select" aria-label="Fitness analysis time frame"></select></div><div id="fitness-readout" class="fitness-readout"></div></article>
  <article class="panel wide"><div class="panel-head"><div><h2>Fitness trend</h2><p>Independent per-run estimates plus the selected robust trailing fitness level.</p></div><span id="fitness-window-tag" class="tag">Primary metric</span></div><div id="fitness" class="chart"></div><div id="fitness-legend" class="legend"></div></article>
  <article class="panel"><div class="panel-head"><div><h2>Raw <span class="target-hr-copy"></span> → standardized <span class="target-hr-copy"></span></h2><p>Each bridge is the total grade, temperature, dew-point, wind, and drift adjustment. Hover for the audit trail.</p></div></div><div id="comparison" class="chart"></div><div id="comparison-legend" class="legend"></div></article>
  <article class="panel"><div class="panel-head"><div><h2>Raw cardiovascular efficiency</h2><p>Pace normalized to <span class="target-hr-copy"></span> before grade or weather adjustments.</p></div></div><div id="efficiency" class="chart"></div><div id="efficiency-legend" class="legend"></div></article>
  <article class="panel wide"><div class="panel-head"><div><h2>Published heat reference</h2><p id="heat-subtitle"></p></div><span class="tag">Population prior</span></div><div id="heat" class="chart"></div><div id="heat-legend" class="legend"></div></article>
  <article class="panel"><div class="panel-head"><div><h2>Cardiac drift</h2><p>Approximate first-half vs second-half speed/HR decoupling from valid segments.</p></div></div><div id="drift" class="chart"></div><div id="drift-legend" class="legend"></div></article>
  <article class="panel"><div class="panel-head"><div><h2>Weather timeline</h2><p>Interpolated run-time temperature and dew point.</p></div></div><div id="weather" class="chart"></div><div id="weather-legend" class="legend"></div></article>
  <article class="panel wide"><div class="panel-head"><div><h2>Method diagnostics</h2><p>Literature priors, matched personal evidence, within-run HR calibration, retained windows, and explicit limitations.</p></div><span class="tag">No seasonal coefficient fitting</span></div><div class="grid"><div><div id="cv" class="chart"></div><div id="cv-legend" class="legend"></div></div><div id="cv-details"></div></div></article>
  <article class="panel wide"><div class="panel-head"><div><h2>Run table</h2><p>All activities are preserved. Click a row for its score decomposition and HR-zone detail.</p></div><span id="table-count" class="tag"></span></div><div class="controls"><input id="search" placeholder="Search date, type, notes, or exclusion"><select id="eligibility"><option value="all">All activities</option><option value="eligible">Model eligible</option><option value="excluded">Excluded</option><option value="scored">Scored</option></select></div><div class="table-wrap"><table class="runs"><thead><tr id="run-head"></tr></thead><tbody id="run-body"></tbody></table></div></article>
</section>
<p class="method">The standardized score is a per-run aerobic-efficiency estimate, not VO₂ max or a judgment about whether a pace is “fast.” Grade uses the published Minetti transform where altitude exists. Heat begins with a literature prior and is updated only by calendar-local hot/cool matches; the displayed evidence level states how much those personal data influence the result. Missing elevation is retained and visibly left unadjusted.</p>
<div class="footer" id="generated"></div>
</main><div id="tooltip" class="tooltip"></div><div id="modal" class="modal"><div class="modal-card"><div class="modal-head"><div><div class="eyebrow">Run diagnostic</div><h2 id="modal-title"></h2></div><button class="close" onclick="closeModal()">Close</button></div><div id="modal-body"></div></div></div>
<script>const DATA=__REPORT_DATA__;</script>
<script>
const C={cyan:'#56d6c9',orange:'#ffb45b',lime:'#a7d96f',red:'#ff7b72',blue:'#70a7ff',muted:'#91a8a6'};
const tooltip=document.getElementById('tooltip');
const pace=v=>{if(v==null)return'—';const seconds=Math.round(v*60);return`${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`};
const num=(v,d=1)=>v==null?'—':Number(v).toFixed(d);
const signed=(v,d=1)=>v==null?'—':`${v>0?'+':''}${Number(v).toFixed(d)}`;
const dateLabel=s=>new Date(s).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'});
function showTip(e,html){tooltip.innerHTML=html;tooltip.style.display='block';tooltip.style.left=Math.min(innerWidth-275,e.clientX+14)+'px';tooltip.style.top=Math.max(8,e.clientY-30)+'px'}
function hideTip(){tooltip.style.display='none'}
function svgEl(name,attrs={}){const e=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e}
function legend(id,series){document.getElementById(id).innerHTML=series.map(s=>`<span><i class="swatch" style="background:${s.color}"></i>${s.name}</span>`).join('')}
function lineChart(id,series,opt={}){
 const root=document.getElementById(id),W=760,H=310,p={l:55,r:18,t:18,b:35};root.innerHTML='';
 const all=series.flatMap(s=>s.data).filter(d=>d.y!=null&&Number.isFinite(d.y));if(!all.length){root.innerHTML='<p>No supported data.</p>';return}
 const xs=all.map(d=>+new Date(d.x)), ys=all.flatMap(d=>[d.y,d.low,d.high].filter(v=>v!=null));let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmin===xmax)xmax+=86400000;if(ymin===ymax){ymin-=1;ymax+=1}const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
 const sx=x=>p.l+(+new Date(x)-xmin)/(xmax-xmin)*(W-p.l-p.r),sy=y=>p.t+(y-ymin)/(ymax-ymin)*(H-p.t-p.b);
 const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});root.appendChild(svg);
 for(let i=0;i<5;i++){const y=ymin+(ymax-ymin)*i/4,py=sy(y);svg.appendChild(svgEl('line',{x1:p.l,y1:py,x2:W-p.r,y2:py,class:'gridline'}));const t=svgEl('text',{x:p.l-8,y:py+4,'text-anchor':'end',class:'tick'});t.textContent=opt.pace?pace(y):(opt.seconds?`${Math.round(y)}s`:num(y));svg.appendChild(t)}
 [0,.5,1].forEach(fr=>{const x=xmin+(xmax-xmin)*fr,t=svgEl('text',{x:p.l+(W-p.l-p.r)*fr,y:H-10,'text-anchor':fr===0?'start':fr===1?'end':'middle',class:'tick'});t.textContent=new Date(x).toLocaleDateString(undefined,{year:'numeric',month:'short'});svg.appendChild(t)});
 series.forEach(s=>{const pts=s.data.filter(d=>d.y!=null).sort((a,b)=>+new Date(a.x)-+new Date(b.x));if(s.line!==false){const path=svgEl('path',{d:pts.map((d,i)=>`${i?'L':'M'}${sx(d.x)},${sy(d.y)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':s.width||2,'stroke-dasharray':s.dash||''});svg.appendChild(path)}pts.forEach(d=>{if(d.low!=null&&d.high!=null){svg.appendChild(svgEl('line',{x1:sx(d.x),x2:sx(d.x),y1:sy(d.low),y2:sy(d.high),stroke:s.color,'stroke-opacity':.32}))}const c=svgEl('circle',{cx:sx(d.x),cy:sy(d.y),r:s.radius||3,fill:s.color,'fill-opacity':s.opacity||.85,tabindex:0});c.addEventListener('mousemove',e=>showTip(e,d.tip||`${dateLabel(d.x)}<br>${opt.pace?pace(d.y):num(d.y)}`));c.addEventListener('mouseleave',hideTip);svg.appendChild(c)})});
}
function adjustmentTip(r){const c=r.contributions||{},e=r.adjustment_evidence||{},conf=k=>(e[k]||{}).confidence||'unavailable',pct=k=>(e[k]||{}).personal_data_weight==null?'':` · ${Math.round(e[k].personal_data_weight*100)}% personal evidence`;return `${dateLabel(r.start_time_utc)}<br><b>${pace(r.raw_pace_at_target_hr)}/mi raw @${DATA.target_hr}</b><br>${signed(r.environmental_adjustment*60,0)} sec/mi environmental<br><b>${pace(r.standardized_pace)}/mi standardized</b><hr>Grade ${signed((c.grade_adjustment||0)*60,0)}s · ${conf('grade')}<br>Temperature ${signed((c.temperature_adjustment||0)*60,0)}s · ${conf('temperature')}${pct('temperature')}<br>Dew point ${signed((c.dewpoint_adjustment||0)*60,0)}s · ${conf('dew_point')}<br>Wind ${signed((c.wind_adjustment||0)*60,0)}s · ${conf('wind')}<br>Drift ${signed((c.drift_adjustment||0)*60,0)}s · ${conf('drift')}`}
function bridgeChart(){const runs=DATA.runs.filter(r=>r.raw_pace_at_target_hr!=null&&r.standardized_pace!=null),root=document.getElementById('comparison'),W=760,H=310,p={l:55,r:18,t:18,b:35};root.innerHTML='';if(!runs.length){root.innerHTML='<p>No supported data.</p>';return}const xs=runs.map(r=>+new Date(r.start_time_utc)),ys=runs.flatMap(r=>[r.raw_pace_at_target_hr,r.standardized_pace]),xmin=Math.min(...xs),rawXmax=Math.max(...xs),xmax=rawXmax===xmin?rawXmax+86400000:rawXmax,ymin=Math.min(...ys)-.15,ymax=Math.max(...ys)+.15,sx=x=>p.l+(+new Date(x)-xmin)/(xmax-xmin)*(W-p.l-p.r),sy=y=>p.t+(y-ymin)/(ymax-ymin)*(H-p.t-p.b),svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});root.appendChild(svg);for(let i=0;i<5;i++){const y=ymin+(ymax-ymin)*i/4,py=sy(y);svg.appendChild(svgEl('line',{x1:p.l,y1:py,x2:W-p.r,y2:py,class:'gridline'}));const t=svgEl('text',{x:p.l-8,y:py+4,'text-anchor':'end',class:'tick'});t.textContent=pace(y);svg.appendChild(t)}[0,.5,1].forEach(fr=>{const x=xmin+(xmax-xmin)*fr,t=svgEl('text',{x:p.l+(W-p.l-p.r)*fr,y:H-10,'text-anchor':fr===0?'start':fr===1?'end':'middle',class:'tick'});t.textContent=new Date(x).toLocaleDateString(undefined,{year:'numeric',month:'short'});svg.appendChild(t)});runs.forEach(r=>{const x=sx(r.start_time_utc),tip=adjustmentTip(r),line=svgEl('line',{x1:x,x2:x,y1:sy(r.raw_pace_at_target_hr),y2:sy(r.standardized_pace),stroke:r.environmental_adjustment<=0?C.lime:C.red,'stroke-width':2,'stroke-opacity':.55});line.addEventListener('mousemove',e=>showTip(e,tip));line.addEventListener('mouseleave',hideTip);svg.appendChild(line);[[r.raw_pace_at_target_hr,C.muted],[r.standardized_pace,C.cyan]].forEach(([value,color])=>{const dot=svgEl('circle',{cx:x,cy:sy(value),r:3.5,fill:color});dot.addEventListener('mousemove',e=>showTip(e,tip));dot.addEventListener('mouseleave',hideTip);svg.appendChild(dot)})});legend('comparison-legend',[{name:`Raw pace @${DATA.target_hr}`,color:C.muted},{name:'Standardized',color:C.cyan},{name:'Adjustment bridge',color:C.lime}])}
function heatChart(){
 const heat=DATA.model.heat_response||[],colors={45:C.cyan,60:C.orange,70:C.red},series=[45,60,70].map(dp=>({name:`${dp}°F dew point`,color:colors[dp],data:heat.filter(d=>d.dewpoint_f===dp)})),root=document.getElementById('heat'),W=760,H=310,p={l:55,r:18,t:18,b:35};root.innerHTML='';if(!heat.length){root.innerHTML='<p>No heat-response data.</p>';return}const ys=heat.map(d=>d.pace_penalty_seconds_per_mile),ymin=Math.min(...ys)-5,ymax=Math.max(...ys)+5,sx=t=>p.l+(t-55)/35*(W-p.l-p.r),sy=y=>p.t+(y-ymin)/(ymax-ymin)*(H-p.t-p.b),svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});root.appendChild(svg);
 for(let i=0;i<5;i++){const y=ymin+(ymax-ymin)*i/4,py=sy(y);svg.appendChild(svgEl('line',{x1:p.l,y1:py,x2:W-p.r,y2:py,class:'gridline'}));const t=svgEl('text',{x:p.l-8,y:py+4,'text-anchor':'end',class:'tick'});t.textContent=Math.round(y)+'s';svg.appendChild(t)}
 [55,60,65,70,75,80,85,90].forEach(temp=>{const t=svgEl('text',{x:sx(temp),y:H-10,'text-anchor':'middle',class:'tick'});t.textContent=temp+'°';svg.appendChild(t)});
 series.forEach(s=>{const path=svgEl('path',{d:s.data.map((d,i)=>`${i?'L':'M'}${sx(d.temperature_f)},${sy(d.pace_penalty_seconds_per_mile)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':2});svg.appendChild(path);s.data.forEach(d=>{const c=svgEl('circle',{cx:sx(d.temperature_f),cy:sy(d.pace_penalty_seconds_per_mile),r:4,fill:s.color});c.addEventListener('mousemove',e=>showTip(e,`${d.temperature_f}°F / ${d.dewpoint_f}°F dew point<br><b>${signed(d.pace_penalty_seconds_per_mile,1)} sec/mi</b><br>95% uncertainty ±${num(d.uncertainty_95_seconds_per_mile,1)} sec<br>${d.supporting_runs_near_conditions} nearby runs · ${d.coverage} coverage`));c.addEventListener('mouseleave',hideTip);svg.appendChild(c)})});legend('heat-legend',series)
}
function bars(){const cv=DATA.model.grouped_cross_validation||{},names=Object.keys(cv),root=document.getElementById('cv'),W=700,H=280,p={l:55,r:12,t:12,b:35};root.innerHTML='';if(!names.length)return;const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});root.appendChild(svg);const max=Math.max(...names.flatMap(n=>[cv[n].mae_seconds_per_mile,cv[n].rmse_seconds_per_mile]))*1.12,group=(W-p.l-p.r)/names.length,sy=v=>H-p.b-v/max*(H-p.t-p.b);[0,25,50,75].filter(v=>v<=max).forEach(v=>{svg.appendChild(svgEl('line',{x1:p.l,x2:W-p.r,y1:sy(v),y2:sy(v),class:'gridline'}));const t=svgEl('text',{x:p.l-7,y:sy(v)+4,'text-anchor':'end',class:'tick'});t.textContent=v+'s';svg.appendChild(t)});names.forEach((n,i)=>{[['mae_seconds_per_mile',C.cyan,-11],['rmse_seconds_per_mile',C.orange,11]].forEach(([key,color,off])=>{const v=cv[n][key],rect=svgEl('rect',{x:p.l+group*(i+.5)+off-9,y:sy(v),width:18,height:H-p.b-sy(v),rx:3,fill:color});rect.addEventListener('mousemove',e=>showTip(e,`Model ${n}<br>${key.startsWith('mae')?'MAE':'RMSE'}: <b>${num(v,1)} sec/mi</b>`));rect.addEventListener('mouseleave',hideTip);svg.appendChild(rect)});const t=svgEl('text',{x:p.l+group*(i+.5),y:H-12,'text-anchor':'middle',class:'tick'});t.textContent=n;svg.appendChild(t)});legend('cv-legend',[{name:'MAE',color:C.cyan},{name:'RMSE',color:C.orange}])}
function currentAnalysis(){const select=document.getElementById('fitness-window'),days=select&&select.value?select.value:String(DATA.fitness_analytics.default_window_days);return DATA.fitness_analytics.by_window[days]}
function initAnalysisControl(){const select=document.getElementById('fitness-window'),settings=DATA.fitness_analytics;select.innerHTML=settings.available_windows.map(days=>`<option value="${days}" ${days===settings.default_window_days?'selected':''}>${days} days</option>`).join('');select.onchange=()=>{initSummary();renderFitnessReadout();initCharts()}}
function renderFitnessReadout(){const a=currentAnalysis(),c=a.current,change=a.change_prior_window,best=a.best_sustained,status={improving:'Improving',declining:'Declining',stable_or_uncertain:'Stable / uncertain',insufficient_comparison:'Not enough prior data'}[a.status]||a.status,delta=change?signed(change.pace_change_seconds_per_mile,0)+' sec/mi':'—',prob=change?Math.round(change.probability_faster*100)+'%':'—',bestDate=best?dateLabel(best.as_of_utc):'—';document.getElementById('fitness-window-tag').textContent=`${a.window_days}-day level`;document.getElementById('fitness-readout').innerHTML=`<div class="fitness-story"><b>${status}.</b><br>Your ${a.window_days}-day reference-condition pace at ${DATA.target_hr} bpm is <b>${pace(c.pace_min_mile)}/mi</b>. ${change?`That is ${Math.abs(change.pace_change_seconds_per_mile).toFixed(0)} sec/mi ${change.pace_change_seconds_per_mile<0?'faster':'slower'} than the preceding window.`:'There is no comparable preceding window.'}<p>${c.run_count} scored runs contribute. “${a.evidence_quality}” describes evidence density, not your fitness.</p></div><div class="metric"><span>95% range</span><b>±${Math.round(c.uncertainty_95_min_mile*60)} sec</b></div><div class="metric"><span>Change</span><b>${delta}</b><small>${prob} probability faster</small></div><div class="metric"><span>Personal percentile</span><b>${a.personal_history_percentile==null?'—':Math.round(a.personal_history_percentile)}</b><small>100 = your best history</small></div><div class="metric"><span>Best sustained</span><b>${best?pace(best.pace_min_mile)+'/mi':'—'}</b><small>${bestDate}${best?` · ${best.run_count} runs`:''}</small></div>`}
function initSummary(){document.getElementById('target-copy').textContent=DATA.target_hr;document.querySelectorAll('.target-hr-copy').forEach(node=>{node.textContent=`${DATA.target_hr} bpm`});const s=DATA.summary,a=currentAnalysis(),prior=['published_reference','prior_anchored'].includes(DATA.model.method_kind),change=a.change_prior_window,hp=DATA.model.heat_posterior||{};document.getElementById('model-note').className='model-note good';document.getElementById('model-note').innerHTML=prior?`<b>Fitness is allowed to move.</b><br>Heat begins with literature, then local matched evidence gets ${Math.round((hp.personal_data_weight||0)*100)}% influence. Current heat evidence: <b>${hp.confidence||'low'}</b>.`:`<b>Model result.</b><br>${DATA.model.selected_model||'No model metadata.'}`;const cards=[['Archive activities',s.activities],['HR/GPS scored subset',s.scored_runs],[`${a.window_days}-day fitness`,pace(a.current.pace_min_mile)],['Vs prior window',change?`${signed(change.pace_change_seconds_per_mile,0)} sec/mi`:'—'],['Personal percentile',a.personal_history_percentile==null?'—':Math.round(a.personal_history_percentile)]];document.getElementById('cards').innerHTML=cards.map(([l,v])=>`<div class="card"><div class="label">${l}</div><div class="value">${v}</div></div>`).join('');document.getElementById('heat-subtitle').innerHTML=prior?`Prior 0.20%/°C WBGT → personal likelihood ${hp.personal_likelihood_mean_fraction_per_c==null?'insufficient':num(hp.personal_likelihood_mean_fraction_per_c*100,2)+'%/°C'} → posterior ${num((hp.posterior_mean_fraction_per_c||0)*100,2)}%/°C. ${hp.matched_run_count||0} matched runs; ${hp.confidence||'low'} confidence.`:'Exploratory model output.';document.getElementById('generated').textContent=`Generated locally ${new Date(DATA.generated_at).toLocaleString()} · ${DATA.audit.total_tcx_files} TCX files · Open this file without a server.`}
function initCharts(){const scored=DATA.runs.filter(r=>r.standardized_pace!=null),analysis=currentAnalysis(),trend=analysis.historical||[],tip=r=>`${dateLabel(r.start_time_utc)} · ${num(r.distance_miles,2)} mi<br>Standardized: <b>${pace(r.standardized_pace)}/mi</b><br>Moving: ${pace(r.moving_pace)}/mi · HR ${num(r.average_hr,0)}<br>${num(r.temperature_f,0)}°F / dew ${num(r.dewpoint_f,0)}°F`;let s=[{name:`Per-run pace @${DATA.target_hr}`,color:C.cyan,line:false,radius:4,data:scored.map(r=>({x:r.start_time_utc,y:r.standardized_pace,low:r.standardized_pace-r.uncertainty_95,high:r.standardized_pace+r.uncertainty_95,tip:tip(r)}))},{name:`Robust ${analysis.window_days}-day level`,color:C.orange,width:3,radius:1,data:trend.map(r=>({x:r.as_of_utc,y:r.pace_min_mile,low:r.pace_min_mile-r.uncertainty_95_min_mile,high:r.pace_min_mile+r.uncertainty_95_min_mile,tip:`${dateLabel(r.as_of_utc)}<br><b>${pace(r.pace_min_mile)}/mi</b> · ${r.run_count} runs`}))}];lineChart('fitness',s,{pace:true});legend('fitness-legend',s);
 bridgeChart();
 s=[{name:`Raw pace @${DATA.target_hr}`,color:C.blue,line:false,data:scored.map(r=>({x:r.start_time_utc,y:r.raw_pace_at_target_hr,tip:`${dateLabel(r.start_time_utc)}<br><b>${pace(r.raw_pace_at_target_hr)}/mi</b> before environmental adjustment`}))}];lineChart('efficiency',s,{pace:true});legend('efficiency-legend',s);
 const drift=DATA.runs.filter(r=>r.decoupling_percent!=null&&r.distance_miles>=3);s=[{name:'Decoupling',color:C.orange,line:false,data:drift.map(r=>({x:r.start_time_utc,y:r.decoupling_percent,tip:`${dateLabel(r.start_time_utc)} · ${num(r.distance_miles,1)} mi<br><b>${signed(r.decoupling_percent,1)}%</b> decoupling<br>First ${pace(r.first_half_pace)}/mi @ ${num(r.first_half_hr,0)}<br>Second ${pace(r.second_half_pace)}/mi @ ${num(r.second_half_hr,0)}`}))}];lineChart('drift',s);legend('drift-legend',s);
 const wr=DATA.runs.filter(r=>r.temperature_f!=null);s=[{name:'Temperature',color:C.orange,line:false,data:wr.map(r=>({x:r.start_time_utc,y:r.temperature_f,tip:`${dateLabel(r.start_time_utc)}<br>${num(r.temperature_f,1)}°F · dew ${num(r.dewpoint_f,1)}°F`}))},{name:'Dew point',color:C.cyan,line:false,data:wr.map(r=>({x:r.start_time_utc,y:r.dewpoint_f,tip:`${dateLabel(r.start_time_utc)}<br>Dew point ${num(r.dewpoint_f,1)}°F`}))}];lineChart('weather',s);legend('weather-legend',s);heatChart();bars()}
const cols=[['date','Date'],['distance_miles','Miles'],['moving_pace','Moving'],['average_hr','Avg HR'],['temperature_f','Temp'],['dewpoint_f','Dew'],['raw_pace_at_target_hr',`Raw @${DATA.target_hr}`],['environmental_adjustment','Env Δ'],['standardized_pace',`Standardized @${DATA.target_hr}`],['uncertainty_95','95% ±'],['decoupling_percent','Drift'],['gps_quality','GPS'],['workout_type','Type'],['model_eligible','Status']];let sortKey='date',sortDir=-1;
function fmtCell(r,k){if(['moving_pace','raw_pace_at_target_hr','standardized_pace'].includes(k))return `<span class="pace">${pace(r[k])}</span>`;if(k==='environmental_adjustment')return r[k]==null?'—':signed(r[k]*60,0)+'s';if(k==='uncertainty_95')return r[k]==null?'—':`±${Math.round(r[k]*60)}s`;if(k==='distance_miles')return num(r[k],2);if(['temperature_f','dewpoint_f','average_hr'].includes(k))return num(r[k],0);if(k==='decoupling_percent')return r[k]==null?'—':signed(r[k],1)+'%';if(k==='model_eligible')return `<span class="pill ${r[k]?'ok':'no'}">${r[k]?'eligible':'excluded'}</span>`;if(k==='gps_quality')return r[k]==='gps_missing'?'<span class="pill no">GPS missing</span>':r[k].replace('gps_','');return r[k]??'—'}
function renderTable(){const q=document.getElementById('search').value.toLowerCase(),f=document.getElementById('eligibility').value;let rows=DATA.runs.filter(r=>JSON.stringify([r.date,r.workout_type,r.notes,r.exclusion_reason]).toLowerCase().includes(q)).filter(r=>f==='all'||(f==='eligible'&&r.model_eligible)||(f==='excluded'&&!r.model_eligible)||(f==='scored'&&r.standardized_pace!=null));rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(x==null)return 1;if(y==null)return-1;return (x<y?-1:x>y?1:0)*sortDir});document.getElementById('table-count').textContent=`${rows.length} shown`;document.getElementById('run-body').innerHTML=rows.map(r=>`<tr data-id="${r.database_id}">${cols.map(([k])=>`<td>${fmtCell(r,k)}</td>`).join('')}</tr>`).join('');document.querySelectorAll('#run-body tr').forEach(tr=>tr.onclick=()=>openRun(Number(tr.dataset.id)))}
function initTable(){document.getElementById('run-head').innerHTML=cols.map(([k,l])=>`<th data-key="${k}">${l}</th>`).join('');document.querySelectorAll('#run-head th').forEach(th=>th.onclick=()=>{const k=th.dataset.key;sortDir=sortKey===k?-sortDir:1;sortKey=k;renderTable()});document.getElementById('search').oninput=renderTable;document.getElementById('eligibility').onchange=renderTable;renderTable()}
function evidenceStep(r,label,contributionKey,evidenceKey){const v=(r.contributions||{})[contributionKey]||0,e=(r.adjustment_evidence||{})[evidenceKey]||{},personal=e.personal_data_weight==null?'':` · ${Math.round(e.personal_data_weight*100)}% personal`;return `<div class="step"><b>${label}</b> ${signed(v*60,1)} sec/mi <span class="pill ${e.confidence==='unavailable'?'no':'ok'}">${e.confidence||'unavailable'}</span><small>${personal}${e.local_match_count==null?'':` · ${e.local_match_count} local matches`}<br>${e.basis||''}</small></div>`}
function openRun(id){const r=DATA.runs.find(x=>x.database_id===id);document.getElementById('modal-title').textContent=`${r.date} · ${num(r.distance_miles,2)} miles`;const details=[['Moving pace',pace(r.moving_pace)+'/mi'],['Average HR',num(r.average_hr,0)+' bpm'],['Temperature',r.temperature_f==null?'—':num(r.temperature_f,1)+'°F'],['Dew point',r.dewpoint_f==null?'—':num(r.dewpoint_f,1)+'°F'],['Wind',r.wind_mph==null?'—':num(r.wind_mph,1)+' mph'],['95% total uncertainty',r.uncertainty_95==null?'—':`±${Math.round(r.uncertainty_95*60)} sec/mi`],['First half',`${pace(r.first_half_pace)} @ ${num(r.first_half_hr,0)}`],['Second half',`${pace(r.second_half_pace)} @ ${num(r.second_half_hr,0)}`],['Decoupling',r.decoupling_percent==null?'—':signed(r.decoupling_percent,1)+'%']];const chain=r.raw_pace_at_target_hr==null?'<p>This activity was not scored.</p>':`<div class="detail-grid"><div class="detail"><span>Raw pace @${DATA.target_hr}</span><b>${pace(r.raw_pace_at_target_hr)}/mi</b></div><div class="detail"><span>Environmental adjustment</span><b>${signed(r.environmental_adjustment*60,0)} sec/mi</b></div><div class="detail"><span>Standardized pace @${DATA.target_hr}</span><b>${pace(r.standardized_pace)}/mi</b></div></div>`;const steps=r.raw_pace_at_target_hr==null?'':evidenceStep(r,'Grade','grade_adjustment','grade')+evidenceStep(r,'Temperature','temperature_adjustment','temperature')+evidenceStep(r,'Dew point','dewpoint_adjustment','dew_point')+evidenceStep(r,'Wind','wind_adjustment','wind')+evidenceStep(r,'Drift','drift_adjustment','drift');document.getElementById('modal-body').innerHTML=`<h3>Audit chain</h3>${chain}<p>HR normalization from the observed windows to raw @${DATA.target_hr}: <b>${signed(((r.contributions||{}).hr_normalization||0)*60,1)} sec/mi</b>.</p><h3>Environmental decomposition and evidence</h3><div class="steps">${steps||'No decomposition.'}</div><div class="detail-grid">${details.map(([a,b])=>`<div class="detail"><span>${a}</span><b>${b}</b></div>`).join('')}</div><h3>Context</h3><p>GPS: ${r.gps_quality} · HR: ${r.hr_quality} · Distance: ${r.distance_source}<br>Type: ${r.workout_type} · Model: ${r.selected_model||'not eligible'}<br>Exclusion: ${r.exclusion_reason||'none'}<br>Notes: ${r.notes||'none'}</p><h3>Moving HR zones</h3><p>${Object.entries(r.zones_minutes).map(([z,m])=>`${z.toUpperCase()} ${m} min`).join(' · ')||'Unavailable'}</p>`;document.getElementById('modal').classList.add('open')}
function closeModal(){document.getElementById('modal').classList.remove('open')}document.getElementById('modal').addEventListener('click',e=>{if(e.target.id==='modal')closeModal()});document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal()});
initAnalysisControl();initSummary();renderFitnessReadout();initCharts();initTable();const decisions=DATA.model.selection_decisions||[],cv=DATA.model.grouped_cross_validation||{},tcv=DATA.model.time_blocked_cross_validation||{},hrc=DATA.model.hr_calibration||{};document.getElementById('cv-details').innerHTML=`<h3>Method: ${DATA.model.selected_model||'—'}</h3><p>${DATA.model.model_observation_unit||''}</p><p>${decisions.join('<br>')}</p>${Object.keys(cv).length?`<table class="cv-table"><tr><th>Model</th><th>Grouped MAE</th><th>Time-blocked MAE</th></tr>${Object.entries(cv).map(([n,v])=>`<tr><td>${n}</td><td>${num(v.mae_seconds_per_mile,1)}s</td><td>${num(tcv[n]?.mae_seconds_per_mile,1)}s</td></tr>`).join('')}</table>`:''}<p>Within-run HR calibration: <b>${num(hrc.speed_mps_per_bpm,4)} m/s per bpm</b> from ${hrc.contributing_runs||'—'} runs. Residual SD: ${num(DATA.model.residual_standard_deviation_seconds_per_mile,1)} sec/mi.</p>`;
</script></body></html>'''
