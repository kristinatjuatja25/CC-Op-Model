# ==============================
# Erlang FTE Calculator + Staffing Optimizer (Streamlit)
# ==============================
import io
import math
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt

# Try to import OR-Tools CBC (install on first run if needed)
try:
    from ortools.linear_solver import pywraplp
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ortools"])
    from ortools.linear_solver import pywraplp

st.set_page_config(page_title="Erlang + Staffing Optimizer", layout="wide")
st.title("📊 Erlang FTE Calculator + 👷 Staffing Optimization")

# ---------------- Session state defaults ----------------
for key, default in {
    "df_input": None,
    "time_col": None,
    "results_full_df": None,
    "results_df": None,
    "_last_uploaded_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================
# Step 1: Upload Input File
# ==============================
st.header("📁 Step 1: Upload Input File")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # If user re-uploads a new file, reset downstream state
    if st.session_state._last_uploaded_name != uploaded_file.name:
        st.session_state._last_uploaded_name = uploaded_file.name
        st.session_state.results_full_df = None
        st.session_state.results_df = None

    st.session_state.df_input = pd.read_excel(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(st.session_state.df_input.head())

    # Detect time column (first column with 'time' in its name)
    time_col_candidates = [col for col in st.session_state.df_input.columns if "time" in str(col).lower()]
    if not time_col_candidates:
        st.error("No time column detected in the uploaded file. Please include a column with 'time' in the name.")
        st.session_state.time_col = None
    else:
        st.session_state.time_col = time_col_candidates[0]

# ==============================
# Step 2: Erlang Input Parameters
# ==============================
st.header("⚙️ Step 2: Erlang Input Parameters")

# Default AHTs per lane (seconds)
default_aht_per_lane = {
    "Premium": 1400,
    "Elite": 760,
    "OS": 623,
    "IH_L1": 429,
    "IH_L2": 429,
    "IH_L3": 429,
}

with st.expander("Default AHTs (seconds) — update if needed", expanded=False):
    aht_per_lane = {}
    cols = st.columns(3)
    for i, (lane, default) in enumerate(default_aht_per_lane.items()):
        with cols[i % 3]:
            aht_per_lane[lane] = st.number_input(f"AHT for {lane}", min_value=1, value=default, step=1)

# Fallback if user never opens/touches the expander
if 'aht_per_lane' not in locals() or not aht_per_lane:
    aht_per_lane = default_aht_per_lane.copy()

sl_target = st.number_input("Service Level Target (%)", min_value=1, max_value=100, value=90, step=1)
sl_time_seconds = st.number_input("Service Level Time (seconds)", min_value=1, value=30, step=1)

st.subheader("Other Operational Parameters")
max_occupancy = st.number_input("Max Occupancy (%)", min_value=1, max_value=100, value=85, step=1)
shrinkage = st.number_input("Shrinkage (%)", min_value=0, max_value=100, value=20, step=1)
fte_hours_per_week = st.number_input("FTE work hours per week", min_value=1.0, value=37.5, step=0.1)
patience_time_seconds = st.number_input("Patience Time (seconds)", min_value=1, value=60, step=1)
min_call_volume = st.number_input("Min Call Volume", min_value=0, value=1, step=1)
min_agents_for_low_volume = st.number_input("Min Agents for Low Volume", min_value=0, value=0, step=1)

# ==============================
# Step 3: Run Erlang Calculation
# ==============================
st.header("🧮 Step 3: Calculate Erlang A & FTEs")
run_calc = st.button("Run Calculation", key="run_calc")

def _erlang_c_probability_of_delay(a: float, k: int) -> float:
    if k <= 0:
        return 1.0
    if a <= 0:
        return 0.0
    if k <= a:
        return 1.0
    s = 0.0
    term = 1.0
    for i in range(k):
        if i > 0:
            term *= a / i
        s += term
    term *= a / k
    num = term * (k / (k - a))
    den = s + num
    return num / den

def calculate_agents_erlang_a(
    call_volume,
    aht_seconds,
    sl_target_percent,
    sl_time_seconds,
    shrinkage,
    max_occupancy,
    patience_time_seconds,   # kept for interface consistency
    min_call_volume,
    min_agents_for_low_volume
):
    if call_volume < min_call_volume:
        raw_k = int(max(min_agents_for_low_volume, 0))
    else:
        a = (float(call_volume) * float(aht_seconds)) / 1800.0  # 30-min interval
        k = max(int(math.ceil(a)), 1)
        sl_target_f = float(sl_target_percent) / 100.0
        while True:
            if k > 1000:  # safety cap
                break
            p_delay = _erlang_c_probability_of_delay(a, k)
            if k > a and aht_seconds > 0:
                sl = 1.0 - p_delay * math.exp(-(k - a) * (sl_time_seconds / aht_seconds))
            else:
                sl = 0.0
            if sl >= sl_target_f:
                break
            k += 1
        raw_k = k

    # Apply shrinkage
    shrink_factor = 1.0 - (shrinkage / 100.0)
    sched = int(math.ceil(raw_k / shrink_factor)) if shrink_factor > 0 else raw_k

    # Occupancy cap
    a_for_cap = (float(call_volume) * float(aht_seconds)) / 1800.0
    max_occ_f = float(max_occupancy) / 100.0
    if a_for_cap > 0 and sched > 0:
        while (a_for_cap / sched) > max_occ_f:
            sched += 1

    # Final metrics
    if a_for_cap <= 0 or sched <= 0:
        final_sl = 1.0 if a_for_cap <= 0 else 0.0
        final_occupancy = 0.0
        p_abandon = 0.0
    else:
        p_delay = _erlang_c_probability_of_delay(a_for_cap, sched)
        if sched > a_for_cap and aht_seconds > 0:
            final_sl = 1.0 - p_delay * math.exp(-(sched - a_for_cap) * (sl_time_seconds / aht_seconds))
        else:
            final_sl = 0.0
        final_occupancy = min(a_for_cap / sched, max_occ_f)
        p_abandon = max(0.0, min(1.0, p_delay))

    return {
        "call_volume": float(call_volume),
        "raw_agents": int(raw_k),
        "scheduled_agents": int(sched),
        "service_level": float(final_sl),
        "occupancy": float(final_occupancy),
        "abandonment_probability": float(p_abandon),
    }

if run_calc:
    if st.session_state.df_input is None or st.session_state.time_col is None:
        st.error("Please upload a valid Excel with a time column in Step 1.")
    else:
        rows = []
        for seq, r in enumerate(st.session_state.df_input.itertuples(index=False)):
            day = getattr(r, 'Day', 'Unknown')
            t_val = getattr(r, st.session_state.time_col)

            # Convert time-like values to HH:MM
            if isinstance(t_val, pd.Timestamp):
                hhmm = f"{t_val.hour:02d}:{t_val.minute:02d}"
            else:
                try:
                    tt = pd.to_datetime(t_val)
                    hhmm = f"{tt.hour:02d}:{tt.minute:02d}"
                except Exception:
                    hhmm = str(t_val)

            for lane, aht_sec in aht_per_lane.items():
                vol = float(getattr(r, lane, 0))
                res = calculate_agents_erlang_a(
                    call_volume=vol,
                    aht_seconds=aht_sec,
                    sl_target_percent=sl_target,
                    sl_time_seconds=sl_time_seconds,
                    shrinkage=shrinkage,
                    max_occupancy=max_occupancy,
                    patience_time_seconds=patience_time_seconds,
                    min_call_volume=min_call_volume,
                    min_agents_for_low_volume=min_agents_for_low_volume
                )
                rows.append({
                    "seq": seq,
                    "Lane": lane,
                    "Day": day,
                    "Interval_Time": hhmm,
                    "call_volume": res['call_volume'],
                    "raw_agents": res['raw_agents'],
                    "scheduled_agents": res['scheduled_agents'],
                    "FTEs_per_interval": (res['scheduled_agents'] * 0.5) / fte_hours_per_week,
                    "service_level": res['service_level'] * 100,
                    "occupancy": res['occupancy'],
                    "abandonment_probability": res['abandonment_probability']
                })

        results_full_df = pd.DataFrame(rows).sort_values(['Lane', 'seq']).drop(columns=['seq'])
        results_df = results_full_df[['Lane', 'Day', 'Interval_Time', 'scheduled_agents']].copy()

        # Persist in session state so optimizer can read even after reruns
        st.session_state.results_full_df = results_full_df
        st.session_state.results_df = results_df

        total_fte = results_full_df.groupby('Lane', as_index=True)['FTEs_per_interval'].sum().to_frame('Total FTEs')
        st.subheader("Total FTEs per Lane")
        st.dataframe(total_fte)

        st.subheader("Detailed Results (first 12 rows)")
        st.dataframe(results_full_df.head(12))

# If results already exist from a previous run, show a small reminder
if st.session_state.results_df is not None and not st.session_state.results_df.empty:
    st.info("Erlang results are loaded in memory and ready for the optimizer.")

# ==============================
# Step 4: Staffing Optimization (reads from session_state.results_df)
# ==============================
st.header("🧩 Step 4: Staffing Optimization")

# ---- Optimizer controls (Streamlit widgets)
st.subheader("Optimizer Controls")
colA, colB, colC = st.columns(3)
with colA:
    required_coverage_percent = st.number_input("Required coverage % (vs RAW demand)", value=95.0, min_value=0.0, max_value=100.0, step=0.5)
    start_on_full_hour_only = st.checkbox("Start shifts on full hour only (:00)", value=False)
    allow_part_time = st.checkbox("Allow part-time shifts", value=True)
with colB:
    use_overstaff_cap = st.checkbox("Enable overstaff cap", value=False)
    max_overstaff_pct = st.number_input("Max overstaff % (if enabled)", value=0.0, min_value=0.0, step=1.0)
    cmax_reference = st.selectbox("Overstaff cap reference", options=["raw", "min"], index=0)
with colC:
    time_limit_sec = st.number_input("Solver time limit per lane (sec)", value=60, min_value=1, step=1)
    ft_cost = st.number_input("FT unit cost", value=1.0, min_value=0.0, step=0.1)
    pt_cost = st.number_input("PT unit cost", value=0.6, min_value=0.0, step=0.1)

st.markdown("**Full-time config (unchanged):** 9h span, breaks [60,30], work=7.5h, 5 consecutive days.")
st.markdown("**Part-time config:** set span & breaks; work = span − sum(breaks).")

col1, col2, col3 = st.columns(3)
with col1:
    ft_earliest_start = st.text_input("FT earliest start", value="05:00")
    ft_latest_start   = st.text_input("FT latest start", value="22:00")
    pt_earliest_start = st.text_input("PT earliest start", value="05:00")
with col2:
    pt_latest_start   = st.text_input("PT latest start", value="21:00")
    ft_consec_days    = st.number_input("FT consecutive days", value=5, min_value=1, max_value=7, step=1)
    pt_consec_days    = st.number_input("PT consecutive days", value=5, min_value=1, max_value=7, step=1)
with col3:
    pt_span_hours     = st.number_input("PT span hours (incl. breaks)", value=4.0, min_value=1.0, step=0.25)
    pt_breaks_text    = st.text_input("PT breaks minutes (comma-separated)", value="15")
    interval_minutes  = st.selectbox("Interval minutes", options=[15,30,60], index=1)

# Lane operating windows
st.subheader("Lane operating windows")
colL, colR = st.columns(2)
with colL:
    premium_open  = st.text_input("Premium open", value="07:00")
    premium_close = st.text_input("Premium close (exclusive)", value="00:00")
with colR:
    others_open   = st.text_input("Others open (blank=24/7)", value="")
    others_close  = st.text_input("Others close (blank=24/7)", value="")

# Single button instance to avoid duplicate element ID
run_optimizer = st.button("Run Optimizer", key="run_optimizer")

# ==============================
# Optimizer function (reads results_df from session_state)
# ==============================
def run_staffing_optimizer(results_df: pd.DataFrame) -> dict:
    CFG = {
        "ft_work_hours": 7.5,
        "pt_work_hours": None,  # computed from PT span & breaks
        "consec_days": int(ft_consec_days),
        "pt_consec_days": int(pt_consec_days),
        "interval_minutes": int(interval_minutes),

        "ft_span_hours": 9.0,
        "ft_breaks_min": [60, 30],
        "pt_span_hours": float(pt_span_hours),
        "pt_breaks_min": [int(x) for x in pt_breaks_text.split(",") if x.strip()],

        "ft_earliest_start": ft_earliest_start,
        "ft_latest_start":   ft_latest_start,
        "pt_earliest_start": pt_earliest_start,
        "pt_latest_start":   pt_latest_start,

        "required_coverage_percent": float(required_coverage_percent),
        "max_overstaff_pct": (float(max_overstaff_pct) if use_overstaff_cap else None),

        "start_on_full_hour_only": bool(start_on_full_hour_only),
        "allow_part_time": bool(allow_part_time),

        "cmax_reference": cmax_reference,

        "use_cost_per_hour": False,
        "ft_cost": float(ft_cost),
        "pt_cost": float(pt_cost),

        "fte_hours_per_week": float(fte_hours_per_week),

        "premium_open": premium_open,
        "premium_close": premium_close,
        "others_open": others_open,
        "others_close": others_close,

        "time_limit_sec": int(time_limit_sec),
    }

    # Compute PT work hours from span & breaks
    CFG["pt_work_hours"] = CFG["pt_span_hours"] - sum(CFG["pt_breaks_min"]) / 60.0
    if CFG["pt_work_hours"] < 0:
        raise ValueError("PT work hours became negative — check PT span and breaks.")

    # ---------- Helpers ----------
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_to_idx = {d:i for i,d in enumerate(day_order)}
    idx_to_day = {i:d for d,i in day_to_idx.items()}

    def to_time(x):
        if isinstance(x, dt.time): return x
        try: return pd.to_datetime(str(x)).time()
        except: return None

    def parse_hhmm(s):
        h,m = map(int, str(s).split(":")); return dt.time(h,m)

    def in_window(t, open_t, close_t):
        if open_t is None or close_t is None: return True
        if open_t <= close_t: return open_t <= t < close_t
        return (t >= open_t) or (t < close_t)

    def compute_break_windows(span_len, break_lens):
        if not break_lens: return []
        total_break = sum(break_lens)
        work_total = span_len - total_break
        segs = len(break_lens)+1
        base,extra = divmod(work_total,segs)
        windows,ofs=[],0
        for i,bl in enumerate(break_lens):
            ofs += base+(1 if i<extra else 0)
            windows.append((ofs,bl))
            ofs+=bl
        return windows

    def covered_pairs_with_breaks(start_idx, span_len, break_windows, num_intervals):
        pairs=[]; is_break=[False]*span_len
        for off,ln in break_windows:
            for k in range(off,min(span_len,off+ln)): is_break[k]=True
        for k in range(span_len):
            if not is_break[k]:
                off=start_idx+k
                pairs.append((off//num_intervals, off%num_intervals))
        return pairs

    # ---------- Prepare demand ----------
    need_cols = {"Lane", "Day", "Interval_Time", "scheduled_agents"}
    if not need_cols.issubset(results_df.columns):
        raise ValueError(f"results_df missing columns: {need_cols - set(results_df.columns)}")

    df_req = results_df.copy()
    df_req['Day'] = pd.Categorical(df_req['Day'], categories=day_order, ordered=True)
    df_req['Interval_Time'] = df_req['Interval_Time'].apply(to_time)
    df_req['scheduled_agents'] = pd.to_numeric(df_req['scheduled_agents'], errors='coerce').fillna(0)

    all_times = sorted(df_req['Interval_Time'].unique())
    num_intervals = len(all_times)
    time_to_idx = {t:i for i,t in enumerate(all_times)}
    hhmm_from_idx = {i:f"{all_times[i].hour:02d}:{all_times[i].minute:02d}" for i in range(num_intervals)}

    ft_earliest,ft_latest = parse_hhmm(CFG["ft_earliest_start"]),parse_hhmm(CFG["ft_latest_start"])
    pt_earliest,pt_latest = parse_hhmm(CFG["pt_earliest_start"]),parse_hhmm(CFG["pt_latest_start"])

    def _starts_between(t0,t1,only_full):
        return [time_to_idx[t] for t in all_times if (t0<=t<=t1) and (not only_full or t.minute==0)]

    ft_start_idxs=_starts_between(ft_earliest,ft_latest,CFG["start_on_full_hour_only"])
    pt_start_idxs=_starts_between(pt_earliest,pt_latest,CFG["start_on_full_hour_only"])

    iv=CFG["interval_minutes"]
    ft_span_len=int(round(CFG["ft_span_hours"]*60/iv))
    pt_span_len=int(round(CFG["pt_span_hours"]*60/iv))
    ft_work_len=int(round(CFG["ft_work_hours"]*60/iv))
    pt_work_len=int(round(CFG["pt_work_hours"]*60/iv))
    ft_break_lens=[int(round(m/iv)) for m in CFG["ft_breaks_min"]]
    pt_break_lens=[int(round(m/iv)) for m in CFG["pt_breaks_min"]]
    ft_break_windows=compute_break_windows(ft_span_len,ft_break_lens)
    pt_break_windows=compute_break_windows(pt_span_len,pt_break_lens)

    # Avoid shadowing: use parsed names with _t suffix
    prem_open_t, prem_close_t = parse_hhmm(CFG["premium_open"]), parse_hhmm(CFG["premium_close"])
    if CFG["others_open"] and CFG["others_close"]:
        others_open_t, others_close_t = parse_hhmm(CFG["others_open"]), parse_hhmm(CFG["others_close"])
    else:
        others_open_t, others_close_t = None, None  # 24/7

    lanes=sorted(df_req['Lane'].unique())
    req_min_mult=CFG["required_coverage_percent"]/100.0

    req_min,req_raw={},{}
    lane_allowed_starts_ft,lane_allowed_starts_pt={},{}
    for lane in lanes:
        is_prem=(str(lane).strip().lower()=="premium")
        open_t, close_t = (prem_open_t, prem_close_t) if is_prem else (others_open_t, others_close_t)

        allowed_ft=[si for si in ft_start_idxs if in_window(all_times[si],open_t,close_t)]
        allowed_pt=[si for si in pt_start_idxs if in_window(all_times[si],open_t,close_t)] if CFG["allow_part_time"] else []
        lane_allowed_starts_ft[lane]=allowed_ft; lane_allowed_starts_pt[lane]=allowed_pt
        Mraw=[[0]*num_intervals for _ in range(7)]
        Mmin=[[0]*num_intervals for _ in range(7)]
        sub=df_req[df_req['Lane']==lane]
        for _,r in sub.iterrows():
            d,ti=day_to_idx[r['Day']],time_to_idx[r['Interval_Time']]
            raw_need=math.ceil(float(r['scheduled_agents']))
            min_need=math.ceil(float(r['scheduled_agents'])*req_min_mult)
            if not in_window(all_times[ti],open_t,close_t): raw_need=min_need=0
            Mraw[d][ti]=raw_need; Mmin[d][ti]=min_need
        req_raw[lane]=Mraw; req_min[lane]=Mmin

    def build_cover_maps_with_breaks(allowed_starts, span_len, break_windows, consec_days):
        cover={(d,ti):[] for d in range(7) for ti in range(num_intervals)}
        if not allowed_starts:
            return cover
        pairs_cache={si:covered_pairs_with_breaks(si,span_len,break_windows,num_intervals) for si in allowed_starts}
        for sday in range(7):
            for si in allowed_starts:
                pairs=pairs_cache[si]
                for dblock in range(consec_days):
                    for doff,ti in pairs:
                        day_target=(sday+dblock+doff)%7
                        cover[(day_target,ti)].append((sday,si))
        return cover

    # --- NEW: progress bar before solving lanes
    progress = st.progress(0, text="Starting optimization...")

    # ---------- Solve ----------
    ft_schedules,pt_schedules,coverage_rows={}, {}, []
    for i, lane in enumerate(lanes, start=1):
        allowed_ft,allowed_pt=lane_allowed_starts_ft[lane],lane_allowed_starts_pt[lane]
        if not allowed_ft and not allowed_pt:
            progress.progress(i / len(lanes), text=f"⚠️ Skipped {lane} (no allowed starts)")
            continue
        ft_cover=build_cover_maps_with_breaks(allowed_ft,ft_span_len,ft_break_windows,CFG["consec_days"])
        pt_cover=build_cover_maps_with_breaks(allowed_pt,pt_span_len,pt_break_windows,CFG["pt_consec_days"])

        solver=pywraplp.Solver.CreateSolver("CBC")
        solver.SetTimeLimit(int(CFG["time_limit_sec"]*1000))
        solver.SetSolverSpecificParametersAsString("random_seed=1234")  # stable

        FT={(d,si):solver.IntVar(0,solver.infinity(),f"FT_{d}_{si}") for d in range(7) for si in allowed_ft}
        PT={(d,si):solver.IntVar(0,solver.infinity(),f"PT_{d}_{si}") for d in range(7) for si in allowed_pt}

        obj=solver.Objective()
        for d in range(7):
            for si in allowed_ft: obj.SetCoefficient(FT[(d,si)],CFG["ft_cost"])
            for si in allowed_pt: obj.SetCoefficient(PT[(d,si)],CFG["pt_cost"])
        obj.SetMinimization()

        for d in range(7):
            for ti in range(num_intervals):
                need_min=req_min[lane][d][ti]
                if need_min>0:
                    cmin=solver.Constraint(need_min,solver.infinity())
                    for sday,sidx in ft_cover[(d,ti)]: cmin.SetCoefficient(FT[(sday,sidx)],1)
                    for sday,sidx in pt_cover[(d,ti)]: cmin.SetCoefficient(PT[(sday,sidx)],1)
                if CFG["max_overstaff_pct"] is not None:
                    ref_need=req_raw[lane][d][ti] if CFG["cmax_reference"]=="raw" else req_min[lane][d][ti]
                    if ref_need>0:
                        over_allow=int(math.floor(ref_need*(CFG["max_overstaff_pct"]/100.0)))
                        max_allowed=ref_need+over_allow
                        cmax=solver.Constraint(-solver.infinity(),max_allowed)
                        for sday,sidx in ft_cover[(d,ti)]: cmax.SetCoefficient(FT[(sday,sidx)],1)
                        for sday,sidx in pt_cover[(d,ti)]: cmax.SetCoefficient(PT[(sday,sidx)],1)

        status=solver.Solve()
        # Build outputs even if only FEASIBLE
        if allowed_ft:
            idx2hhmm_ft={i:f"{all_times[i].hour:02d}:{all_times[i].minute:02d}" for i in allowed_ft}
            ft_tab=pd.DataFrame(0,index=[idx2hhmm_ft[i] for i in allowed_ft],columns=day_order)
            for d in range(7):
                for si in allowed_ft:
                    v=int(round(FT[(d,si)].solution_value()))
                    if v: ft_tab.loc[idx2hhmm_ft[si], day_order[d]] = v
            ft_schedules[lane]=ft_tab
        else:
            ft_schedules[lane]=pd.DataFrame(columns=day_order)

        if allowed_pt:
            idx2hhmm_pt={i:f"{all_times[i].hour:02d}:{all_times[i].minute:02d}" for i in allowed_pt}
            pt_tab=pd.DataFrame(0,index=[idx2hhmm_pt[i] for i in allowed_pt],columns=day_order)
            for d in range(7):
                for si in allowed_pt:
                    v=int(round(PT[(d,si)].solution_value()))
                    if v: pt_tab.loc[idx2hhmm_pt[si], day_order[d]] = v
            pt_schedules[lane]=pt_tab
        else:
            pt_schedules[lane]=pd.DataFrame(columns=day_order)

        # Coverage rows
        raw_need_map={(day_to_idx[r['Day']], time_to_idx[r['Interval_Time']]): float(r['scheduled_agents'])
                      for _,r in df_req[df_req['Lane']==lane].iterrows()}
        for d in range(7):
            for ti in range(num_intervals):
                raw_need = int(math.ceil(raw_need_map.get((d, ti), 0.0)))
                need_min = req_min[lane][d][ti]
                if raw_need==0 and need_min==0:
                    continue
                ft_staff = sum(int(round(FT[(sday, sidx)].solution_value())) for sday, sidx in (ft_cover[(d, ti)] if allowed_ft else []))
                pt_staff = sum(int(round(PT[(sday, sidx)].solution_value())) for sday, sidx in (pt_cover[(d, ti)] if allowed_pt else []))
                tot = ft_staff + pt_staff
                cov_raw = "N/A" if raw_need == 0 else f"{(tot / raw_need * 100):.1f}%"
                cov_min = "N/A" if need_min == 0 else f"{(tot / need_min * 100):.1f}%"
                coverage_rows.append({
                    "Lane": lane,
                    "Day": day_order[d],
                    "Interval_Time": f"{all_times[ti].hour:02d}:{all_times[ti].minute:02d}",
                    "Raw_Required": raw_need,
                    "Min_Required": need_min,
                    "FT Staffed": ft_staff,
                    "PT Staffed": pt_staff,
                    "Total Staffed": tot,
                    "Coverage % vs Raw": cov_raw,
                    "Coverage % vs Min": cov_min
                })

        # --- update progress bar for this lane
        progress.progress(i / len(lanes), text=f"✅ Completed {lane}")

    progress.empty()  # remove the bar at the end

    coverage_df = pd.DataFrame(coverage_rows)

    # Agent IDs and headcount
    agent_rows=[]; aid_ft=aid_pt=0
    def add_agents(tab, lane, kind):
        nonlocal aid_ft, aid_pt
        if tab is None or tab.empty: return
        for hhmm in tab.index:
            for d in tab.columns:
                n = int(tab.loc[hhmm, d])
                for _ in range(n):
                    if kind=="FT":
                        aid_ft += 1; aid = f"FT_{aid_ft:05d}"
                    else:
                        aid_pt += 1; aid = f"PT_{aid_pt:05d}"
                    agent_rows.append({"Agent_ID":aid,"Lane":lane,"Agent_Type":kind,"Anchor_Day":d,"Start_Time":hhmm})

    for lane,tab in ft_schedules.items(): add_agents(tab,lane,"FT")
    for lane,tab in pt_schedules.items(): add_agents(tab,lane,"PT")

    agents_df = pd.DataFrame(agent_rows).sort_values(["Lane","Agent_Type","Anchor_Day","Start_Time"]) if agent_rows else pd.DataFrame(
        columns=["Agent_ID","Lane","Agent_Type","Anchor_Day","Start_Time"]
    )
    hc_ft = (agents_df["Agent_Type"]=="FT").sum() if not agents_df.empty else 0
    hc_pt = (agents_df["Agent_Type"]=="PT").sum() if not agents_df.empty else 0
    hc_total = len(agents_df)
    ft_fte = hc_ft * (CFG["ft_work_hours"] * (CFG["consec_days"])) / CFG["fte_hours_per_week"] if hc_ft else 0.0
    pt_fte = hc_pt * (CFG["pt_work_hours"] * (CFG["pt_consec_days"])) / CFG["fte_hours_per_week"] if hc_pt else 0.0
    fte_total = ft_fte + pt_fte
    headcount_summary = pd.DataFrame({"Count":[hc_ft,hc_pt,hc_total],
                                      "FTE_equivalent":[ft_fte,pt_fte,fte_total]},
                                     index=["FT","PT","TOTAL"])

    # Roster (SPAN end times)
    hhmm_to_idx = {f"{t.hour:02d}:{t.minute:02d}": i for i,t in enumerate(all_times)}
    def end_time_and_offset(start_hhmm, kind):
        si = hhmm_to_idx[start_hhmm]
        span_len = ft_span_len if kind=="FT" else pt_span_len
        end_idx = (si + span_len) % num_intervals
        end_day_offset = (si + span_len) // num_intervals
        return hhmm_from_idx[end_idx], end_day_offset

    def rotate_days(d0, span):
        i = day_to_idx[d0]
        return [day_order[(i + k) % 7] for k in range(span)]

    def expand_daily(tab, consec_days):
        if tab is None or tab.empty:
            return pd.DataFrame(columns=day_order)
        out = pd.DataFrame(0, index=tab.index, columns=day_order)
        for hhmm in tab.index:
            for d in tab.columns:
                n = int(tab.loc[hhmm, d])
                if n <= 0: continue
                for dd in rotate_days(d, consec_days):
                    out.loc[hhmm, dd] += n
        return out

    expanded_daily = {}
    for lane in lanes:
        expanded_daily.setdefault(lane,{})
        expanded_daily[lane]['FT'] = expand_daily(ft_schedules.get(lane), CFG["consec_days"])
        expanded_daily[lane]['PT'] = expand_daily(pt_schedules.get(lane), CFG["pt_consec_days"])

    roster_rows=[]
    for _, r in agents_df.iterrows():
        anchor_day = r["Anchor_Day"]; start_hhmm=r["Start_Time"]; kind=r["Agent_Type"]
        end_hhmm, off = end_time_and_offset(start_hhmm, kind)
        days_to_expand = CFG["consec_days"] if kind=="FT" else CFG["pt_consec_days"]
        for k in range(days_to_expand):
            start_day_idx = (day_to_idx[anchor_day] + k) % 7
            start_day = idx_to_day[start_day_idx]
            end_day = idx_to_day[(start_day_idx + off) % 7]
            roster_rows.append({
                "Agent_ID": r["Agent_ID"], "Lane": r["Lane"], "Agent_Type": kind,
                "Day": start_day, "Start": start_hhmm, "End": end_hhmm, "End_Day": end_day,
                "Span_Hours": 9.0 if kind=="FT" else CFG["pt_span_hours"],
                "Work_Hours": 7.5 if kind=="FT" else CFG["pt_work_hours"]
            })
    roster_df = pd.DataFrame(roster_rows).sort_values(["Lane","Agent_Type","Agent_ID","Day","Start"]) if roster_rows else pd.DataFrame(
        columns=["Agent_ID","Lane","Agent_Type","Day","Start","End","End_Day","Span_Hours","Work_Hours"]
    )

    # Build Excel (to bytes)
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        pd.DataFrame({
            "NOTE": [
                "Coverage uses WORK minutes only (no coverage during breaks).",
                "Roster end times use SPAN (work + breaks).",
                "FT: span=9h with 60+30 breaks → 7.5h work; PT: work = span − sum(PT breaks).",
                "Min_Required = raw demand * required_coverage_percent (per interval).",
                "Objective minimizes unit cost (FT/PT costs set above).",
                "Overstaff cap: staffed ≤ ref_need + floor(ref_need * max_overstaff_pct).",
                "Premium lane window = 07:00 to 00:00 (overnight).",
                "start_on_full_hour_only=True => starts restricted to :00; otherwise uses your grid (:00 & :30).",
                "allow_part_time=False => FT-only optimization."
            ]
        }).to_excel(writer, sheet_name="READ_ME", index=False)

        coverage_df.to_excel(writer, sheet_name="Coverage_By_Interval", index=False)
        headcount_summary.to_excel(writer, sheet_name="Headcount_Summary")
        if not agents_df.empty:
            agents_df.to_excel(writer, sheet_name="Agent_IDs", index=False)
            roster_df.to_excel(writer, sheet_name="Roster_By_Agent", index=False)

        for lane in lanes:
            if lane in ft_schedules and not ft_schedules[lane].empty:
                ft_schedules[lane].to_excel(writer, sheet_name=f"{lane}_FT_Anchor_Starts")
                expanded_daily.get(lane, {}).get('FT', pd.DataFrame()).to_excel(
                    writer, sheet_name=f"{lane}_FT_Daily_Starts"
                )
            if allow_part_time and lane in pt_schedules and not pt_schedules[lane].empty:
                pt_schedules[lane].to_excel(writer, sheet_name=f"{lane}_PT_Anchor_Starts")
                expanded_daily.get(lane, {}).get('PT', pd.DataFrame()).to_excel(
                    writer, sheet_name=f"{lane}_PT_Daily_Starts"
                )
    out_buf.seek(0)

    return {
        "coverage_df": coverage_df,
        "headcount_summary": headcount_summary,
        "roster_df": roster_df,
        "excel_bytes": out_buf.getvalue()
    }

# ==============================
# Trigger optimizer (uses session_state.results_df)
# ==============================
if run_optimizer:
    results_df_ss = st.session_state.results_df
    if results_df_ss is None or results_df_ss.empty:
        st.error("No optimizer input found. Please run Step 3 (Erlang calculation) first.")
    else:
        try:
            opt_out = run_staffing_optimizer(results_df_ss)
            st.subheader("Headcount Summary")
            st.dataframe(opt_out["headcount_summary"])

            st.subheader("Coverage (sample)")
            st.dataframe(opt_out["coverage_df"].head())

            st.subheader("Roster (sample)")
            st.dataframe(opt_out["roster_df"].head())

            st.download_button(
                "📥 Download Staffing_Optimization.xlsx",
                data=opt_out["excel_bytes"],
                file_name="Staffing_Optimization.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_optimizer_xlsx"
            )
            st.success("✅ Optimizer completed with deterministic settings (CBC, random_seed=1234).")
        except Exception as e:
            st.exception(e)
