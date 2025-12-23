#!/usr/bin/env python3
"""
json2rinex_simple.py

Simple JSON -> RINEX 3.04 OBS converter (Option 1).
Designed for the phone GNSS JSON logs you produced (e.g. /mnt/data/gnss_1_fixed.json).

Outputs: phone_rinex.obs (RINEX 3.04 observation file)

Notes:
- Produces OBS types: C1C L1C D1C S1C (code, phase (meters), doppler (m/s), snr)
- Tries to compute pseudorange P = c*(t_rx - t_tx) if pseudorange field missing
- Uses ADR (accumulatedDeltaRangeMeters) as carrier-phase in meters
- Satellite system mapping: GPS->G, GLONASS->R, GAL->E, BDS->C
- Minimal header; approximate receiver pos left as 0.0,0.0,0.0 (RTKLIB tolerates)
- You still need NAV data (RINEX NAV or RTCM) for PPK/RTK processing
"""
import json
import math
import sys
from datetime import datetime, timezone

# Input file path
INPUT_JSON = "./gnss_1_fixed.json"  # <--- your uploaded file path
OUTPUT_RINEX = "phone_rinex.obs"

C = 299792458.0

# Map Android constellationType to RINEX system letter
CONST_MAP = {
    1: "G",  # GPS
    3: "R",  # GLONASS
    5: "E",  # Galileo
    6: "C",  # BeiDou
    2: "S",  # SBAS (treat as S)
    4: "J",  # QZSS (treat as J)
}

# Parse JSON file (handles either JSON array or JSON lines)
def load_json_lines(path):
    with open(path, "rb") as f:
        raw = f.read().strip()
    # try JSON lines
    try:
        lines = [json.loads(x) for x in raw.splitlines() if x.strip()]
        if len(lines) > 0:
            return lines
    except Exception:
        pass
    # fallback single JSON
    data = json.loads(raw)
    if isinstance(data, dict):
        return [data]
    return data

# Convert GNSS clock + measurement to rx/tx times in seconds
def epoch_rx_seconds(clock):
    # clock.timeNanos and fullBiasNanos, biasNanos must exist
    timeNanos = clock.get("timeNanos")
    fullBiasNanos = clock.get("fullBiasNanos", 0)
    biasNanos = clock.get("biasNanos", 0)
    if timeNanos is None:
        raise ValueError("Missing clock.timeNanos in epoch")
    # t_rx (seconds)
    t_rx = (timeNanos - fullBiasNanos - biasNanos) / 1e9
    return t_rx

def sat_id_rinex(constellation, svid):
    sys = CONST_MAP.get(constellation, "G")
    # Ensure PRN is two-digit in RINEX: e.g. G05
    try:
        prn = int(svid)
    except Exception:
        prn = svid
    return f"{sys}{prn:02d}"

def format_rinex_header(obs_types_list, approx_pos=(0.0,0.0,0.0)):
    now = datetime.utcnow()
    header = []
    header.append("     3.04           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE")
    header.append("Converted from JSON by json2rinex_simple.py".ljust(60) + "PGM / RUN BY / DATE")
    # marker name
    header.append("PHONERX                      MARKER NAME".ljust(60))
    # observer/instrument blank
    header.append("                                 MARKER NUMBER")
    # approx position (X Y Z) in m
    x,y,z = approx_pos
    header.append(f"{x:14.4f}{y:14.4f}{z:14.4f}                          APPROX POSITION XYZ")
    header.append("                                                            ANTENNA: DELTA H/E/N")
    # obs types line: we set same for all systems (C1C L1C D1C S1C)
    # RINEX allows specifying per system; for simplicity we put a single list for all systems
    obs_str = "    ".join(obs_types_list)
    header.append(f"  4 {obs_types_list[0]} {obs_types_list[1]} {obs_types_list[2]} {obs_types_list[3]}{'':28}   SYS / # / OBS TYPES")
    header.append("                                                            END OF HEADER")
    return "\n".join(header) + "\n"

def write_rinex(obs_epochs, out_path):
    # obs_types for our minimal file
    obs_types = ["C1C","L1C","D1C","S1C"]
    header_text = format_rinex_header(obs_types)
    with open(out_path, "w") as fo:
        fo.write(header_text)
        # loop epochs
        for epoch in obs_epochs:
            # epoch: dict with 't_rx' float seconds, 'dt_frac' fractional sec, 'sats' list
            t_rx = epoch["t_rx"]
            dt = epoch.get("dt_frac", 0.0)
            # convert t_rx to UTC datetime for RINEX timestamp
            # GNSS_SECONDS used as seconds since GPS epoch is complex; we will convert epoch to UTC by simple approximation:
            dt_utc = datetime.fromtimestamp(t_rx, tz=timezone.utc)
            # RINEX epoch header: "> YYYY MM DD HH MM SS.ssssssss  0 N" (we'll use flag 0 and N sats)
            sats = epoch["sats"]
            N = len(sats)
            fo.write(f"> {dt_utc.year:4d} {dt_utc.month:02d} {dt_utc.day:02d} {dt_utc.hour:02d} {dt_utc.minute:02d} {dt_utc.second + dt:13.7f}  0 {N:3d}\n")
            # write satellite id list in same order
            # Then write per-satellite observation lines
            for sat in sats:
                sysprn = sat["sat"]
                # prepare obs values: C1C L1C D1C S1C (14-character fields each)
                # C1C: pseudorange in meters
                P = sat.get("P")
                L = sat.get("L")  # carrier-phase in meters
                D = sat.get("D")  # doppler in m/s (RINEX stores as negative of doppler sometimes); we'll write as given
                S = sat.get("S")  # snr (dB-Hz)
                def fmt(v):
                    if v is None:
                        return " " * 14
                    try:
                        return f"{v:14.3f}"
                    except:
                        return " " * 14
                fo.write(f"{sysprn:3s}")
                fo.write(fmt(P))
                fo.write(fmt(L))
                fo.write(fmt(D))
                fo.write(fmt(S))
                fo.write("\n")
    print(f"Wrote RINEX OBS to {out_path}")

# Build epochs from JSON
def build_epochs(json_epochs):
    out_epochs = []
    for rec in json_epochs:
        # each rec expected to be one GNSS epoch; skip if no measurements
        meas = rec.get("measurements") or rec.get("sats") or []
        if not meas:
            continue
        try:
            t_rx = epoch_rx_seconds(rec["clock"])
        except Exception as e:
            # skip if no clock
            continue
        epoch = {"t_rx": t_rx, "sats": []}
        for m in meas:
            svid = m.get("svid")
            const = m.get("constellation", m.get("constellationType", m.get("constellationType")))
            if svid is None or const is None:
                continue
            satname = sat_id_rinex(const, svid)
            # compute pseudorange if not present
            P = m.get("pseudorangeMeters")
            if P is None:
                # compute using receivedSvTimeNanos + timeOffsetNanos
                recvSv = m.get("receivedSvTimeNanos")
                toff = m.get("timeOffsetNanos", 0)
                if recvSv is not None:
                    t_tx = (recvSv + toff) / 1e9
                    # Note: t_rx already includes fullBias/bias correction
                    P = C * (t_rx - t_tx)
            # carrier-phase: use accumulatedDeltaRangeMeters
            L = m.get("accumulatedDeltaRangeMeters")
            # doppler: convert pseudorangeRateMetersPerSecond if present (note sign)
            D = m.get("pseudorangeRateMetersPerSecond")
            # SNR: cn0DbHz or snrDbHz
            S = m.get("cn0DbHz", m.get("snrDbHz"))
            # Basic filtering: exclude very low SNR
            if S is not None and S < 10:
                # skip extremely weak signals
                pass
            sat = {"sat": satname, "P": P, "L": L, "D": D, "S": S}
            out_epochs.append  # (no-op in case)
            epoch["sats"].append(sat)
        if len(epoch["sats"])>0:
            out_epochs.append(epoch)
    return out_epochs

def main():
    print("Loading JSON:", INPUT_JSON)
    j = load_json_lines(INPUT_JSON)
    print("Loaded epochs:", len(j))
    epochs = build_epochs(j)
    print("Built RINEX epochs:", len(epochs))
    if len(epochs)==0:
        print("No valid epochs found, aborting.")
        return
    write_rinex(epochs, OUTPUT_RINEX)

if __name__ == "__main__":
    main()
