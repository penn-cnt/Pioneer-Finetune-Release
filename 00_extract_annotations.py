#!/usr/bin/env python3
"""Extract EDF+ annotations via raw header/TAL bytes; same event logic as
``00_extract_annotations.py``.

Differences vs ``00_extract_annotations.py``:
  - Reads annotations / start time from EDF bytes (no MNE / pyedflib)
  - Absolute times: header startdate/time + raw TAL onset (EDF+ wall clock)

Event filtering and seizure/patient event construction are copied from
``00_extract_annotations.py``.

Examples:
  python 00_extract_annotations_bytes.py -p RID2207
  python 00_extract_annotations_bytes.py -p PID001,PID002
  python 00_extract_annotations_bytes.py -p all
  python 00_extract_annotations_bytes.py -p all --plot
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

working_dir = os.path.abspath("")
raw_data_path = "edf"


# ── Inlined EDF byte helpers ────────────────────────────────────────────────

MAIN_HEADER_FIELDS = {
    "version": (0, 8),
    "patient": (8, 80),
    "recording": (88, 80),
    "startdate": (168, 8),
    "starttime": (176, 8),
    "n_header_bytes": (184, 8),
    "reserved": (192, 44),
    "n_records": (236, 8),
    "record_dur": (244, 8),
    "n_signals": (252, 4),
}

SIGNAL_HEADER_FIELDS = [
    ("label", 16),
    ("transducer", 80),
    ("phys_dim", 8),
    ("phys_min", 8),
    ("phys_max", 8),
    ("dig_min", 8),
    ("dig_max", 8),
    ("prefiltering", 80),
    ("n_samples", 8),
    ("reserved", 32),
]


@dataclass
class MainHeader:
    version: str
    patient: str
    recording: str
    startdate: str
    starttime: str
    n_header_bytes: int
    reserved: str
    n_records: int
    record_dur: float
    n_signals: int


@dataclass
class SignalHeader:
    label: str
    transducer: str
    phys_dim: str
    phys_min: str
    phys_max: str
    dig_min: str
    dig_max: str
    prefiltering: str
    n_samples: int
    reserved: str


@dataclass
class EdfHeader:
    main: MainHeader
    signals: list = field(default_factory=list)
    raw_bytes: bytes = b""


@dataclass(frozen=True)
class FileExtract:
    annot_df: pd.DataFrame
    start_dt: Optional[datetime]
    number_of_channels: int
    sampling_freq: float
    channel_names: list[str]


def _decode(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace").strip()


def parse_main_header(raw: bytes) -> MainHeader:
    def field_str(name: str) -> str:
        pos, n = MAIN_HEADER_FIELDS[name]
        return _decode(raw[pos : pos + n])

    return MainHeader(
        version=field_str("version"),
        patient=field_str("patient"),
        recording=field_str("recording"),
        startdate=field_str("startdate"),
        starttime=field_str("starttime"),
        n_header_bytes=int(field_str("n_header_bytes")),
        reserved=field_str("reserved"),
        n_records=int(field_str("n_records")),
        record_dur=float(field_str("record_dur") or 0),
        n_signals=int(field_str("n_signals")),
    )


def parse_signal_headers(raw: bytes, n_signals: int) -> list[SignalHeader]:
    pos = 0
    blocks: dict[str, list[str]] = {}
    for name, width in SIGNAL_HEADER_FIELDS:
        vals = []
        for _ in range(n_signals):
            vals.append(_decode(raw[pos : pos + width]))
            pos += width
        blocks[name] = vals

    signals: list[SignalHeader] = []
    for i in range(n_signals):
        signals.append(
            SignalHeader(
                label=blocks["label"][i],
                transducer=blocks["transducer"][i],
                phys_dim=blocks["phys_dim"][i],
                phys_min=blocks["phys_min"][i],
                phys_max=blocks["phys_max"][i],
                dig_min=blocks["dig_min"][i],
                dig_max=blocks["dig_max"][i],
                prefiltering=blocks["prefiltering"][i],
                n_samples=int(blocks["n_samples"][i]),
                reserved=blocks["reserved"][i],
            )
        )
    return signals


def parse_header(f) -> EdfHeader:
    f.seek(0)
    main_raw = f.read(256)
    main = parse_main_header(main_raw)
    f.seek(256)
    sig_raw = f.read(main.n_header_bytes - 256)
    signals = parse_signal_headers(sig_raw, main.n_signals)
    f.seek(0)
    full_raw = f.read(main.n_header_bytes)
    return EdfHeader(main=main, signals=signals, raw_bytes=full_raw)


def record_bytes_per_signal(signals: list[SignalHeader]) -> list[int]:
    return [s.n_samples * 2 for s in signals]


def find_annotation_channels(signals: list[SignalHeader]) -> list[int]:
    return [
        i
        for i, s in enumerate(signals)
        if "EDF Annotations" in s.label or "BDF Annotations" in s.label
    ]


def parse_tal(raw: bytes) -> list[tuple[float, float, str]]:
    """Parse onset/duration/description triples from a TAL byte stream."""
    annotations: list[tuple[float, float, str]] = []
    for tal in raw.split(b"\x00"):
        if not tal:
            continue
        parts = tal.split(b"\x14")
        if not parts or not parts[0]:
            continue
        time_part = parts[0]
        if b"\x15" in time_part:
            t_bytes, d_bytes = time_part.split(b"\x15", 1)
        else:
            t_bytes, d_bytes = time_part, b"0"
        try:
            onset = float(t_bytes.decode("ascii", errors="replace").strip())
            duration = float(
                d_bytes.decode("ascii", errors="replace").strip() or "0"
            )
        except ValueError:
            continue
        for ann_bytes in parts[1:]:
            desc = ann_bytes.decode("utf-8", errors="replace").strip()
            if desc:
                annotations.append((onset, duration, desc))
    return annotations


def parse_edf_start(header: EdfHeader) -> Optional[datetime]:
    """Header startdate/starttime (dd.mm.yy + hh.mm.ss, whole seconds)."""
    try:
        d = header.main.startdate.strip()
        t = header.main.starttime.strip().replace(":", ".")
        day, month, year = (int(x) for x in d.split("."))
        if year < 100:
            year += 2000 if year < 70 else 1900
        hh, mm, ss = (int(x) for x in t.split("."))
        return datetime(year, month, day, hh, mm, ss)
    except Exception:
        return None


def _is_annot_label(label: str) -> bool:
    return "EDF Annotations" in label or "BDF Annotations" in label


def extract_edf_bytes(edf_path: str) -> FileExtract:
    """Stream annotation-channel bytes and parse TALs; gather INFO fields."""
    with open(edf_path, "rb") as fh:
        header = parse_header(fh)

    main = header.main
    signals = header.signals
    start_dt = parse_edf_start(header)

    channel_names = [s.label for s in signals]
    eeg_sigs = [s for s in signals if not _is_annot_label(s.label)]
    if eeg_sigs and main.record_dur > 0:
        sfreq = float(eeg_sigs[0].n_samples) / float(main.record_dur)
    else:
        sfreq = float("nan")

    annot_indices = find_annotation_channels(signals)
    rows: list[dict[str, Any]] = []
    if annot_indices and main.n_records > 0:
        sizes = record_bytes_per_signal(signals)
        rec_bytes = sum(sizes)
        annot_meta = [
            (idx, sum(sizes[:idx]), sizes[idx]) for idx in annot_indices
        ]
        channel_raw = {idx: bytearray() for idx, _, _ in annot_meta}

        with open(edf_path, "rb") as fh:
            fh.seek(main.n_header_bytes)
            for _ in range(main.n_records):
                record = fh.read(rec_bytes)
                if len(record) < rec_bytes:
                    break
                for idx, off, length in annot_meta:
                    channel_raw[idx] += record[off : off + length]

        # EDF+ wall clock: header startdate/time + raw TAL onset
        for idx, _, _ in annot_meta:
            for onset, dur, desc in parse_tal(bytes(channel_raw[idx])):
                if start_dt is not None:
                    onset_val: Any = start_dt + timedelta(seconds=float(onset))
                else:
                    onset_val = float(onset)
                rows.append(
                    {
                        "onset": onset_val,
                        "duration": float(dur),
                        "description": desc,
                    }
                )

    annot_df = pd.DataFrame(rows, columns=["onset", "duration", "description"])
    if not annot_df.empty and start_dt is not None:
        annot_df["onset"] = pd.to_datetime(annot_df["onset"])

    return FileExtract(
        annot_df=annot_df,
        start_dt=start_dt,
        number_of_channels=len(eeg_sigs) if eeg_sigs else main.n_signals,
        sampling_freq=sfreq,
        channel_names=channel_names,
    )


def _strip_onset_tz(onset: pd.Series) -> pd.Series:
    """Same intent as release ``.dt.tz_localize(None)``; safe if already naive."""
    if not pd.api.types.is_datetime64_any_dtype(onset):
        return onset
    if getattr(onset.dt, "tz", None) is not None:
        return onset.dt.tz_localize(None)
    return onset


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Annotation Extraction (byte TAL)")
    parser.add_argument(
        "-p", "--patient", type=str, required=True, help="Patient to run on"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate trending plot for each event type",
    )
    args = parser.parse_args()

    # Same wording / event logic as 00_extract_annotations.py
    annot_types = ["spike", "seizure", "sleep", "patient"]
    annot_wording_dict = {
        "spike": ["spike"],
        "seizure": ["ueo", "eec", "sz", "seizure", "off"],
        "sleep": [
            "n1",
            "n2",
            "n3",
            "sleep",
            "wake",
            "rem",
            "sleep",
            "drowsy",
            "arousal",
        ],
        "patient": ["patient event"],
    }
    if args.patient == "all":
        rids = os.listdir(raw_data_path)
    else:
        rids = args.patient.split(",")
    do_plot = args.plot

    for rid in rids:
        print(f"Processing patient {rid}")
        os.makedirs(os.path.join("annots", rid), exist_ok=True)
        os.makedirs("INFO", exist_ok=True)
        folder = os.path.join(raw_data_path, rid)
        files = os.listdir(folder)
        files = [f for f in files if f.endswith(".edf")]
        all_annots = []
        info = {}
        last_edf_name = None
        for e, f in enumerate(files):
            edf_file_path = os.path.join(folder, f)
            last_edf_name = f
            extracted = extract_edf_bytes(edf_file_path)
            annot_df = extracted.annot_df
            annot_path = f"annots/{rid}/{f.replace('.edf', '.csv')}"
            annot_df.to_csv(annot_path, index=False)
            all_annots.append(annot_df)

            # _______________extract info__________________
            tmp = re.split(r"[~ _.]", f)
            patient_name_first = tmp[2] if len(tmp) > 2 else ""
            patient_name_last = tmp[0] if len(tmp) > 0 else ""
            patient_guid = tmp[3] if len(tmp) > 3 else ""
            start_dt = extracted.start_dt
            if not info:
                info = {
                    "firstName": patient_name_first,
                    "lastName": patient_name_last,
                    "patientGuid": patient_guid,  # is actually file_id
                    "fileName": edf_file_path,
                    "studyType": "eeg",
                    "xlCreationTime": start_dt,
                    "numberOfChannels": extracted.number_of_channels,
                    "samplingFreq": extracted.sampling_freq,
                    "samplingFreqRaw": extracted.sampling_freq,
                    "channelNames": extracted.channel_names,
                }
                if (
                    info["samplingFreqRaw"] == info["samplingFreqRaw"]
                    and info["samplingFreqRaw"] > 512
                ):
                    info["studyType"] = "ieeg"
                    info["samplingFreq"] = 512
            else:
                if start_dt is not None and (
                    info["xlCreationTime"] is None
                    or start_dt <= info["xlCreationTime"]
                ):
                    info["xlCreationTime"] = start_dt

        with open(f"INFO/{rid}.pkl", "wb") as fh:
            pickle.dump(info, fh)

        if not all_annots:
            print("No annotations in any file.")
            continue
        all_annots = pd.concat(all_annots).sort_values(by="onset")
        for t in annot_types:
            judge = (
                all_annots["description"]
                .str.lower()
                .replace("speech", "")
                .apply(lambda x: any(word in x for word in annot_wording_dict[t]))
            )
            sub_df = all_annots[judge]
            if sub_df.empty:
                print(f"No {t} annotations.")
                continue
            if t == "seizure":
                sub_df = sub_df[sub_df["description"] != "Persyst SeizureDetection"]
                sub_df = sub_df[sub_df["description"] != "Video Recording OFF"]
                sub_df = sub_df[sub_df["description"] != "Photic: Off"]
            sub_df = sub_df.copy()
            sub_df["onset"] = _strip_onset_tz(sub_df["onset"])
            sub_df.to_csv(f"annots/{rid}/{t}.csv", index=False)
            if do_plot:
                figure_path = f"annots/{rid}/figures"
                os.makedirs(figure_path, exist_ok=True)
                fig, ax = plt.subplots(figsize=(10, 3))
                for i, row in sub_df.iterrows():
                    ax.axvline(row["onset"], color="k")
                    if t != "spike":
                        ax.text(
                            row["onset"],
                            0.5,
                            row["description"],
                            rotation=90,
                            fontsize=9,
                        )
                ax.set_ylim(0, 1.2)
                ax.set_yticks([])
                ax.set_xlabel("Time")
                ax.set_title(t)
                ax.xaxis_date()
                fig.autofmt_xdate()
                plt.tight_layout()
                plot_stem = last_edf_name or "annots"
                fig.savefig(
                    f"annots/{rid}/figures/{plot_stem.replace('.edf', '_' + t + '.png')}"
                )
                plt.close(fig)
            if t == "seizure":
                # Exact seizure-event logic from 00_extract_annotations.py
                seizure_annots = sub_df
                events = []
                current = None
                pending_ts = []  # store keywords that appear before OFF
                if not seizure_annots.empty:
                    for _, row in seizure_annots.iterrows():
                        comment = str(row["description"]).lower()
                        if "ueo" in comment or "eec" in comment:
                            pending_ts.append(row["onset"])
                            current = True

                        # OFF found
                        last_comment = comment.split(":")[-1].strip(" ")
                        if current and last_comment.startswith("off"):
                            # check gap from first keyword to OFF
                            stop_ts = row["onset"]
                            while True:
                                if (stop_ts - pending_ts[0]).total_seconds() / 60 > 20:
                                    e = {
                                        "Start": pending_ts[0],
                                        "Stop": pending_ts[0] + timedelta(minutes=2),
                                        "Comment": "No OFF found in 20 minutes",
                                    }
                                    events.append(e)
                                    pending_ts = [t for t in pending_ts if t > e["Stop"]]
                                    if not pending_ts:
                                        current = False
                                        break
                                    else:
                                        continue
                                else:
                                    e = {
                                        "Start": pending_ts[0],
                                        "Stop": stop_ts,
                                        "Comment": "",
                                    }
                                    events.append(e)
                                    pending_ts = []
                                    current = False
                                    break
                events = pd.DataFrame(events)
                if events.empty:
                    events = pd.DataFrame(columns=["Start", "Stop", "Comment"])
                events = events[events["Comment"] != "No OFF found in 20 minutes"]
                events.to_csv(f"annots/{rid}/seizure_events.csv", index=False)
            if t == "patient":
                # Exact patient-event logic from 00_extract_annotations.py
                patient_annots = sub_df
                events = []
                if not patient_annots.empty:
                    for _, row in patient_annots.iterrows():
                        events.append(
                            {
                                "Start": row["onset"] - timedelta(minutes=1),
                                "Stop": row["onset"] + timedelta(minutes=2),
                                "Comment": "-1-2min from patient reporting",
                            }
                        )
                events = pd.DataFrame(events)
                if events.empty:
                    events = pd.DataFrame(columns=["Start", "Stop", "Comment"])
                events.to_csv(f"annots/{rid}/patient_events.csv", index=False)
