#!/usr/bin/env python3

# -*- coding: utf-8 -*-

'''
temporal_window_pipeline.py

Pipeline para transformar datasets tabulares de APT baseados em flows e/ou logs
(e.g. DAPT2020, Unraveled) em um dataset temporal agregado por janelas fixas.

Foco: construção correta do novo dataset temporal.
NÃO realiza treinamento, NÃO gera gráficos, NÃO é notebook.

Lê um CSV, normaliza timestamps, detecta flow x evento pontual, expande flows
que cruzam múltiplas janelas (repartição proporcional de bytes/packets),
agrega por entidade+janela, gera atributos temporais (deltas, rolling, novidade
histórica, entropias, persistência, labels) e exporta CSV (+ opcional Parquet).

Autor: pipeline para pesquisa em segurança de redes / APT.
'''

from **future** import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any

import numpy as np
import pandas as pd

# =============================================================================

# Colunas canônicas esperadas (schema interno).

# O dataset real não precisa ter todas; o que faltar é simplesmente ignorado.

# =============================================================================

CANONICAL_COLUMNS = ['start_time', 'end_time', 'timestamp',
                      'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
                      'bytes', 'packets', 'duration',
                      'label', 'attack_stage',
                      'host', 'event_type', 'user', 'process',
                     ]

# Tolerância para teste de preservação de massa (bytes/packets).

MASS_RELATIVE_TOLERANCE = 1e-6
MASS_ABSOLUTE_TOLERANCE = 1.0  # 1 unidade absoluta é aceitável por arredondamento

# =============================================================================

# 1. Carga e schema

# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    '''Lê um CSV de entrada de forma tolerante.'''
    if not sys.path.exists():
        raise FileNotFoundError(f”Arquivo de entrada não encontrado: {path}”)
    # low_memory=False evita inferência parcial problemática em CSVs grandes.
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f”[load_data] Lidas {len(df)} linhas e {len(df.columns)} colunas de {path}”)
    return df

def apply_schema_mapping(df: pd.DataFrame,
    mapping_path: Optional[Path]) -> pd.DataFrame:
    '''
    Renomeia colunas do dataset para os nomes canônicos, usando um JSON do tipo:

    ```
        { "start_time": "flow_start", "bytes": "tot_bytes", ... }

    Chaves = nomes canônicos (alvo). Valores = nomes no dataset original (origem).
    '''
    if mapping_path is None:
        return df
    if not mapping_path.exists():
        raise FileNotFoundError(f"Arquivo de mapeamento não encontrado: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Inverte: {origem_no_dataset: nome_canonico}
    rename_dict = {}
    for canonical, original in mapping.items():
        if canonical not in CANONICAL_COLUMNS:
            print(f"[schema_mapping] Aviso: chave '{canonical}' não é canônica. Ignorada.")
            continue
        if original in df.columns:
            rename_dict[original] = canonical
        else:
            print(f"[schema_mapping] Aviso: coluna '{original}' não existe no dataset.")

    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f"[schema_mapping] Colunas renomeadas: {rename_dict}")
    return df

# =============================================================================

# 2. Normalização temporal

# =============================================================================

def normalize_timestamps(df: pd.DataFrame,
timezone: Optional[str] = None) -> pd.DataFrame:
“””
Converte start_time, end_time e timestamp para datetime64[ns] (tz-aware se
`timezone` fornecido). Usa pd.to_datetime com errors=‘coerce’.
“””
time_cols = [c for c in (“start_time”, “end_time”, “timestamp”) if c in df.columns]
for col in time_cols:
df[col] = pd.to_datetime(df[col], errors=“coerce”, utc=(timezone is not None))
if timezone is not None:
try:
df[col] = df[col].dt.tz_convert(timezone)
except Exception:
# já é naive ou não convertível — mantém como está
pass

```
# Se só existe 'timestamp' mas não start/end, segue. Se existe start sem end,
# assume flow instantâneo => end=start.
if "start_time" in df.columns and "end_time" not in df.columns:
    df["end_time"] = df["start_time"]
    print("[normalize_timestamps] 'end_time' ausente. Assumindo end_time == start_time.")

# Se só existe timestamp, não força start/end — será tratado como evento pontual.
# duration, se existir, deve ser numérico.
if "duration" in df.columns:
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

print(f"[normalize_timestamps] Colunas temporais normalizadas: {time_cols}")
return df
```

# =============================================================================

# 3. Detecção flow x evento pontual

# =============================================================================

def detect_record_type(df: pd.DataFrame) -> pd.DataFrame:
“””
Adiciona a coluna booleana `_is_flow`:
- True  se start_time e end_time existem e são válidos
- False se é evento pontual (apenas timestamp válido)

```
Também constrói `_event_time` para eventos pontuais:
  _event_time = timestamp, se existir; senão start_time.
"""
has_start = "start_time" in df.columns
has_end = "end_time" in df.columns
has_ts = "timestamp" in df.columns

if not (has_start or has_ts):
    raise ValueError(
        "Dataset precisa ter ao menos 'start_time' ou 'timestamp'."
    )

if has_start and has_end:
    flow_mask = df["start_time"].notna() & df["end_time"].notna()
else:
    flow_mask = pd.Series(False, index=df.index)

df["_is_flow"] = flow_mask

# Tempo de referência para eventos pontuais
if has_ts:
    df["_event_time"] = df["timestamp"]
else:
    df["_event_time"] = pd.NaT

# Fallback: se _event_time é NaT e temos start_time, usa start_time
if has_start:
    fallback = df["_event_time"].isna() & df["start_time"].notna()
    df.loc[fallback, "_event_time"] = df.loc[fallback, "start_time"]

n_flows = int(df["_is_flow"].sum())
n_events = int(len(df) - n_flows)
print(f"[detect_record_type] flows={n_flows}  eventos_pontuais={n_events}")
return df
```

# =============================================================================

# 4. Identificador de entidade

# =============================================================================

def build_entity_id(df: pd.DataFrame, entity_kind: str) -> pd.DataFrame:
“””
Adiciona a coluna `entity_id` conforme `entity_kind`:

```
    host       -> usa 'host' se existir, senão 'src_ip'
    host_pair  -> 'src_ip->dst_ip' (direcionado)
    global     -> string fixa 'GLOBAL'

Registros sem informação suficiente para formar a entidade ficam com
entity_id = 'UNKNOWN'.
"""
if entity_kind == "host":
    if "host" in df.columns:
        df["entity_id"] = df["host"].astype("string")
    elif "src_ip" in df.columns:
        df["entity_id"] = df["src_ip"].astype("string")
    else:
        raise ValueError("entity=host requer coluna 'host' ou 'src_ip'.")
elif entity_kind == "host_pair":
    if "src_ip" not in df.columns or "dst_ip" not in df.columns:
        raise ValueError("entity=host_pair requer 'src_ip' e 'dst_ip'.")
    df["entity_id"] = (
        df["src_ip"].astype("string").fillna("?")
        + "->"
        + df["dst_ip"].astype("string").fillna("?")
    )
elif entity_kind == "global":
    df["entity_id"] = "GLOBAL"
else:
    raise ValueError(f"entity_kind desconhecido: {entity_kind}")

df["entity_id"] = df["entity_id"].fillna("UNKNOWN").astype(str)
n_entities = df["entity_id"].nunique()
print(f"[build_entity_id] entidade='{entity_kind}'  n_entidades_distintas={n_entities}")
return df
```

# =============================================================================

# 5. Expansão flow -> contribuições por janela

# =============================================================================

def _floor_to_window(ts: pd.Timestamp, window_ns: int) -> pd.Timestamp:
“”“Retorna o início da janela [w_start, w_start+window) que contém ts.”””
# Trabalha em nanosegundos inteiros para estabilidade numérica.
# Preserva tz se houver.
v = ts.value  # int ns
floored = (v // window_ns) * window_ns
return pd.Timestamp(floored, tz=ts.tz) if ts.tz is not None else pd.Timestamp(floored)

def expand_flows_to_windows(df_flows: pd.DataFrame,
window_seconds: int) -> pd.DataFrame:
“””
Para cada flow, gera uma linha por janela interceptada, com:
- window_start, window_end
- overlap_seconds
- bytes_window, packets_window (proporcionais ao overlap)
- flags: flow_started_here, flow_ended_here,
flow_crosses_window, flow_active_in_window

```
Se flow_duration == 0, trata como evento pontual (associa à janela de start).
"""
if df_flows.empty:
    return df_flows.assign(
        window_start=pd.NaT, window_end=pd.NaT,
        overlap_seconds=0.0,
        bytes_window=0.0, packets_window=0.0,
        flow_started_here=False, flow_ended_here=False,
        flow_crosses_window=False, flow_active_in_window=False,
    )

window_ns = int(window_seconds * 1_000_000_000)
bytes_col = "bytes" if "bytes" in df_flows.columns else None
packets_col = "packets" if "packets" in df_flows.columns else None

rows: List[Dict[str, Any]] = []

# Iteração em ndarray para desempenho razoável.
starts = df_flows["start_time"].values
ends = df_flows["end_time"].values
bytes_vals = (df_flows[bytes_col].astype(float).values
              if bytes_col else np.zeros(len(df_flows)))
packets_vals = (df_flows[packets_col].astype(float).values
                if packets_col else np.zeros(len(df_flows)))

# Demais colunas são copiadas por índice
base_records = df_flows.drop(
    columns=[c for c in (bytes_col, packets_col) if c is not None],
    errors="ignore",
).to_dict("records")

for i in range(len(df_flows)):
    s = pd.Timestamp(starts[i])
    e = pd.Timestamp(ends[i])
    if pd.isna(s) or pd.isna(e):
        continue
    if e < s:
        # inconsistência: trata como evento pontual em s
        e = s

    total_bytes = float(bytes_vals[i]) if not np.isnan(bytes_vals[i]) else 0.0
    total_packets = float(packets_vals[i]) if not np.isnan(packets_vals[i]) else 0.0
    flow_duration_s = (e.value - s.value) / 1e9  # em segundos

    base = base_records[i]

    if flow_duration_s <= 0:
        # Evento pontual: associa à janela de s.
        w_start = _floor_to_window(s, window_ns)
        w_end = w_start + pd.Timedelta(seconds=window_seconds)
        rec = dict(base)
        rec.update({
            "window_start": w_start,
            "window_end": w_end,
            "overlap_seconds": 0.0,
            "bytes_window": total_bytes,
            "packets_window": total_packets,
            "flow_started_here": True,
            "flow_ended_here": True,
            "flow_crosses_window": False,
            "flow_active_in_window": True,
        })
        rows.append(rec)
        continue

    # Itera janelas que interceptam [s, e)
    w_start = _floor_to_window(s, window_ns)
    first_window_start = w_start
    # Última janela a considerar: a que contém (e - epsilon). Como [w, w+W) é
    # semiaberta, se e coincide com w_end, essa janela NÃO é contabilizada.
    while w_start < e:
        w_end = w_start + pd.Timedelta(seconds=window_seconds)
        ov_start = max(s, w_start)
        ov_end = min(e, w_end)
        overlap_ns = ov_end.value - ov_start.value
        if overlap_ns <= 0:
            w_start = w_end
            continue
        overlap_s = overlap_ns / 1e9
        frac = overlap_s / flow_duration_s

        started_here = (s >= w_start) and (s < w_end)
        ended_here = (e > w_start) and (e <= w_end)
        # "Cruza" = flow estende-se além desta janela em alguma direção
        crosses = (s < w_start) or (e > w_end)

        rec = dict(base)
        rec.update({
            "window_start": w_start,
            "window_end": w_end,
            "overlap_seconds": overlap_s,
            "bytes_window": total_bytes * frac,
            "packets_window": total_packets * frac,
            "flow_started_here": bool(started_here),
            "flow_ended_here": bool(ended_here),
            "flow_crosses_window": bool(crosses),
            "flow_active_in_window": True,
        })
        rows.append(rec)
        w_start = w_end

out = pd.DataFrame(rows)
print(f"[expand_flows_to_windows] flows={len(df_flows)} -> "
      f"contribuições={len(out)}")
return out
```

# =============================================================================

# 6. Atribuição de eventos pontuais a janelas

# =============================================================================

def assign_events_to_windows(df_events: pd.DataFrame,
window_seconds: int) -> pd.DataFrame:
“””
Para cada evento pontual, anexa window_start, window_end e flags neutras.
bytes_window / packets_window preenchidos a partir de ‘bytes’/‘packets’ se
existirem; caso contrário, 0.
“””
if df_events.empty:
return df_events.assign(
window_start=pd.NaT, window_end=pd.NaT,
overlap_seconds=0.0,
bytes_window=0.0, packets_window=0.0,
flow_started_here=False, flow_ended_here=False,
flow_crosses_window=False, flow_active_in_window=False,
)

```
window_ns = int(window_seconds * 1_000_000_000)
out = df_events.copy()
times = out["_event_time"]
valid = times.notna()
w_starts = []
w_ends = []
for t in times:
    if pd.isna(t):
        w_starts.append(pd.NaT)
        w_ends.append(pd.NaT)
    else:
        ws = _floor_to_window(pd.Timestamp(t), window_ns)
        w_starts.append(ws)
        w_ends.append(ws + pd.Timedelta(seconds=window_seconds))
out["window_start"] = w_starts
out["window_end"] = w_ends
out["overlap_seconds"] = 0.0
out["bytes_window"] = (out["bytes"].astype(float)
                       if "bytes" in out.columns else 0.0)
out["packets_window"] = (out["packets"].astype(float)
                         if "packets" in out.columns else 0.0)
out["flow_started_here"] = False
out["flow_ended_here"] = False
out["flow_crosses_window"] = False
out["flow_active_in_window"] = False

# descarta linhas sem janela válida
out = out[valid].copy()
print(f"[assign_events_to_windows] eventos atribuídos a janelas: {len(out)}")
return out
```

# =============================================================================

# 7. Utilitários de entropia / direção

# =============================================================================

def _shannon_entropy(values: Iterable[Any]) -> float:
“”“Entropia de Shannon (log2) sobre contagens dos valores não nulos.”””
vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
if not vals:
return 0.0
counts = Counter(vals)
total = sum(counts.values())
if total <= 0:
return 0.0
ent = 0.0
for c in counts.values():
p = c / total
if p > 0:
ent -= p * math.log2(p)
return ent

def _top1_share(values: Iterable[Any]) -> float:
vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
if not vals:
return 0.0
counts = Counter(vals)
top = max(counts.values())
return top / sum(counts.values())

def _safe_div(a: float, b: float) -> float:
if b is None or b == 0 or (isinstance(b, float) and math.isnan(b)):
return 0.0
return a / b

def _is_in_direction(row: pd.Series, entity_kind: str) -> Optional[bool]:
“””
Decide se um registro é ‘in’ (entrada para a entidade) ou ‘out’ (saída).
- host: ‘out’ quando entity_id == src_ip; ‘in’ quando entity_id == dst_ip
- host_pair / global: direção sempre ‘out’ (toda a rede é a entidade)
— o ratio in/out não é significativo para essas entidades, mas o cálculo
não falha (dividirá por zero de forma segura).
Retorna True (in), False (out) ou None (desconhecido).
“””
if entity_kind != “host”:
return False  # tudo contabilizado como “out” pela perspectiva da entidade
sip = row.get(“src_ip”, None)
dip = row.get(“dst_ip”, None)
eid = row.get(“entity_id”, None)
if eid is not None and sip is not None and str(eid) == str(sip):
return False  # out
if eid is not None and dip is not None and str(eid) == str(dip):
return True   # in
return None

# =============================================================================

# 8. Agregação por entidade + janela

# =============================================================================

def aggregate_window_features(df_contrib: pd.DataFrame,
entity_kind: str,
profile: str,
available_cols: set) -> pd.DataFrame:
“””
Agrega as contribuições por (entity_id, window_start).

```
profile = 'minimal' ou 'full'.
available_cols controla quais famílias de features são calculadas.
"""
if df_contrib.empty:
    return pd.DataFrame()

# Direção por registro (apenas para entity=host faz sentido real)
df_contrib = df_contrib.copy()
df_contrib["_is_in"] = df_contrib.apply(
    lambda r: _is_in_direction(r, entity_kind), axis=1
)

groups = df_contrib.groupby(["entity_id", "window_start"], sort=True)

records: List[Dict[str, Any]] = []
for (eid, wstart), g in groups:
    rec: Dict[str, Any] = {
        "entity_id": eid,
        "window_start": wstart,
        "window_end": g["window_end"].iloc[0],
    }

    # ---- A. Atividade básica ----
    n_records = len(g)
    n_flows = int(g["_is_flow"].sum()) if "_is_flow" in g.columns else 0
    n_events = n_records - n_flows
    bytes_total = float(g["bytes_window"].sum()) if "bytes_window" in g.columns else 0.0
    packets_total = float(g["packets_window"].sum()) if "packets_window" in g.columns else 0.0

    rec.update({
        "n_records": n_records,
        "n_flows": n_flows,
        "n_events": n_events,
        "bytes_total": bytes_total,
        "packets_total": packets_total,
    })

    if profile == "full":
        # bytes_in / bytes_out
        in_mask = g["_is_in"] == True  # noqa: E712
        out_mask = g["_is_in"] == False  # noqa: E712
        rec["bytes_in"] = float(g.loc[in_mask, "bytes_window"].sum())
        rec["bytes_out"] = float(g.loc[out_mask, "bytes_window"].sum())
        rec["packets_in"] = float(g.loc[in_mask, "packets_window"].sum())
        rec["packets_out"] = float(g.loc[out_mask, "packets_window"].sum())
        rec["n_new_flows"] = int(g.get("flow_started_here", pd.Series(dtype=bool)).sum())
        rec["n_ended_flows"] = int(g.get("flow_ended_here", pd.Series(dtype=bool)).sum())
        rec["n_active_cross_window_flows"] = int(
            g.get("flow_crosses_window", pd.Series(dtype=bool)).sum()
        )
        # duração (apenas sobre flows originais, não contribuições parciais)
        if "_is_flow" in g.columns and "start_time" in g.columns and "end_time" in g.columns:
            flows_g = g[g["_is_flow"]].drop_duplicates(
                subset=[c for c in ("start_time", "end_time", "src_ip",
                                    "dst_ip", "src_port", "dst_port",
                                    "protocol") if c in g.columns]
            )
            if not flows_g.empty:
                durs = (flows_g["end_time"].astype("int64")
                        - flows_g["start_time"].astype("int64")) / 1e9
                rec["avg_flow_duration"] = float(durs.mean())
                rec["max_flow_duration"] = float(durs.max())
            else:
                rec["avg_flow_duration"] = 0.0
                rec["max_flow_duration"] = 0.0
        else:
            rec["avg_flow_duration"] = 0.0
            rec["max_flow_duration"] = 0.0

        # ---- B. Diversidade ----
        if "src_ip" in available_cols:
            rec["unique_src_ip"] = g["src_ip"].nunique(dropna=True)
        if "dst_ip" in available_cols:
            rec["unique_dst_ip"] = g["dst_ip"].nunique(dropna=True)
        if "src_port" in available_cols:
            rec["unique_src_port"] = g["src_port"].nunique(dropna=True)
        if "dst_port" in available_cols:
            rec["unique_dst_port"] = g["dst_port"].nunique(dropna=True)
        if "protocol" in available_cols:
            rec["unique_protocols"] = g["protocol"].nunique(dropna=True)

        # ---- D. Dispersão / concentração ----
        if "dst_ip" in available_cols:
            rec["dst_ip_entropy"] = _shannon_entropy(g["dst_ip"].tolist())
            rec["top1_dst_ip_share"] = _top1_share(g["dst_ip"].tolist())
        if "dst_port" in available_cols:
            rec["dst_port_entropy"] = _shannon_entropy(g["dst_port"].tolist())
            rec["top1_dst_port_share"] = _top1_share(g["dst_port"].tolist())
        if "protocol" in available_cols:
            rec["protocol_entropy"] = _shannon_entropy(g["protocol"].tolist())

        # ---- G. Direção / assimetria ----
        rec["in_out_byte_ratio"] = _safe_div(rec["bytes_in"], rec["bytes_out"])
        rec["in_out_packet_ratio"] = _safe_div(rec["packets_in"], rec["packets_out"])
        total_bp = rec["bytes_in"] + rec["bytes_out"]
        rec["egress_dominance"] = _safe_div(rec["bytes_out"], total_bp)

        # ---- H. Host log features ----
        if "user" in available_cols:
            rec["unique_users"] = g["user"].nunique(dropna=True)
        if "process" in available_cols:
            rec["unique_processes"] = g["process"].nunique(dropna=True)
        if "event_type" in available_cols:
            rec["event_type_entropy"] = _shannon_entropy(g["event_type"].tolist())

        # Conjuntos de peers / destinos desta janela (p/ persistência e novidade)
        rec["_dst_ip_set"] = set(g["dst_ip"].dropna().astype(str).tolist()) \
            if "dst_ip" in available_cols else set()
        rec["_dst_port_set"] = set(g["dst_port"].dropna().astype(str).tolist()) \
            if "dst_port" in available_cols else set()
        rec["_protocol_set"] = set(g["protocol"].dropna().astype(str).tolist()) \
            if "protocol" in available_cols else set()
        rec["_user_set"] = set(g["user"].dropna().astype(str).tolist()) \
            if "user" in available_cols else set()
        rec["_process_set"] = set(g["process"].dropna().astype(str).tolist()) \
            if "process" in available_cols else set()

    else:
        # ---- Perfil minimal ----
        # bytes_out — sem direção real em host_pair/global, equivale a bytes_total
        in_mask = g["_is_in"] == True  # noqa: E712
        out_mask = g["_is_in"] == False  # noqa: E712
        bytes_in = float(g.loc[in_mask, "bytes_window"].sum())
        bytes_out = float(g.loc[out_mask, "bytes_window"].sum())
        rec["bytes_out"] = bytes_out
        rec["bytes_in"] = bytes_in
        rec["in_out_byte_ratio"] = _safe_div(bytes_in, bytes_out)
        total_bp = bytes_in + bytes_out
        rec["egress_dominance"] = _safe_div(bytes_out, total_bp)

        if "dst_ip" in available_cols:
            rec["unique_dst_ip"] = g["dst_ip"].nunique(dropna=True)
            rec["dst_ip_entropy"] = _shannon_entropy(g["dst_ip"].tolist())
            rec["_dst_ip_set"] = set(g["dst_ip"].dropna().astype(str).tolist())
        else:
            rec["_dst_ip_set"] = set()
        if "dst_port" in available_cols:
            rec["unique_dst_port"] = g["dst_port"].nunique(dropna=True)

    # ---- I. Labels agregadas ----
    if "label" in available_cols:
        # Heurística: qualquer valor diferente de 0/'benign'/'normal'/'' é malicioso.
        lbl = g["label"]
        def _is_mal(v):
            if pd.isna(v):
                return False
            s = str(v).strip().lower()
            if s in ("0", "benign", "normal", "none", "", "false"):
                return False
            return True
        mal_mask = lbl.apply(_is_mal)
        rec["malicious_record_count"] = int(mal_mask.sum())
        rec["malicious_record_ratio"] = _safe_div(
            float(mal_mask.sum()), float(len(mal_mask))
        )
        rec["label_any_malicious"] = bool(mal_mask.any())

    # ---- J. Attack stage ----
    if profile == "full" and "attack_stage" in available_cols:
        stages = g["attack_stage"].dropna().astype(str).tolist()
        if stages:
            rec["stage_mode"] = Counter(stages).most_common(1)[0][0]
            rec["stage_set_size"] = len(set(stages))
        else:
            rec["stage_mode"] = None
            rec["stage_set_size"] = 0

    records.append(rec)

out = pd.DataFrame(records)
out = out.sort_values(["entity_id", "window_start"]).reset_index(drop=True)
print(f"[aggregate_window_features] amostras temporais geradas: {len(out)}")
return out
```

# =============================================================================

# 9. Novidade histórica por entidade

# =============================================================================

def compute_historical_novelty(df_agg: pd.DataFrame,
profile: str,
available_cols: set) -> pd.DataFrame:
“””
Para cada entidade, percorre janelas em ordem cronológica, mantendo um
conjunto acumulado do que já foi visto. Para cada janela calcula:
new_dst_ip_count, new_dst_port_count, new_protocol_count,
new_user_count, new_process_count, new_peer_ratio,
time_since_last_new_dst (em segundos).
“””
if df_agg.empty:
return df_agg

```
df_agg = df_agg.sort_values(["entity_id", "window_start"]).reset_index(drop=True)

# Estado por entidade
seen_dst_ip: Dict[str, set] = defaultdict(set)
seen_dst_port: Dict[str, set] = defaultdict(set)
seen_protocol: Dict[str, set] = defaultdict(set)
seen_user: Dict[str, set] = defaultdict(set)
seen_process: Dict[str, set] = defaultdict(set)
last_new_dst_time: Dict[str, Optional[pd.Timestamp]] = defaultdict(lambda: None)

new_dst_ip_count = []
new_dst_port_count = []
new_protocol_count = []
new_user_count = []
new_process_count = []
new_peer_ratio = []
time_since_last_new_dst = []

for _, row in df_agg.iterrows():
    eid = row["entity_id"]
    wstart = row["window_start"]

    dst_ips = row.get("_dst_ip_set", set()) or set()
    dst_ports = row.get("_dst_port_set", set()) or set()
    protos = row.get("_protocol_set", set()) or set()
    users = row.get("_user_set", set()) or set()
    procs = row.get("_process_set", set()) or set()

    new_ips = dst_ips - seen_dst_ip[eid]
    new_ports = dst_ports - seen_dst_port[eid]
    new_protos = protos - seen_protocol[eid]
    new_users = users - seen_user[eid]
    new_procs = procs - seen_process[eid]

    n_new_ip = len(new_ips)
    n_new_port = len(new_ports)
    n_new_proto = len(new_protos)
    n_new_user = len(new_users)
    n_new_proc = len(new_procs)

    total_peers = len(dst_ips)
    npr = _safe_div(float(n_new_ip), float(total_peers)) if total_peers > 0 else 0.0

    # time_since_last_new_dst
    if n_new_ip > 0:
        last_new_dst_time[eid] = wstart
        tsl = 0.0
    else:
        prev = last_new_dst_time[eid]
        if prev is None:
            tsl = np.nan  # nunca houve novidade ainda
        else:
            tsl = (wstart.value - prev.value) / 1e9

    # Atualiza acumulados
    seen_dst_ip[eid] |= dst_ips
    seen_dst_port[eid] |= dst_ports
    seen_protocol[eid] |= protos
    seen_user[eid] |= users
    seen_process[eid] |= procs

    new_dst_ip_count.append(n_new_ip)
    new_dst_port_count.append(n_new_port)
    new_protocol_count.append(n_new_proto)
    new_user_count.append(n_new_user)
    new_process_count.append(n_new_proc)
    new_peer_ratio.append(npr)
    time_since_last_new_dst.append(tsl)

if "dst_ip" in available_cols:
    df_agg["new_dst_ip_count"] = new_dst_ip_count
    df_agg["new_peer_ratio"] = new_peer_ratio
    df_agg["time_since_last_new_dst"] = time_since_last_new_dst
if profile == "full":
    if "dst_port" in available_cols:
        df_agg["new_dst_port_count"] = new_dst_port_count
    if "protocol" in available_cols:
        df_agg["new_protocol_count"] = new_protocol_count
    if "user" in available_cols:
        df_agg["new_user_count"] = new_user_count
    if "process" in available_cols:
        df_agg["new_process_count"] = new_process_count

# Limpa colunas auxiliares
for aux in ("_dst_ip_set", "_dst_port_set", "_protocol_set",
            "_user_set", "_process_set"):
    if aux in df_agg.columns:
        df_agg = df_agg.drop(columns=[aux])

print("[compute_historical_novelty] OK")
return df_agg
```

# =============================================================================

# 10. Deltas entre janelas consecutivas

# =============================================================================

def compute_temporal_deltas(df_agg: pd.DataFrame, profile: str) -> pd.DataFrame:
“””
delta_X[t] = X[t] - X[t-1], por entidade, ordenado por tempo.
“””
if df_agg.empty:
return df_agg

```
df_agg = df_agg.sort_values(["entity_id", "window_start"]).reset_index(drop=True)
g = df_agg.groupby("entity_id", sort=False)

delta_cols = ["bytes_total", "n_flows"]
if "bytes_out" in df_agg.columns:
    delta_cols.append("bytes_out")
if profile == "full" and "unique_dst_ip" in df_agg.columns:
    delta_cols.append("unique_dst_ip")

for col in delta_cols:
    if col in df_agg.columns:
        df_agg[f"delta_{col}"] = g[col].diff().fillna(0.0)

print(f"[compute_temporal_deltas] deltas calculados para: {delta_cols}")
return df_agg
```

# =============================================================================

# 11. Rolling features

# =============================================================================

def compute_rolling_features(df_agg: pd.DataFrame,
profile: str,
rolling_size: int = 5) -> pd.DataFrame:
“””
Rolling mean/std sobre janelas consecutivas da mesma entidade:
rolling_mean_bytes_total, rolling_std_bytes_total,
rolling_mean_n_flows,
persistence_peer_count (# janelas recentes em que unique_dst_ip>0),
time_since_last_high_bytes_out (segundos desde último pico > mediana global).
“””
if df_agg.empty:
return df_agg

```
df_agg = df_agg.sort_values(["entity_id", "window_start"]).reset_index(drop=True)

# rolling_mean / std por entidade
def _rolling(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    if "bytes_total" in g.columns:
        g["rolling_mean_bytes_total"] = (
            g["bytes_total"].rolling(rolling_size, min_periods=1).mean()
        )
        g["rolling_std_bytes_total"] = (
            g["bytes_total"].rolling(rolling_size, min_periods=1).std().fillna(0.0)
        )
    if "n_flows" in g.columns:
        g["rolling_mean_n_flows"] = (
            g["n_flows"].rolling(rolling_size, min_periods=1).mean()
        )
    # persistence_peer_count: nº de janelas com unique_dst_ip>0 nos últimos K
    if "unique_dst_ip" in g.columns:
        active = (g["unique_dst_ip"] > 0).astype(int)
        g["persistence_peer_count"] = (
            active.rolling(rolling_size, min_periods=1).sum()
        )
    return g

# Calcula rolling por entidade sem usar groupby.apply (que em algumas versões
# do pandas consome a chave do grupo como índice e dificulta o reset).
out_parts: List[pd.DataFrame] = []
for eid, g in df_agg.groupby("entity_id", sort=False):
    g2 = _rolling(g.copy())
    out_parts.append(g2)
df_agg = pd.concat(out_parts, ignore_index=True)
df_agg = df_agg.sort_values(["entity_id", "window_start"]).reset_index(drop=True)

# time_since_last_high_bytes_out: usa mediana global de bytes_out como limiar
if "bytes_out" in df_agg.columns:
    thr = float(df_agg["bytes_out"].median())
    last_high: Dict[str, Optional[pd.Timestamp]] = defaultdict(lambda: None)
    out = []
    for _, row in df_agg.iterrows():
        eid = row["entity_id"]
        wstart = row["window_start"]
        val = row["bytes_out"]
        if pd.notna(val) and val > thr:
            last_high[eid] = wstart
            out.append(0.0)
        else:
            prev = last_high[eid]
            if prev is None:
                out.append(np.nan)
            else:
                out.append((wstart.value - prev.value) / 1e9)
    df_agg["time_since_last_high_bytes_out"] = out

print(f"[compute_rolling_features] rolling_size={rolling_size}")
return df_agg
```

# =============================================================================

# 12. Labels / estágio (placeholder; já calculados em aggregate)

# =============================================================================

def compute_labels_and_stage_features(df_agg: pd.DataFrame,
available_cols: set) -> pd.DataFrame:
“””
As principais colunas de label e stage já foram adicionadas em
aggregate_window_features. Aqui apenas garantimos tipos consistentes.
“””
if df_agg.empty:
return df_agg
if “label_any_malicious” in df_agg.columns:
df_agg[“label_any_malicious”] = df_agg[“label_any_malicious”].astype(bool)
if “malicious_record_count” in df_agg.columns:
df_agg[“malicious_record_count”] = df_agg[“malicious_record_count”].astype(int)
return df_agg

# =============================================================================

# 13. Validação de preservação de massa

# =============================================================================

def validate_mass_preservation(df_input: pd.DataFrame,
df_contrib: pd.DataFrame) -> None:
“”“Confere se a soma total de bytes/packets foi preservada.”””
print(”\n=== Relatório de preservação de massa ===”)
for col in (“bytes”, “packets”):
if col not in df_input.columns:
continue
orig = float(pd.to_numeric(df_input[col], errors=“coerce”).fillna(0).sum())
agg_col = f”{col}_window”
if agg_col not in df_contrib.columns:
print(f”  {col}: coluna agregada ‘{agg_col}’ ausente”)
continue
agg = float(df_contrib[agg_col].fillna(0).sum())
diff = agg - orig
rel = abs(diff) / orig if orig != 0 else 0.0
status = “OK”
if abs(diff) > MASS_ABSOLUTE_TOLERANCE and rel > MASS_RELATIVE_TOLERANCE:
status = “ATENÇÃO: discrepância acima da tolerância”
print(f”  {col:8s}  original={orig:.3f}  agregado={agg:.3f}  “
f”diff_abs={diff:.6f}  diff_rel={rel:.3e}  [{status}]”)
print(”=========================================\n”)

# =============================================================================

# 14. Exportação

# =============================================================================

def export_output(df_agg: pd.DataFrame,
out_csv: Path,
also_parquet: bool = True) -> None:
out_csv.parent.mkdir(parents=True, exist_ok=True)
df_agg.to_csv(out_csv, index=False)
print(f”[export_output] CSV escrito em: {out_csv}  ({len(df_agg)} linhas)”)
if also_parquet:
try:
parquet_path = out_csv.with_suffix(”.parquet”)
df_agg.to_parquet(parquet_path, index=False)
print(f”[export_output] Parquet escrito em: {parquet_path}”)
except Exception as e:
print(f”[export_output] Parquet não gerado ({e}). Seguindo só com CSV.”)

# =============================================================================

# 15. Main

# =============================================================================

def parse_args() -> argparse.Namespace:
p = argparse.ArgumentParser(
description=“Pipeline de janelamento temporal para datasets APT.”
)
p.add_argument(”–input”, required=True, type=Path,
help=“CSV de entrada.”)
p.add_argument(”–output”, required=True, type=Path,
help=“CSV de saída (Parquet será gerado ao lado, se possível).”)
p.add_argument(”–window-seconds”, type=int, default=10,
help=“Tamanho da janela em segundos. Default: 10.”)
p.add_argument(”–entity”, choices=[“host”, “host_pair”, “global”],
default=“host”, help=“Entidade temporal.”)
p.add_argument(”–profile”, choices=[“minimal”, “full”],
default=“full”, help=“Perfil de features.”)
p.add_argument(”–schema”, type=Path, default=None,
help=“JSON opcional com mapeamento de colunas.”)
p.add_argument(”–timezone”, type=str, default=None,
help=“Timezone, e.g. ‘UTC’ ou ‘America/Sao_Paulo’.”)
p.add_argument(”–rolling-size”, type=int, default=5,
help=“Tamanho da janela rolling para features de memória curta.”)
p.add_argument(”–no-parquet”, action=“store_true”,
help=“Não tentar escrever Parquet.”)
return p.parse_args()

def main() -> int:
args = parse_args()

```
print("=" * 70)
print("PIPELINE TEMPORAL APT — parâmetros")
print("=" * 70)
print(f"  input           = {args.input}")
print(f"  output          = {args.output}")
print(f"  window_seconds  = {args.window_seconds}")
print(f"  entity          = {args.entity}")
print(f"  profile         = {args.profile}")
print(f"  schema          = {args.schema}")
print(f"  timezone        = {args.timezone}")
print(f"  rolling_size    = {args.rolling_size}")
print("=" * 70)

# 1. Load
df = load_data(args.input)
n_input = len(df)

# 2. Schema mapping
df = apply_schema_mapping(df, args.schema)

available_cols = set(df.columns)
print(f"[main] Colunas canônicas presentes: "
      f"{sorted(available_cols & set(CANONICAL_COLUMNS))}")
missing = set(CANONICAL_COLUMNS) - available_cols
if missing:
    print(f"[main] Colunas canônicas ausentes (serão ignoradas): {sorted(missing)}")

# 3. Normalização temporal
df = normalize_timestamps(df, timezone=args.timezone)

# 4. Detecção de tipo
df = detect_record_type(df)

# 5. Entidade
df = build_entity_id(df, args.entity)

# 6. Expansão
df_flows = df[df["_is_flow"]].copy()
df_events = df[~df["_is_flow"]].copy()

df_flow_contrib = expand_flows_to_windows(df_flows, args.window_seconds)
df_event_contrib = assign_events_to_windows(df_events, args.window_seconds)

# União das contribuições
common_cols = sorted(set(df_flow_contrib.columns) | set(df_event_contrib.columns))
for c in common_cols:
    if c not in df_flow_contrib.columns:
        df_flow_contrib[c] = np.nan
    if c not in df_event_contrib.columns:
        df_event_contrib[c] = np.nan
df_contrib = pd.concat(
    [df_flow_contrib[common_cols], df_event_contrib[common_cols]],
    ignore_index=True,
)
# Garante _is_flow como bool
if "_is_flow" in df_contrib.columns:
    df_contrib["_is_flow"] = df_contrib["_is_flow"].fillna(False).astype(bool)

n_contrib = len(df_contrib)
print(f"[main] contribuições totais após expansão: {n_contrib}")

# Validação de massa
validate_mass_preservation(df, df_contrib)

# 7. Agregação por entidade + janela
df_agg = aggregate_window_features(df_contrib, args.entity,
                                   args.profile, available_cols)

# 8. Novidade histórica por entidade
df_agg = compute_historical_novelty(df_agg, args.profile, available_cols)

# 9. Deltas
df_agg = compute_temporal_deltas(df_agg, args.profile)

# 10. Rolling features
df_agg = compute_rolling_features(df_agg, args.profile, args.rolling_size)

# 11. Labels / stage
df_agg = compute_labels_and_stage_features(df_agg, available_cols)

# 12. Exporta
export_output(df_agg, args.output, also_parquet=(not args.no_parquet))

# 13. Relatório final
print("\n" + "=" * 70)
print("RESUMO")
print("=" * 70)
print(f"  Registros de entrada            : {n_input}")
print(f"  Contribuições pós-expansão      : {n_contrib}")
print(f"  Amostras temporais geradas      : {len(df_agg)}")
if not df_agg.empty:
    t_min = df_agg["window_start"].min()
    t_max = df_agg["window_end"].max()
    print(f"  Intervalo temporal coberto      : {t_min}  ->  {t_max}")
print(f"  Tamanho da janela (segundos)    : {args.window_seconds}")
print(f"  Entidade                        : {args.entity}")
print(f"  Perfil de features              : {args.profile}")
print(f"  Colunas geradas ({len(df_agg.columns)}):")
for c in df_agg.columns:
    print(f"     - {c}")
print("=" * 70)

return 0
```

if **name** == “**main**”:
sys.exit(main())