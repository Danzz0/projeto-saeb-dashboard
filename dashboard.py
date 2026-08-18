import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data/processed/base_analitica.parquet"
st.set_page_config(page_title="Desigualdade educacional — SAEB", layout="wide")
st.title("Desigualdade educacional no Brasil — 9º ano")
st.caption("SAEB + Censo Escolar | associações observacionais, não causais | escores na escala SAEB (0-500)")
if not DATA.exists():
    st.error("Execute primeiro: python pipeline.py --config config.json")
    st.stop()
df = pd.read_parquet(DATA)

with st.sidebar:
    st.header("Filtros")
    for col, label in [("region", "Região"), ("state", "Estado"), ("urban_rural", "Localização"), ("school_network", "Rede")]:
        if col in df.columns and df[col].notna().any():
            options = sorted(df[col].dropna().astype(str).unique())
            chosen = st.multiselect(label, options, default=options)
            if chosen:
                df = df[df[col].astype(str).isin(chosen)]

group_order = ["Muito baixo", "Baixo", "Médio", "Alto", "Muito alto"]

c1, c2, c3 = st.columns(3)
c1.metric("Escolas na base", f"{len(df):,}".replace(",", "."))
if "math" in df: c2.metric("Matemática média", f"{df['math'].mean():.1f}")
if "portuguese" in df: c3.metric("Português médio", f"{df['portuguese'].mean():.1f}")

left, right = st.columns(2)
with left:
    if {"socioeconomic_group", "math"}.issubset(df.columns):
        chart = df.groupby("socioeconomic_group", observed=True, as_index=False)["math"].mean()
        st.plotly_chart(px.bar(chart, x="socioeconomic_group", y="math", category_orders={"socioeconomic_group": group_order}, title="Matemática por nível socioeconômico"), use_container_width=True)
with right:
    if {"region", "math"}.issubset(df.columns) and df["region"].nunique() > 1:
        chart = df.groupby("region", as_index=False)["math"].mean()
        st.plotly_chart(px.bar(chart, x="region", y="math", title="Matemática por região"), use_container_width=True)

if {"socioeconomic", "math"}.issubset(df.columns):
    color = "region" if "region" in df.columns else None
    hover = [c for c in ["state", "urban_rural", "school_network"] if c in df]
    st.plotly_chart(px.scatter(df, x="socioeconomic", y="math", color=color, hover_data=hover, title="Nível socioeconômico e proficiência em Matemática"), use_container_width=True)

left, right = st.columns(2)
with left:
    if {"socioeconomic_group", "portuguese"}.issubset(df.columns):
        chart_port = df.groupby("socioeconomic_group", observed=True, as_index=False)["portuguese"].mean()
        st.plotly_chart(px.bar(chart_port, x="socioeconomic_group", y="portuguese", category_orders={"socioeconomic_group": group_order}, title="Português por nível socioeconômico"), use_container_width=True)
with right:
    if {"region", "portuguese"}.issubset(df.columns) and df["region"].nunique() > 1:
        chart = df.groupby("region", as_index=False)["portuguese"].mean()
        st.plotly_chart(px.bar(chart, x="region", y="portuguese", title="Português por região"), use_container_width=True)

if {"socioeconomic", "portuguese"}.issubset(df.columns):
    color = "region" if "region" in df.columns else None
    hover = [c for c in ["state", "urban_rural", "school_network"] if c in df]
    st.plotly_chart(px.scatter(df, x="socioeconomic", y="portuguese", color=color, hover_data=hover, title="Nível socioeconômico e proficiência em Português"), use_container_width=True)

st.subheader("Níveis de proficiência do SAEB")
st.caption("Percentual médio de alunos por nível de proficiência (escala SAEB), agregado por escola.")
for subject, subject_label in [("math", "Matemática"), ("portuguese", "Português")]:
    level_cols = sorted(
        [c for c in df.columns if re.match(rf"^{subject}_nivel_\d+_pct$", c)],
        key=lambda c: int(re.search(r"\d+", c).group()),
    )
    if not level_cols or "socioeconomic_group" not in df.columns:
        continue
    long = (
        df.groupby("socioeconomic_group", observed=True)[level_cols]
        .mean()
        .reindex(group_order)
        .reset_index()
        .melt(id_vars="socioeconomic_group", var_name="nivel", value_name="pct")
    )
    long["nivel"] = long["nivel"].str.extract(r"(\d+)").astype(int)
    st.plotly_chart(
        px.bar(
            long, x="socioeconomic_group", y="pct", color="nivel",
            category_orders={"socioeconomic_group": group_order},
            color_continuous_scale="RdYlGn",
            title=f"Distribuição de níveis de proficiência em {subject_label} por nível socioeconômico",
            labels={"pct": "% de alunos", "socioeconomic_group": "Nível socioeconômico", "nivel": "Nível SAEB"},
        ),
        use_container_width=True,
    )

st.subheader("Base filtrada (por escola)")
st.dataframe(df, use_container_width=True, hide_index=True)
