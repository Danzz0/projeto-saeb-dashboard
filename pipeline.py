"""Pipeline reproduzível para SAEB + Censo Escolar (9o ano do Ensino Fundamental).

Uso: python pipeline.py --config config.json

Observação importante sobre os dados de origem: ID_ESCOLA/ID_MUNICIPIO em
data/raw/saeb.csv não usam o mesmo espaço de códigos que CO_ENTIDADE/CO_MUNICIPIO
em data/raw/censo_escolar.csv (interseção vazia nos dois casos). O único código
compatível entre os dois arquivos é o de UF (ID_UF/CO_UF). Por isso o Censo
Escolar é usado apenas para montar uma tabela pequena de código -> nome de
UF/região, aplicada sobre os códigos que o próprio SAEB já traz — não há
cruzamento linha a linha por escola com o Censo neste conjunto de dados.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)

TEXT_ENCODINGS = ("utf-8", "latin1", "cp1252")
NUMERIC_COLUMNS = [
    "math", "portuguese", "socioeconomic",
    "weight_math", "weight_portuguese", "weight_inse",
]
CODE_COLUMNS = ["school_id", "grade", "year", "region_code", "state_code", "urban_rural_code", "school_network_code"]
CATEGORICAL_COLUMNS = ["region", "state", "urban_rural", "school_network"]


def norm_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def sniff_header(path: Path) -> tuple[list[str], str, str]:
    """Return (column_names, separator, encoding) by reading only the first line."""
    for encoding in TEXT_ENCODINGS:
        try:
            with path.open("r", encoding=encoding) as fh:
                header_line = fh.readline().rstrip("\n\r")
            sep = max((";", ",", "\t", "|"), key=header_line.count)
            return header_line.split(sep), sep, encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8/latin1/cp1252", b"", 0, 1, f"Não foi possível decodificar {path}")


def resolve_column_names(available: list[str], aliases: dict[str, list[str]], label: str) -> dict[str, str]:
    lookup = {norm_name(c): c for c in available}
    resolved = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if norm_name(candidate) in lookup:
                resolved[canonical] = lookup[norm_name(candidate)]
                break
    LOG.info("%s: colunas encontradas: %s", label, resolved)
    return resolved


def load_and_prepare(path: Path, aliases: dict[str, list[str]], label: str) -> pd.DataFrame:
    """Load only the columns needed (fast path for large CSVs) and rename to canonical names."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        columns, sep, encoding = sniff_header(path)
        resolved = resolve_column_names(columns, aliases, label)
        usecols = list(dict.fromkeys(resolved.values()))
        raw = pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols, dtype="string", low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
        resolved = resolve_column_names(list(raw.columns), aliases, label)
        raw = raw[list(dict.fromkeys(resolved.values()))].astype("string")
    elif suffix == ".parquet":
        raw = pd.read_parquet(path)
        resolved = resolve_column_names(list(raw.columns), aliases, label)
        raw = raw[list(dict.fromkeys(resolved.values()))].astype("string")
    else:
        raise ValueError(f"Formato não suportado: {suffix}")

    out = raw.rename(columns={actual: canonical for canonical, actual in resolved.items()})
    for key in CODE_COLUMNS:
        if key in out:
            out[key] = out[key].str.strip().str.replace(r"\.0$", "", regex=True)
    for key in NUMERIC_COLUMNS:
        if key in out:
            out[key] = pd.to_numeric(out[key].str.replace(",", ".", regex=False), errors="coerce")
    return out


def apply_code_labels(df: pd.DataFrame, code_labels: dict[str, dict[str, str]]) -> pd.DataFrame:
    out = df.copy()
    for column, mapping in code_labels.items():
        if column in out:
            out[column] = out[column].map(mapping).fillna(out[column])
    return out


def build_geo_lookup(censo_geo: pd.DataFrame) -> tuple[dict, dict]:
    state_lookup = {}
    region_lookup = {}
    if {"state_code", "state"}.issubset(censo_geo.columns):
        state_lookup = (
            censo_geo[["state_code", "state"]].dropna().drop_duplicates("state_code")
            .set_index("state_code")["state"].to_dict()
        )
    if {"region_code", "region"}.issubset(censo_geo.columns):
        region_lookup = (
            censo_geo[["region_code", "region"]].dropna().drop_duplicates("region_code")
            .set_index("region_code")["region"].to_dict()
        )
    LOG.info("Tabela UF->estado: %s códigos. Tabela código->região: %s códigos.", len(state_lookup), len(region_lookup))
    return state_lookup, region_lookup


def apply_geo_lookup(saeb: pd.DataFrame, state_lookup: dict, region_lookup: dict) -> pd.DataFrame:
    out = saeb.copy()
    if "state_code" in out:
        out["state"] = out["state_code"].map(state_lookup)
        unmatched = out["state"].isna().mean()
        if unmatched > 0:
            LOG.warning("%.1f%% dos alunos sem UF reconhecida na tabela do Censo.", 100 * unmatched)
    if "region_code" in out:
        out["region"] = out["region_code"].map(region_lookup)
    return out


def filter_grade(saeb: pd.DataFrame, target_grade: str) -> pd.DataFrame:
    if "grade" not in saeb or not target_grade:
        LOG.warning("Sem coluna de série/etapa; nenhum filtro de ano escolar foi aplicado.")
        return saeb
    before = len(saeb)
    out = saeb[saeb["grade"] == str(target_grade)].copy()
    LOG.info("Filtro de série (%s): %s de %s alunos mantidos.", target_grade, len(out), before)
    return out


def classify_levels(score: pd.Series, cutoffs: list[float]) -> pd.Series:
    bins = [-np.inf, *cutoffs, np.inf]
    labels = list(range(len(cutoffs) + 1))
    return pd.cut(score, bins=bins, labels=labels, right=False)


def weighted_mean(values: pd.Series, weights: pd.Series | None) -> float:
    valid = values.notna()
    if not valid.any():
        return np.nan
    if weights is not None:
        w = weights.where(valid).fillna(0)
        if (w > 0).any():
            return float(np.average(values[valid], weights=w[valid]))
    return float(values[valid].mean())


def most_common(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.mode().iat[0]


def aggregate_saeb(saeb: pd.DataFrame, level_counts: dict[str, int]) -> pd.DataFrame:
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in saeb]
    rows = []
    for school_id, group in saeb.groupby("school_id", dropna=False, sort=False):
        row = {"school_id": school_id, "n_observations": len(group)}
        row["math"] = weighted_mean(group["math"], group.get("weight_math"))
        row["portuguese"] = weighted_mean(group["portuguese"], group.get("weight_portuguese"))
        row["socioeconomic"] = weighted_mean(group["socioeconomic"], group.get("weight_inse"))
        for col in cat_cols:
            row[col] = most_common(group[col])
        for subject, n_levels in level_counts.items():
            level_col = f"{subject}_level"
            valid = group[level_col].dropna()
            denom = len(valid)
            for level in range(n_levels):
                share = float((valid == level).sum()) / denom * 100 if denom else np.nan
                row[f"{subject}_nivel_{level}_pct"] = share
        rows.append(row)
    return pd.DataFrame(rows)


def validate_levels(school_agg: pd.DataFrame, ts_escola_path: Path, level_counts: dict[str, int], out_dir: Path) -> None:
    """Cross-check computed level shares against INEP's own NIVEL_*_MT9/LP9 percentages in TS_ESCOLA."""
    if not ts_escola_path.exists():
        return
    columns, sep, encoding = sniff_header(ts_escola_path)
    pattern = re.compile(r"^NIVEL_(\d+)_(MT9|LP9)$")
    subject_map = {"MT9": "math", "LP9": "portuguese"}
    official_cols = {c: (subject_map[pattern.match(c).group(2)], int(pattern.match(c).group(1))) for c in columns if pattern.match(c)}
    if not official_cols:
        LOG.warning("TS_ESCOLA sem colunas NIVEL_*_MT9/LP9; validação dos cortes de proficiência pulada.")
        return
    school_col = next((c for c in columns if norm_name(c) == "IDESCOLA"), None)
    if school_col is None:
        return
    usecols = [school_col, *official_cols.keys()]
    official = pd.read_csv(ts_escola_path, sep=sep, encoding=encoding, usecols=usecols, dtype="string")
    official = official.rename(columns={school_col: "school_id"})
    official["school_id"] = official["school_id"].str.strip().str.replace(r"\.0$", "", regex=True)
    for c in official_cols:
        official[c] = pd.to_numeric(official[c].str.replace(",", ".", regex=False), errors="coerce")

    merged = school_agg.merge(official, on="school_id", how="inner")
    lines = ["Validação dos cortes de proficiência (config.json -> proficiency_levels)",
             "Comparação entre percentuais calculados no pipeline e os percentuais oficiais do TS_ESCOLA por escola.",
             f"Escolas comparadas: {len(merged)}", ""]
    for official_col, (subject, level) in official_cols.items():
        computed_col = f"{subject}_nivel_{level}_pct"
        if computed_col not in merged:
            continue
        diff = (merged[computed_col] - merged[official_col]).abs()
        lines.append(f"{subject} nível {level}: erro médio absoluto = {diff.mean():.2f} pp (n={diff.notna().sum()})")
    report = "\n".join(lines)
    LOG.info(report.replace("\n", " | "))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validacao_niveis_proficiencia.txt").write_text(report, encoding="utf-8")


def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "socioeconomic" in out and out["socioeconomic"].nunique() >= 5:
        out["socioeconomic_group"] = pd.qcut(
            out["socioeconomic"], q=5,
            labels=["Muito baixo", "Baixo", "Médio", "Alto", "Muito alto"],
            duplicates="drop",
        )
    return out


def save_descriptives(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for value in ["math", "portuguese"]:
        if value not in df:
            continue
        dimensions = [c for c in ["socioeconomic_group", "region", "state", "urban_rural", "school_network"] if c in df]
        if dimensions:
            table = df.groupby(dimensions, observed=True)[value].agg(["count", "mean", "median", "std"]).reset_index()
            table.to_csv(out_dir / f"descritiva_{value}.csv", index=False, encoding="utf-8-sig")

    level_cols = [c for c in df.columns if re.match(r"^(math|portuguese)_nivel_\d+_pct$", c)]
    if level_cols and "socioeconomic_group" in df:
        table = df.groupby("socioeconomic_group", observed=True)[level_cols].mean().reset_index()
        table.to_csv(out_dir / "descritiva_niveis_por_inse.csv", index=False, encoding="utf-8-sig")
    if level_cols and "region" in df:
        table = df.groupby("region", observed=True)[level_cols].mean().reset_index()
        table.to_csv(out_dir / "descritiva_niveis_por_regiao.csv", index=False, encoding="utf-8-sig")


def fit_models(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for subject in ["math", "portuguese"]:
        if subject not in df or "socioeconomic" not in df:
            LOG.warning("Modelo de %s não executado: faltam colunas.", subject)
            continue
        columns = [c for c in [subject, "socioeconomic", "region", "urban_rural", "school_network"] if c in df]
        model_df = df[columns].dropna()
        if len(model_df) < 30:
            LOG.warning("Poucas observações para regressão de %s: %s", subject, len(model_df))
            continue
        terms = ["socioeconomic"]
        for categorical in ["region", "urban_rural", "school_network"]:
            if categorical in model_df and model_df[categorical].nunique() > 1:
                terms.append(f"C({categorical})")
        formula = f"{subject} ~ " + " + ".join(terms)
        model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
        (out_dir / f"modelo_desempenho_{subject}.txt").write_text(model.summary().as_text(), encoding="utf-8")
        if "region" in model_df and model_df["region"].nunique() > 1:
            interaction = smf.ols(f"{subject} ~ socioeconomic * C(region)", data=model_df).fit(cov_type="HC3")
            (out_dir / f"modelo_interacao_regiao_{subject}.txt").write_text(interaction.summary().as_text(), encoding="utf-8")


def make_figures(df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    group_order = ["Muito baixo", "Baixo", "Médio", "Alto", "Muito alto"]

    for subject, title in [("math", "Matemática"), ("portuguese", "Português")]:
        if subject not in df.columns:
            continue
        if {"socioeconomic_group", subject}.issubset(df.columns):
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="socioeconomic_group", y=subject, order=group_order)
            plt.title(f"Proficiência em {title} por nível socioeconômico")
            plt.xlabel("Nível socioeconômico"); plt.ylabel("Proficiência média (escala SAEB)"); plt.tight_layout()
            plt.savefig(fig_dir / f"{subject}_por_inse.png", dpi=160); plt.close()
        if {"socioeconomic", subject}.issubset(df.columns):
            plt.figure(figsize=(9, 6))
            sns.regplot(data=df, x="socioeconomic", y=subject, scatter_kws={"alpha": 0.35}, line_kws={"color": "#b42318"})
            plt.title(f"Associação entre nível socioeconômico e {title}"); plt.tight_layout()
            plt.savefig(fig_dir / f"inse_vs_{subject}.png", dpi=160); plt.close()
        if {"region", subject}.issubset(df.columns) and df["region"].nunique() > 1:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="region", y=subject, errorbar=None)
            plt.title(f"Proficiência média em {title} por região"); plt.tight_layout()
            plt.savefig(fig_dir / f"{subject}_por_regiao.png", dpi=160); plt.close()

        level_cols = sorted(
            [c for c in df.columns if re.match(rf"^{subject}_nivel_\d+_pct$", c)],
            key=lambda c: int(re.search(r"\d+", c).group()),
        )
        if level_cols and "socioeconomic_group" in df:
            stacked = df.groupby("socioeconomic_group", observed=True)[level_cols].mean().reindex(group_order)
            stacked.columns = [f"Nível {re.search(r'(\d+)', c).group()}" for c in level_cols]
            stacked.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="RdYlGn")
            plt.title(f"Distribuição de níveis de proficiência em {title} por nível socioeconômico")
            plt.ylabel("% de alunos"); plt.xlabel("Nível socioeconômico"); plt.xticks(rotation=0)
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout()
            plt.savefig(fig_dir / f"niveis_{subject}_por_inse.png", dpi=160); plt.close()


def main(config_path: str) -> None:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    root = config_file.parent
    aliases = config["columns"]
    proficiency_levels = config.get("proficiency_levels", {})
    code_labels = config.get("code_labels", {})

    geo_lookup_aliases = {k: v for k, v in aliases.items() if k in {"state_code", "state", "region_code", "region"}}
    censo_geo = load_and_prepare(root / config["censo_file"], geo_lookup_aliases, "Censo Escolar (tabela UF/região)")
    state_lookup, region_lookup = build_geo_lookup(censo_geo)

    student_aliases = {
        k: v for k, v in aliases.items()
        if k in {"school_id", "grade", "year", "math", "portuguese", "socioeconomic",
                  "weight_math", "weight_portuguese", "weight_inse",
                  "region_code", "state_code", "urban_rural_code", "school_network_code"}
    }
    saeb = load_and_prepare(root / config["saeb_file"], student_aliases, "SAEB (alunos)")
    saeb = filter_grade(saeb, config.get("target_grade", ""))
    saeb = apply_geo_lookup(saeb, state_lookup, region_lookup)
    if "urban_rural_code" in saeb:
        saeb["urban_rural"] = saeb["urban_rural_code"]
    if "school_network_code" in saeb:
        saeb["school_network"] = saeb["school_network_code"]
    saeb = apply_code_labels(saeb, code_labels)

    level_counts = {}
    if "math" in saeb and "math_9ef" in proficiency_levels:
        saeb["math_level"] = classify_levels(saeb["math"], proficiency_levels["math_9ef"])
        level_counts["math"] = len(proficiency_levels["math_9ef"]) + 1
    if "portuguese" in saeb and "portuguese_9ef" in proficiency_levels:
        saeb["portuguese_level"] = classify_levels(saeb["portuguese"], proficiency_levels["portuguese_9ef"])
        level_counts["portuguese"] = len(proficiency_levels["portuguese_9ef"]) + 1

    school_agg = aggregate_saeb(saeb, level_counts)

    results = root / config["output_dir"]
    if level_counts:
        validate_levels(school_agg, root / config["saeb_school_file"], level_counts, results)

    base = create_derived_variables(school_agg)

    processed = root / config["processed_dir"]; figures = root / config["figures_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    base.to_csv(processed / "base_analitica.csv", index=False, encoding="utf-8-sig")
    base.to_parquet(processed / "base_analitica.parquet", index=False)
    save_descriptives(base, results); fit_models(base, results); make_figures(base, figures)
    LOG.info("Concluído. Base: %s", processed / "base_analitica.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    main(parser.parse_args().config)
