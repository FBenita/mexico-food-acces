"""
mxo_utils.py -- shared helpers for the walking access to food pipeline.

Every notebook imports its paths, crosswalks, key builders, parsers and run
logging from here, so that no notebook redefines a crosswalk inline. Every join
in this module is asserted rather than filled with a silent default.

The project root is read from the environment variable MXO_BASE and defaults to
the Google Drive folder used in Colab:

    import os, sys, importlib
    os.environ.setdefault('MXO_BASE', '/content/drive/MyDrive/Mexico_Obesity')
    sys.path.insert(0, os.path.join(os.environ['MXO_BASE'], 'Code'))
    import mxo_utils as mx
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np
import pandas as pd

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = os.environ.get("MXO_BASE", "/content/drive/MyDrive/Mexico_Obesity")

DIRS = {
    "code": f"{BASE}/Code",
    "inputs": f"{BASE}/Inputs",
    "denue": f"{BASE}/Inputs/DENUE",
    "deprivation": f"{BASE}/Inputs/Deprivation",
    "networks": f"{BASE}/Networks/pedestrian_networks",
    "results": f"{BASE}/Results",
    "figures": f"{BASE}/Results/figures",
    "tables": f"{BASE}/Results/tables",
    "logs": f"{BASE}/Results/logs",
}

# The pedestrian networks, one GeoPackage per metropolitan area.
NETWORKS_CANDIDATES = (DIRS["networks"],)


# Mexico's official metric CRS (ITRF2008 / LCC). Areas and lengths in metres.
CRS_METRIC = 6372


# ---------------------------------------------------------------------------
# Sample-flow log: every filter writes one row, so attrition is auditable
# ---------------------------------------------------------------------------

_FLOW: list[dict] = []


def flow(stage: str, n: int, reason: str = "") -> None:
    """Record a sample-size checkpoint. Printed and kept for the flow figure."""
    prev = _FLOW[-1]["n"] if _FLOW else None
    delta = "" if prev is None else f"  ({n - prev:+,})"
    _FLOW.append({"stage": stage, "n": int(n), "reason": reason})
    print(f"[FLOW] {stage:<46s} N = {n:>8,}{delta}   {reason}")


def flow_table() -> pd.DataFrame:
    return pd.DataFrame(_FLOW)


def flow_reset() -> None:
    """Clear the flow log. Call at the top of a notebook that records a waterfall,
    so a re-run in the same kernel does not append to the previous run's rows."""
    _FLOW.clear()


def save_flow(path: str | None = None) -> str:
    path = path or f"{DIRS['logs']}/sample_flow.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flow_table().to_csv(path, index=False)
    print(f"[FLOW] written to {path}")
    return path


# ---------------------------------------------------------------------------
# Geographic keys
# ---------------------------------------------------------------------------

def build_cvegeo(ent, mun, loc, ageb) -> pd.Series:
    """
    Assemble the 13-character urban-AGEB CVEGEO from its components.

    CVEGEO = ENT(2) + MUN(3) + LOC(4) + AGEB(4), e.g. 01 001 0001 006A.

    Needed because CSV/Excel round-trips mangle all-numeric AGEB keys into
    scientific notation (1.001E+11) while keys containing a letter survive as
    text. Never trust a pre-built key column; rebuild it from the parts.

    `mun` may arrive as a bare municipality number (1) or as a state+municipality
    clave (01001); both are handled. Likewise `loc` as 1 or 010010001.
    """
    def s(x, width, take_last=False):
        out = (
            pd.Series(x).astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.upper()
        )
        if take_last:
            # strip any embedded higher-level prefix, then pad
            out = out.where(out.str.len() <= width, out.str[-width:])
        return out.str.zfill(width)

    ent_s = s(ent, 2)
    mun_s = s(mun, 3, take_last=True)
    loc_s = s(loc, 4, take_last=True)
    ageb_s = s(ageb, 4)
    key = ent_s + mun_s + loc_s + ageb_s

    bad = key.str.len() != 13
    if bad.any():
        raise ValueError(
            f"build_cvegeo produced {bad.sum():,} keys that are not 13 chars; "
            f"first offenders: {key[bad].head().tolist()}"
        )
    return key


def build_cvegeo_mza(ent, mun, loc, ageb, mza) -> pd.Series:
    """16-character manzana key: the 13-char AGEB key plus a 3-digit block."""
    base = build_cvegeo(ent, mun, loc, ageb)
    m = (
        pd.Series(mza).astype(str).str.strip()
        .str.replace(r"\.0$", "", regex=True).str.zfill(3)
    )
    key = base.reset_index(drop=True) + m.reset_index(drop=True)
    bad = key.str.len() != 16
    if bad.any():
        raise ValueError(f"{bad.sum():,} manzana keys are not 16 chars")
    return key


def assert_join(left_keys, right_keys, name: str, min_rate: float = 0.95) -> float:
    """Report and enforce a join match rate instead of letting it fail silently."""
    left = pd.Index(pd.Series(left_keys).dropna().unique())
    right = pd.Index(pd.Series(right_keys).dropna().unique())
    hit = left.isin(right).mean() if len(left) else 0.0
    print(f"[JOIN] {name:<40s} {hit:6.2%} of {len(left):,} keys matched")
    if hit < min_rate:
        raise AssertionError(
            f"{name}: only {hit:.2%} matched (expected >= {min_rate:.0%}). "
            "Check key construction before proceeding."
        )
    return hit


# ---------------------------------------------------------------------------
# INEGI census symbols: '*' is interval censoring, 'N/D' is missing
# ---------------------------------------------------------------------------

SUPPRESSED = "*"      # fewer than 3 units (LSNIEG art. 37). True value in {0,1,2}.
NOT_AVAILABLE = {"N/D", "N/A", "ND"}   # area not enumerated at all


def parse_census_column(raw: pd.Series, star_fill: float = 1.0):
    """
    Parse one INEGI AGEB census column, keeping the two kinds of non-numeric
    cell apart instead of collapsing both to zero.

    Returns (values, is_star, is_nd):
      values  -- numeric, with '*' replaced by `star_fill` and 'N/D' left NaN
      is_star -- bool mask of confidentiality-suppressed cells
      is_nd   -- bool mask of unenumerated cells

    '*' means the true count is 0, 1 or 2, so this is interval censoring, not
    missingness. star_fill=1 is the interval midpoint; run the pipeline again
    with 0 and 2 for the sensitivity check.
    """
    s = raw.astype(str).str.strip().str.upper()
    is_star = s.eq(SUPPRESSED)
    is_nd = s.isin(NOT_AVAILABLE)
    num = pd.to_numeric(s.where(~(is_star | is_nd)), errors="coerce")
    num = num.mask(is_star, star_fill)
    return num, is_star, is_nd


def parse_census_frame(df: pd.DataFrame, cols, star_fill: float = 1.0):
    """parse_census_column over several columns; returns (df, report)."""
    out = df.copy()
    rows = []
    for c in cols:
        if c not in out.columns:
            rows.append({"variable": c, "present": False,
                         "n_star": 0, "n_nd": 0, "pct_star": np.nan})
            continue
        vals, star, nd = parse_census_column(out[c], star_fill=star_fill)
        out[c] = vals
        out[f"_star_{c}"] = star
        out[f"_nd_{c}"] = nd
        rows.append({"variable": c, "present": True,
                     "n_star": int(star.sum()), "n_nd": int(nd.sum()),
                     "pct_star": round(100 * star.mean(), 3)})
    report = pd.DataFrame(rows)
    return out, report


# ---------------------------------------------------------------------------
# Deprivation: CONAPO IMU 2020 and CONEVAL GRS AGEB urbana 2020
# ---------------------------------------------------------------------------

IMU_INDICATORS = [
    "P6A14NAE", "SBASC", "PSDSS", "OVSDE", "OVSEE",
    "OVSAE", "OVPT", "OVSREF", "OVSINT", "OVSCEL", "OVHAC",
]

GRADE_ORDER = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]


def _norm_grade(s: pd.Series) -> pd.Series:
    """Normalise grade labels: strip accents, collapse case and whitespace."""
    t = (
        s.astype(str)
        .map(lambda x: unicodedata.normalize("NFKD", x)
             .encode("ascii", "ignore").decode("ascii"))
        .str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    )
    lookup = {
        "muy bajo": "Muy bajo", "bajo": "Bajo", "medio": "Medio",
        "alto": "Alto", "muy alto": "Muy alto",
    }
    return t.map(lookup)


def load_imu(path: str) -> pd.DataFrame:
    """
    CONAPO Indice de Marginacion Urbana 2020, AGEB level.

    Two traps this handles:

    1. CVE_AGEB in the distributed CSV is corrupted by scientific notation
       (1.001E+11) for all-numeric AGEB codes, while codes containing a letter
       survive as text. The key is rebuilt from ENT/MUN/LOC/AGEB instead.
    2. IM_2020 and IMN_2020 run the OPPOSITE way to intuition: higher values
       mean LESS marginalisation (a 'Muy bajo' AGEB scores ~0.98 on IMN_2020,
       a 'Medio' one ~0.93). Dropping IM_2020 straight into a regression and
       reading a positive coefficient as 'more deprived' inverts the result.
       This returns `imu_depriv` = 1 - IMN_2020, so higher always means more
       deprived, plus `imu_depriv_z`, its z-score.
    """
    df = pd.read_csv(path, dtype=str, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]

    need = ["ENT", "MUN", "LOC", "AGEB", "IM_2020", "GM_2020", "IMN_2020"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"IMU file is missing columns {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame({"CVEGEO": build_cvegeo(df["ENT"], df["MUN"], df["LOC"], df["AGEB"])})
    out["imu_raw"] = pd.to_numeric(df["IM_2020"], errors="coerce")
    out["imu_norm"] = pd.to_numeric(df["IMN_2020"], errors="coerce")
    out["imu_grade"] = _norm_grade(df["GM_2020"])
    out["imu_depriv"] = 1.0 - out["imu_norm"]
    out["imu_depriv_z"] = (out["imu_depriv"] - out["imu_depriv"].mean()) / out["imu_depriv"].std()
    for c in IMU_INDICATORS:
        if c in df.columns:
            out[f"imu_{c.lower()}"] = pd.to_numeric(df[c], errors="coerce")

    out = out.drop_duplicates(subset="CVEGEO")

    # Orientation check: the mean deprivation score must rise across the grades.
    chk = out.groupby("imu_grade")["imu_depriv"].mean().reindex(GRADE_ORDER)
    if not chk.dropna().is_monotonic_increasing:
        raise AssertionError(
            "imu_depriv is not increasing across GM_2020 grades; the sign "
            f"convention has changed. Observed:\n{chk}"
        )
    print(f"[IMU]  {len(out):,} AGEBs | imu_depriv rises Muy bajo -> Muy alto: "
          + " < ".join(f"{v:.3f}" for v in chk.dropna()))
    return out


def load_grs(path: str, sheet: str = "GRS 2020") -> pd.DataFrame:
    """
    CONEVAL Grado de Rezago Social a nivel AGEB urbana 2020.

    The workbook has a title block, a two-level header in rows 4 and 5 and two
    footer lines, so positions are pinned explicitly rather than inferred.
    Layout verified against the published file: one sheet, data in rows 7-61436,
    61,430 unique 13-character keys in column 8, grade in column 28.

    The grade is populated for ALL 61,430 AGEBs, including the ~2,700 where the
    underlying indicators are suppressed, which makes it the right backfill for
    tracts the IMU drops under its 20-dwelling rule.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=6, dtype=str)

    # Trim the footer: keep rows whose column 8 is a 13-character key.
    key = raw.iloc[:, 7].astype(str).str.strip()
    raw = raw[key.str.fullmatch(r"[0-9A-Za-z]{13}")].copy()

    out = pd.DataFrame({
        "CVEGEO": raw.iloc[:, 7].astype(str).str.strip().str.upper(),
        "grs_pobtot": pd.to_numeric(raw.iloc[:, 8], errors="coerce"),
    })
    viv, viv_star, _ = parse_census_column(raw.iloc[:, 9])
    out["grs_viv"] = viv
    out["grs_viv_suppressed"] = viv_star.values
    out["grs_grade"] = _norm_grade(raw.iloc[:, 27])
    out["grs_grade_ord"] = out["grs_grade"].map({g: i for i, g in enumerate(GRADE_ORDER)})

    if out["CVEGEO"].duplicated().any():
        raise AssertionError("GRS: duplicate CVEGEO keys found.")
    if out["grs_grade"].isna().any():
        raise AssertionError(
            f"GRS: {out['grs_grade'].isna().sum():,} rows have an unreadable grade."
        )
    print(f"[GRS]  {len(out):,} AGEBs | grades: "
          + ", ".join(f"{g}={int((out['grs_grade'] == g).sum()):,}" for g in GRADE_ORDER))
    return out


def load_deprivation(imu_path: str, grs_path: str) -> pd.DataFrame:
    """Merge IMU and GRS on CVEGEO, keeping both and flagging coverage."""
    imu = load_imu(imu_path)
    grs = load_grs(grs_path)
    assert_join(imu["CVEGEO"], grs["CVEGEO"], "IMU keys found in GRS", min_rate=0.90)
    dep = grs.merge(imu, on="CVEGEO", how="outer", indicator="_dep_src")
    dep["has_imu"] = dep["_dep_src"].isin(["right_only", "both"])
    dep["has_grs"] = dep["_dep_src"].isin(["left_only", "both"])
    print(f"[DEP]  union = {len(dep):,} AGEBs | with IMU {dep['has_imu'].sum():,} "
          f"| with GRS {dep['has_grs'].sum():,}")
    return dep.drop(columns="_dep_src")


# ---------------------------------------------------------------------------
# Metro crosswalk
# ---------------------------------------------------------------------------

NORTH_ENT = {"02", "03", "05", "08", "10", "19", "25", "26", "28"}
SOUTH_ENT = {"04", "07", "12", "20", "23", "27", "30", "31"}


def macro_region(cve_ent: pd.Series) -> pd.Series:
    e = pd.Series(cve_ent).astype(str).str.zfill(2)
    return np.select(
        [e.isin(NORTH_ENT), e.isin(SOUTH_ENT)],
        ["North", "South"],
        default="Centre",
    )


# ---------------------------------------------------------------------------
# Official metropolis typology
# (SEDATU, CONAPO, INEGI: Metropolis de Mexico 2020)
# ---------------------------------------------------------------------------
#
# The size tiers below are this project's own convention, with the 1 million cut
# matching the official delimitation criterion and the 4 million cut matching
# nothing published. This typology is different: it is the three-way
# classification of the 92 metropolitan areas as published, from the same source
# metros_92.csv already draws conapo_pop_2020 and conapo_surface_ha from. A
# paper arguing that categories must be justified rather than inherited is on
# firmer ground stratifying by a category someone else defined, published and
# can be cited for. The cells are also usable at 48, 22 and 22, where the size
# tiers leave only three metros in Tier 1.
#
# The typology is not a size ranking. A metropoli municipal is a metropolis
# contained in a single municipality, which correlates with size but measures
# administrative form, and a zona conurbada is the small end. Describe the
# categories as what they are rather than relabelling them as size bands.

CONAPO_ZM = (
    "Ciudad de Mexico", "Monterrey", "Guadalajara", "Toluca", "Puebla-Tlaxcala",
    "Merida", "Tlaxcala-Apizaco", "Tijuana", "Leon", "La Laguna", "Queretaro",
    "Chihuahua", "Saltillo", "Cuernavaca", "Tampico", "San Luis Potosi",
    "Reynosa", "Oaxaca", "Aguascalientes", "Tuxtla Gutierrez", "Morelia",
    "Pachuca", "Villahermosa", "Tapachula", "Cancun", "Veracruz", "Tulancingo",
    "Monclova-Frontera", "Celaya", "Xalapa", "Cuautla", "Orizaba",
    "San Martin Texmelucan", "Puerto Vallarta", "Minatitlan", "Guaymas",
    "Zacatecas-Guadalupe", "Coatzacoalcos", "La Piedad-Penjamo", "Tepic",
    "Colima-Villa de Alvarez", "Tehuacan", "Poza Rica", "Piedras Negras",
    "Cordoba", "Delicias", "Chilpancingo", "Zamora",
)

CONAPO_MM = (
    "Juarez", "Mexicali", "Hermosillo", "Culiacan", "Acapulco", "Durango",
    "Matamoros", "Ensenada", "Los Cabos", "La Paz", "Nuevo Laredo", "Irapuato",
    "Los Mochis", "Ciudad Obregon", "Mazatlan", "Campeche", "Ciudad Victoria",
    "Chetumal", "Playa del Carmen", "Uruapan", "Nogales", "Guanajuato",
)

CONAPO_ZC = (
    "Lazaro Cardenas", "Rioverde", "Silao", "Tianguistenco", "Caborca",
    "Sabinas", "Matehuala", "Moroleon-Uriangato", "Ocotlan", "Huamantla",
    "Teziutlan", "Valladolid", "Salina Cruz", "Hidalgo del Parral", "Sahuayo",
    "Tecoman", "Huauchinango", "Juchitan", "Ozumba", "Atitalaquia",
    "Tehuantepec", "Acayucan",
)

TIPO_ZM = "Zona metropolitana"
TIPO_MM = "Metropoli municipal"
TIPO_ZC = "Zona conurbada"
TIPO_ORDER = [TIPO_ZM, TIPO_MM, TIPO_ZC]
TIPO_EXPECTED = {TIPO_ZM: 48, TIPO_MM: 22, TIPO_ZC: 22}


def _slug_name(s) -> pd.Series:
    """
    Accent-free, lower-case, separator-free form of a place name.

    'Puebla-Tlaxcala', 'puebla_tlaxcala', 'Puebla Tlaxcala' and
    'metro_puebla_tlaxcala' all have to reach the same entry, because the
    published table, the crosswalk and the network filenames each spell these
    names differently. A leading 'metro' prefix is removed first.
    """
    out = (
        pd.Series(s).astype(str)
        .map(lambda x: unicodedata.normalize("NFKD", x)
             .encode("ascii", "ignore").decode("ascii"))
        .str.lower()
        .str.replace(r"^metro[_\- ]+", "", regex=True)
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )
    return out


def _slug1(name: str) -> str:
    return _slug_name([name]).iloc[0]


_TIPO_LOOKUP: dict[str, str] = {}
for _names, _t in ((CONAPO_ZM, TIPO_ZM), (CONAPO_MM, TIPO_MM), (CONAPO_ZC, TIPO_ZC)):
    for _n in _names:
        _TIPO_LOOKUP[_slug1(_n)] = _t

# Spellings this project may use that differ from the published table. Each maps
# onto a name already in one of the three tuples above, so a typo here raises at
# import rather than silently inventing a category.
_TIPO_ALIASES = {
    "valledemexico": "Ciudad de Mexico",
    "zmvm": "Ciudad de Mexico",
    "cdmx": "Ciudad de Mexico",
    "ciudaddemexico": "Ciudad de Mexico",
    "ciudadjuarez": "Juarez",
    "ahome": "Los Mochis",
    "solidaridad": "Playa del Carmen",
    "obregon": "Ciudad Obregon",
    "cdobregon": "Ciudad Obregon",
    "victoria": "Ciudad Victoria",
    "cdvictoria": "Ciudad Victoria",
    "lapiedad": "La Piedad-Penjamo",
    "lapiedadpenjamo": "La Piedad-Penjamo",
    "colima": "Colima-Villa de Alvarez",
    "colimavilladealvarez": "Colima-Villa de Alvarez",
    "zacatecas": "Zacatecas-Guadalupe",
    "zacatecasguadalupe": "Zacatecas-Guadalupe",
    "monclova": "Monclova-Frontera",
    "monclovafrontera": "Monclova-Frontera",
    "tlaxcala": "Tlaxcala-Apizaco",
    "tlaxcalaapizaco": "Tlaxcala-Apizaco",
    "puebla": "Puebla-Tlaxcala",
    "pueblatlaxcala": "Puebla-Tlaxcala",
    "sanmartintexmelucan": "San Martin Texmelucan",
    "riverde": "Rioverde",
    "rioverde": "Rioverde",
    "lagunalatorreon": "La Laguna",
    "torreon": "La Laguna",
    "moroleonuriangato": "Moroleon-Uriangato",
    "lazarocardenas": "Lazaro Cardenas",
    "sanluispotosi": "San Luis Potosi",
    "tuxtlagutierrez": "Tuxtla Gutierrez",
    "piedrasnegras": "Piedras Negras",
    "nuevolaredo": "Nuevo Laredo",
    "pozarica": "Poza Rica",
    "puertovallarta": "Puerto Vallarta",
    "loscabos": "Los Cabos",
    "lapaz": "La Paz",
    "losmochis": "Los Mochis",
    "playadelcarmen": "Playa del Carmen",
    "hidalgodelparral": "Hidalgo del Parral",
    "salinacruz": "Salina Cruz",
}
for _alias, _canon in _TIPO_ALIASES.items():
    _canon_slug = _slug1(_canon)
    if _canon_slug not in _TIPO_LOOKUP:
        raise RuntimeError(
            f"alias {_alias!r} points at {_canon!r}, which is not in the "
            "CONAPO typology tuples"
        )
    _TIPO_LOOKUP.setdefault(_alias, _TIPO_LOOKUP[_canon_slug])

if len(CONAPO_ZM) + len(CONAPO_MM) + len(CONAPO_ZC) != 92:
    raise RuntimeError(
        "the CONAPO typology tuples do not hold 92 names in total: "
        f"{len(CONAPO_ZM)} + {len(CONAPO_MM)} + {len(CONAPO_ZC)}"
    )


def conapo_tipo(names) -> pd.Series:
    """
    Map metro names onto the official typology, refusing to guess.

    Raises with the unmatched names rather than returning NaN, because a silent
    default on a crosswalk of this kind sends every unmatched area to whichever
    class the default names.
    """
    slug = _slug_name(names)
    out = slug.map(_TIPO_LOOKUP)
    if out.isna().any():
        bad = pd.Series(names).astype(str)[out.isna().values]
        pairs = sorted(set(zip(bad, slug[out.isna().values])))
        raise KeyError(
            f"{len(pairs)} metro names are not in the CONAPO typology.\n"
            + "\n".join(f"  {n!r}  (slug {s!r})" for n, s in pairs)
            + "\nAdd them to _TIPO_ALIASES in mxo_utils."
        )
    return out


def load_metros(path: str | None = None, expected: int = 92) -> pd.DataFrame:
    """
    Load metros_92.csv and refuse to proceed on an incomplete crosswalk.

    Adds two stratifiers. `tier` is this project's own size convention.
    `tipo` is the published SEDATU/CONAPO/INEGI classification, which is the
    one to prefer wherever a stratification has to be justified in print.
    """
    path = path or f"{DIRS['inputs']}/metros_92.csv"
    df = pd.read_csv(path, dtype={"conapo_pop_2020": "Int64"})
    if len(df) != expected:
        raise AssertionError(f"metros crosswalk has {len(df)} rows, expected {expected}")
    if df["metro_key"].duplicated().any():
        raise AssertionError("metros crosswalk has duplicate metro_key values")
    bad = df[df["conapo_pop_2020"].isna()]
    if len(bad):
        raise AssertionError(
            "metros crosswalk has rows with no population:\n"
            f"{bad[['metro_key', 'source_flag']].to_string(index=False)}\n"
            "Fill them from Metropolis de Mexico 2020 before running the pipeline."
        )

    df["tier"] = np.select(
        [df["conapo_pop_2020"] >= 4_000_000, df["conapo_pop_2020"] >= 1_000_000],
        ["Tier 1: Megacities", "Tier 2: Large metros"],
        default="Tier 3: Medium and small",
    )
    df["tipo"] = conapo_tipo(df["metro_name"]).values

    print(f"[METRO] {len(df)} metros | "
          + " | ".join(f"{t}: {n}" for t, n in df["tier"].value_counts().items()))
    print("[METRO] official typology | "
          + " | ".join(f"{t}: {int((df['tipo'] == t).sum())}" for t in TIPO_ORDER))

    got = {t: int((df["tipo"] == t).sum()) for t in TIPO_ORDER}
    if got != TIPO_EXPECTED:
        raise AssertionError(
            f"typology counts are {got}, expected {TIPO_EXPECTED}. "
            "Either the crosswalk lost a metro or an alias maps two names onto one."
        )
    return df


# ---------------------------------------------------------------------------
# Pedestrian network geometry
# ---------------------------------------------------------------------------

def _coords(geom):
    """Coordinate list for a LineString, or for the first part of a multi-part."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return list(geom.coords)
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms[0].coords)
    return None


def _edge_key(coords, ndigits: int = 2) -> bytes:
    """
    Direction-normalised hash of an edge's full geometry.

    The endpoint pair is ordered before hashing, so the reversed copy OSMnx
    writes under `bidirectional=True` collapses onto the forward one, while two
    genuinely parallel edges between the same nodes keep distinct keys because
    their intermediate vertices differ.
    """
    import hashlib

    if coords[0] > coords[-1]:
        coords = coords[::-1]
    payload = ";".join(f"{float(x):.{ndigits}f},{float(y):.{ndigits}f}" for x, y in coords)
    return hashlib.blake2b(payload.encode("ascii"), digest_size=16).digest()


def first_layer(path: str) -> str:
    """
    Name of a GeoPackage's first (or first edge-like) layer, without fiona.

    Colab no longer ships fiona, so pyogrio is used, with a fiona fallback for
    environments that still have it.
    """
    try:
        from pyogrio import list_layers
        names = [row[0] for row in list_layers(path)]
    except ImportError:
        import fiona
        names = fiona.listlayers(path)
    edges = [n for n in names if "edge" in n.lower()]
    return edges[0] if edges else names[0]


def read_gpkg(path: str, columns=None, layer: str | None = None):
    """
    Read a GeoPackage, surviving mis-encoded text fields.

    Some GeoPackages store latin1 bytes in their name columns although the
    format is nominally UTF-8, so a plain read fails on the first accented name.
    Passing `columns` avoids decoding those fields at all, which is also faster.
    If a field that is needed cannot be decoded, the read falls back to fiona
    with an explicit encoding.
    """
    import geopandas as gpd

    lyr = layer or first_layer(path)
    try:
        return gpd.read_file(path, layer=lyr, columns=columns, engine="pyogrio")
    except UnicodeDecodeError:
        pass

    try:
        out = gpd.read_file(path, layer=lyr, encoding="latin1", engine="fiona")
    except (ImportError, ValueError) as exc:
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1,
            f"{os.path.basename(path)} holds non-UTF-8 text. Pass `columns` to skip "
            f"the name fields, or `pip install fiona` for the latin1 fallback ({exc})."
        ) from None
    return out[list(columns) + ["geometry"]] if columns else out


def dedupe_edges(net):
    """
    Collapse the two directed copies OSMnx exports for every walkable way.

    Summing length over the exported frame would count every way twice.
    Returns the undirected edge set.
    """
    keys = []
    for geom in net.geometry.values:
        c = _coords(geom)
        keys.append(None if c is None else _edge_key(c))
    out = net.assign(_k=keys).dropna(subset=["_k"])
    return out.drop_duplicates("_k").drop(columns="_k")


def endpoint_degrees(net, ndigits: int = 1):
    """Map rounded endpoint -> number of incident edge ends. De-duplicate first."""
    from collections import Counter

    deg, pos = Counter(), {}
    for geom in net.geometry.values:
        c = _coords(geom)
        if c is None:
            continue
        for xy in (c[0], c[-1]):
            k = (round(float(xy[0]), ndigits), round(float(xy[1]), ndigits))
            deg[k] += 1
            pos[k] = xy
    return deg, pos


def intersection_nodes(net, min_degree: int = 3):
    """
    Real intersections, reconstructed from edge endpoints.

    The saved networks hold edges only, but OSMnx already merged degree-two
    nodes during simplification, so a surviving endpoint touched by three or
    more edge ends is an intersection. This yields intersection density, the
    measure Frank, Sallis and Boeing actually use, without re-extracting OSM.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    deg, pos = endpoint_degrees(net)
    pts = [pos[k] for k, c in deg.items() if c >= min_degree]
    if not pts:
        return gpd.GeoDataFrame(geometry=[], crs=net.crs)
    return gpd.GeoDataFrame(geometry=[Point(p) for p in pts], crs=net.crs)


# ---------------------------------------------------------------------------
# Text normalisation for the DENUE classifier
# ---------------------------------------------------------------------------

def normalise_text(s: pd.Series) -> pd.Series:
    """
    Upper-case, strip accents and punctuation, collapse whitespace.

    Matching raw upper-cased names would miss every name carrying an accent or
    an apostrophe, such as SAM'S CLUB.
    """
    t = (
        s.fillna("").astype(str)
        .map(lambda x: unicodedata.normalize("NFKD", x)
             .encode("ascii", "ignore").decode("ascii"))
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return t


class stage:
    """
    Tiny timer, so a slow metro says which step is slow instead of just hanging.

        with mx.stage('fair share'):
            ...
    """

    def __init__(self, label: str, quiet: bool = False):
        self.label = label
        self.quiet = quiet
        self.seconds = 0.0

    def __enter__(self):
        import time
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        self.seconds = time.perf_counter() - self._t0
        if not self.quiet:
            print(f"      {self.label:<24s} {self.seconds:>8.1f}s", flush=True)
        return False


def edge_endpoint_nodes(net, ndigits: int = 1):
    """
    Node ids for the two ends of every edge, from rounded coordinates.

    Returns (u, v, lengths, n_nodes). Edges whose geometry is empty are dropped,
    so `u` may be shorter than `net`.
    """
    import shapely

    geoms = np.asarray(net.geometry.values)
    n = len(geoms)
    if n == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty(0, float), 0)

    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    order = np.arange(n)
    lo = np.searchsorted(idx, order, side="left")
    hi = np.searchsorted(idx, order, side="right") - 1
    ok = hi >= lo                      # geometries that contributed coordinates
    lo, hi = lo[ok], hi[ok]

    ends = np.round(np.vstack([coords[lo], coords[hi]]), ndigits)
    uniq, inv = np.unique(ends, axis=0, return_inverse=True)
    k = len(lo)
    u, v = inv[:k].astype(np.int64), inv[k:].astype(np.int64)
    lengths = shapely.length(geoms[ok])
    return u, v, lengths, len(uniq)


def component_summary(net, ndigits: int = 1):
    """
    Connected components of an edge layer, and how much length each holds.

    Exists to answer one question about the saved pedestrian networks: were the
    disconnected parts of each metro thrown away during extraction? OSMnx does
    that whenever `retain_all=False` or `largest_component` is called, and it is
    silent. A metro whose saved network has exactly one component, in a country
    whose metropolitan areas are unions of whole municipios and routinely
    contain detached towns, has almost certainly been through that step.

    Returns a dict: n_components, n_nodes, n_edges, total_km, largest_km,
    largest_share, and km_outside_largest.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    u, v, lengths, n_nodes = edge_endpoint_nodes(net, ndigits=ndigits)
    if len(u) == 0:
        return {"n_components": 0, "n_nodes": 0, "n_edges": 0, "total_km": 0.0,
                "largest_km": 0.0, "largest_share": np.nan,
                "km_outside_largest": 0.0}

    adj = coo_matrix((np.ones(len(u), dtype=np.int8), (u, v)),
                     shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(adj, directed=False)

    per_comp = np.bincount(labels[u], weights=lengths, minlength=n_comp) / 1000.0
    total = float(per_comp.sum())
    largest = float(per_comp.max()) if n_comp else 0.0
    return {
        "n_components": int(n_comp),
        "n_nodes": int(n_nodes),
        "n_edges": int(len(u)),
        "total_km": round(total, 1),
        "largest_km": round(largest, 1),
        "largest_share": round(largest / total, 6) if total else np.nan,
        "km_outside_largest": round(total - largest, 1),
    }


# ---------------------------------------------------------------------------
# Caches that know which inputs made them
# ---------------------------------------------------------------------------
#
# A cache keyed only by area name cannot tell which pedestrian networks
# produced it, so a run on replaced networks would read back numbers computed
# from the previous ones. Keying the cache directory by a fingerprint of the
# network files makes such a cache unreadable rather than wrong: a different
# input set is a different directory, so nothing has to be remembered or
# deleted.

import glob as _glob
import hashlib as _hashlib
import io as _io
import json as _json
import sys as _sys
from datetime import datetime, timezone


def network_fingerprint(path: str | None = None) -> str:
    """
    Short hash of the pedestrian network files currently in use.

    Built from the folder name, the file names and their sizes. Sizes rather
    than modification times, because copying a folder on Drive rewrites every
    mtime without changing a byte, and a fingerprint that changes when nothing
    has changed is as bad as one that does not change when something has.
    """
    path = path or DIRS["networks"]
    files = sorted(_glob.glob(os.path.join(path, "*_pedestrian.gpkg")))
    h = _hashlib.blake2b(digest_size=5)
    h.update(os.path.basename(path.rstrip("/")).encode())
    h.update(str(len(files)).encode())
    for f in files:
        h.update(os.path.basename(f).encode())
        try:
            h.update(str(os.path.getsize(f)).encode())
        except OSError:
            h.update(b"?")
    return h.hexdigest()


def cache_dir(name: str, fingerprint: str | None = None) -> str:
    """
    A cache directory tied to the current network files.

        CACHE = mx.cache_dir('cache_01_networks')

    Change the networks and the path changes with them, so a cache built from
    other networks is never read back.
    """
    fp = fingerprint or network_fingerprint()
    path = os.path.join(DIRS["results"], f"{name}__{fp}")
    os.makedirs(path, exist_ok=True)
    return path


def network_inventory(path: str | None = None) -> pd.DataFrame:
    """One row per pedestrian network file, with size in megabytes."""
    path = path or DIRS["networks"]
    rows = []
    for f in sorted(_glob.glob(os.path.join(path, "*_pedestrian.gpkg"))):
        rows.append({"file": os.path.basename(f),
                     "mb": round(os.path.getsize(f) / 1e6, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One file per notebook run, holding everything needed to assess it
# ---------------------------------------------------------------------------

class _Tee:
    """Writes to the real stdout and to a file at the same time."""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, s):
        self.stream.write(s)
        try:
            self.fh.write(s)
        except Exception:                                        # noqa: BLE001
            pass
        return len(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:                                        # noqa: BLE001
            pass
        try:
            self.fh.flush()
        except Exception:                                        # noqa: BLE001
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


_ACTIVE: list = []


class Report:
    """
    Capture a whole notebook run into one markdown file.

        rep = mx.Report('01_network_metrics').start()
        ...                                  # the notebook runs as usual
        rep.add_table('attribution audit', df_diag)
        rep.save()

    `start` writes a header recording the module version, the library versions,
    which pedestrian networks are in use and their fingerprint, the cache
    directory, and the size and age of every input file that matters, then tees
    stdout so every print lands in the file as well as on screen. Writing goes
    to local disk and is flushed as it happens, so a disconnected runtime still
    leaves a usable partial report. `save` appends any tables, copies the file
    to the logs folder on Drive, and offers it for download.

    The header exists because the hardest failures to see are the ones where a
    notebook runs to completion on the wrong inputs. Reading it takes a second
    and settles what actually went in.
    """

    def __init__(self, name: str, local_dir: str = "/content"):
        self.name = name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        base = f"run_log_{name}_{stamp}.md"
        self.local_path = (os.path.join(local_dir, base)
                           if os.path.isdir(local_dir)
                           else os.path.join(DIRS["logs"], base))
        self.drive_path = os.path.join(DIRS["logs"], base)
        self.tables: dict[str, pd.DataFrame] = {}
        self.notes: list[str] = []
        self._fh = None

    # -- header ------------------------------------------------------------
    def _versions(self) -> list[str]:
        out = [f"mxo_utils {__version__}", f"python {_sys.version.split()[0]}",
               f"numpy {np.__version__}", f"pandas {pd.__version__}"]
        for mod in ("geopandas", "shapely", "pyogrio", "osmnx", "pandana", "scipy"):
            try:
                out.append(f"{mod} {__import__(mod).__version__}")
            except Exception:                                    # noqa: BLE001
                out.append(f"{mod} absent")
        return out

    def _inputs(self) -> pd.DataFrame:
        watch = [
            os.path.join(DIRS["inputs"], "metros_92.csv"),
            os.path.join(DIRS["results"], "01_network_metrics.parquet"),
            os.path.join(DIRS["results"], "03_blocks.parquet"),
            os.path.join(DIRS["results"], "04_tracts.parquet"),
            os.path.join(DIRS["results"], "05_outlets.parquet"),
            os.path.join(DIRS["results"], "07_block_access.parquet"),
            os.path.join(DIRS["results"], "08_tract_access.parquet"),
        ]
        rows = []
        for p in watch:
            if os.path.exists(p):
                st = os.stat(p)
                rows.append({
                    "file": os.path.basename(p),
                    "mb": round(st.st_size / 1e6, 2),
                    "modified_utc": datetime.fromtimestamp(
                        st.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                })
            else:
                rows.append({"file": os.path.basename(p), "mb": np.nan,
                             "modified_utc": "absent"})
        return pd.DataFrame(rows)

    def _header(self) -> str:
        inv = network_inventory()
        fp = network_fingerprint()
        caches = sorted(os.path.basename(p) for p in
                        _glob.glob(os.path.join(DIRS["results"], "cache_*")))
        lines = [
            f"# {self.name}",
            "",
            f"run (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment",
            "",
            "| item | value |",
            "| --- | --- |",
            f"| BASE | `{BASE}` |",
            f"| networks folder | `{DIRS['networks']}` |",
            f"| network fingerprint | `{fp}` |",
            f"| network files | {len(inv)}, {inv['mb'].sum():,.0f} MB total |",
            f"| versions | {', '.join(self._versions())} |",
            f"| cache dirs present | {', '.join(caches) if caches else 'none'} |",
            "",
            "### Input files",
            "",
            self._md_table(self._inputs()),
            "",
            "## Run log",
            "",
            "```",
        ]
        return "\n".join(lines) + "\n"

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
        if df is None or len(df) == 0:
            return "_(empty)_"
        d = df.reset_index() if df.index.name else df
        shown = d.head(max_rows)
        head = "| " + " | ".join(str(c) for c in shown.columns) + " |"
        rule = "| " + " | ".join("---" for _ in shown.columns) + " |"
        body = ["| " + " | ".join(
            ("" if pd.isna(v) else str(v)) for v in row) + " |"
            for row in shown.itertuples(index=False, name=None)]
        extra = ([f"", f"_{len(d) - max_rows} further rows omitted._"]
                 if len(d) > max_rows else [])
        return "\n".join([head, rule] + body + extra)

    # -- api ---------------------------------------------------------------
    def start(self):
        while _ACTIVE:                      # never nest tees
            _ACTIVE.pop().stop(quiet=True)
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        self._fh = open(self.local_path, "w", encoding="utf-8")
        self._fh.write(self._header())
        self._fh.flush()
        self._orig_stdout = _sys.stdout
        _sys.stdout = _Tee(self._orig_stdout, self._fh)
        _ACTIVE.append(self)
        print(f"[report] {self.name}: capturing to {self.local_path}")
        print(f"[report] networks {os.path.basename(DIRS['networks'])} "
              f"fingerprint {network_fingerprint()}")
        return self

    def note(self, text: str):
        self.notes.append(text)
        print(f"[note] {text}")
        return self

    def add_table(self, name: str, df: pd.DataFrame, max_rows: int = 200):
        """Keep a frame in the report in full, beyond whatever was printed."""
        try:
            self.tables[name] = df.copy()
            print(f"[report] table kept: {name} ({len(df)} rows, {max_rows} shown)")
        except Exception as exc:                                 # noqa: BLE001
            print(f"[report] could not keep table {name}: {exc}")
        return self

    def stop(self, quiet: bool = False):
        if self._fh is None:
            return self
        try:
            _sys.stdout = self._orig_stdout
        except Exception:                                        # noqa: BLE001
            pass
        if not quiet:
            pass
        try:
            self._fh.write("```\n")
            self._fh.flush()
        except Exception:                                        # noqa: BLE001
            pass
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        return self

    def save(self, download: bool = True):
        """Close the log, append the tables, copy to Drive, offer a download."""
        self.stop()
        try:
            if self.notes:
                self._fh.write("\n## Notes\n\n")
                for n in self.notes:
                    self._fh.write(f"- {n}\n")
            for name, df in self.tables.items():
                self._fh.write(f"\n## Table: {name}\n\n")
                self._fh.write(self._md_table(df) + "\n")
            self._fh.close()
        except Exception as exc:                                 # noqa: BLE001
            print(f"[report] problem finishing the file: {exc}")
        self._fh = None

        size_kb = os.path.getsize(self.local_path) / 1024.0
        if self.local_path != self.drive_path:
            try:
                os.makedirs(os.path.dirname(self.drive_path), exist_ok=True)
                import shutil
                shutil.copy(self.local_path, self.drive_path)
            except Exception as exc:                             # noqa: BLE001
                print(f"[report] could not copy to Drive: {exc}")
        print(f"[report] written: {self.local_path}  ({size_kb:,.0f} KB)")
        print(f"[report] on Drive: {self.drive_path}")
        if download:
            try:
                from google.colab import files
                files.download(self.local_path)
                print("[report] download started")
            except Exception:                                    # noqa: BLE001
                print("[report] not in Colab, so no automatic download")
        return self.local_path
