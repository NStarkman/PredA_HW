import json
import time
import urllib.request

import numpy as np
import pandas as pd

START = 2010
END = 2020
WB = "https://api.worldbank.org/v2"

TARGET = ("GB.XPD.RSDV.GD.ZS", "rd_pct_gdp")

WB_PREDICTORS = {
    "NY.GDP.PCAP.CD": "gdp_pc_usd", "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct", "NY.GNS.ICTR.ZS":"gross_savings_pct_gdp", "IT.NET.USER.ZS":"internet_pct",
    "IT.NET.BBND.P2":"broadband_per100", "IT.CEL.SETS.P2":"mobile_per100", "SE.TER.ENRR":"tertiary_enrol_pct", "SE.XPD.TOTL.GD.ZS": "gov_educ_pct_gdp", "SP.POP.TOTL":"population",
}

OWID_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
OWID_COLS = {
    "per_capita_electricity": "elec_gen_pc_kwh",
    "energy_per_capita":      "energy_use_pc_kwh",
    "low_carbon_share_elec":  "low_carbon_pct",
}

ISO_URL = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
LATLON_URL = "https://raw.githubusercontent.com/google/dspl/master/samples/google/canonical/countries.csv"

LEAKAGE_EXCLUDED = [
    "SP.POP.SCIE.RD.P6", "SP.POP.TECH.RD.P6",
    "IP.JRN.ARTC.SC", "IP.PAT.RESD", "TX.VAL.TECH.MF.ZS",
]

coverage = []


def note(step, n):
    coverage.append({"step": step, "countries": n})
    print(f"  {n:4d}   {step}")

#get each WB indicator
def wb_get(path, **params):
    params["format"] = "json"
    params["per_page"] = 20000
    query_parts = []
    for key, value in params.items():
        query_parts.append(f"{key}={value}")
    query = "&".join(query_parts)

    rows = []
    page = 1
    while True:
        url = f"{WB}/{path}?{query}&page={page}"
        with urllib.request.urlopen(url, timeout=120) as r:
            payload = json.load(r)
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            break
        rows.extend(payload[1])
        if page >= int(payload[0].get("pages", 1)):
            break
        page += 1
        time.sleep(0.3)
    return rows


def wb_real_countries():
    rows = wb_get("country")
    keep = [r for r in rows if r["region"]["value"] != "Aggregates"]
    data = {
        "iso3": [r["id"] for r in keep],
        "country": [r["name"] for r in keep],
        "wb_region": [r["region"]["value"] for r in keep],
        "income_group": [r["incomeLevel"]["value"] for r in keep],
    }
    return pd.DataFrame(data)


def wb_indicator(code, name):
    rows = wb_get(f"country/all/indicator/{code}", date=f"{START}:{END}")
    data = {
        "iso3": [r.get("countryiso3code") for r in rows],
        "year": [int(r["date"]) for r in rows],
        "value": [r["value"] for r in rows],
    }
    df = pd.DataFrame(data)
    has_value = df.value.notna()
    has_iso = df.iso3.notna()
    has_three_char_iso = df.iso3.str.len() == 3
    df = df[has_value & has_iso & has_three_char_iso]
    df = df.sort_values("year", ascending=False).drop_duplicates("iso3")
    return df.rename(columns={"value": name, "year": name + "_year"})


print(f"window {START}-{END}\n")
print("count  step")

base = wb_real_countries()
note("Country list", len(base))

code, tname = TARGET
tgt = wb_indicator(code, tname)
note(f"have the target ({code})", len(tgt))

table = base.merge(tgt, on="iso3", how="inner")
note("target joined to the country list", len(table))
for code, name in WB_PREDICTORS.items():
    ind = wb_indicator(code, name)
    table = table.merge(ind[["iso3", name]], on="iso3", how="left")
    have = int(table[name].notna().sum())
    note(f"+ {name} ({code}) - countries that have it, of {len(table)}", have)

owid = pd.read_csv(OWID_URL, low_memory=False)
in_years = (owid.year >= START) & (owid.year <= END)
has_iso = owid.iso_code.notna()
has_three_char_iso = owid.iso_code.str.len() == 3
owid = owid[in_years & has_iso & has_three_char_iso]
pieces = []
for src, name in OWID_COLS.items():
    one = owid.loc[owid[src].notna(), ["iso_code", "year", src]]
    one = one.sort_values("year", ascending=False).drop_duplicates("iso_code")
    one = one.rename(columns={"iso_code": "iso3", src: name})
    pieces.append(one[["iso3", name]])
energy = pieces[0]
for p in pieces[1:]:
    energy = energy.merge(p, on="iso3", how="outer")
note(f"OWID/Ember energy table ({len(OWID_COLS)} columns)", len(energy))

table = table.merge(energy, on="iso3", how="left")
note("energy joined on ISO3", len(table))

iso = pd.read_csv(ISO_URL)[["alpha-2", "alpha-3", "region", "sub-region"]]
latlon = pd.read_csv(LATLON_URL)[["country", "latitude"]]
geo = iso.merge(
    latlon,
    left_on="alpha-2",
    right_on="country",
    how="inner",
)
geo = geo.rename(columns={"alpha-3": "iso3", "region": "iso_region"})
geo["abs_latitude"] = geo.latitude.abs()
note("ISO 3166 codes joined to coordinates", len(geo))

table = table.merge(geo[["iso3", "iso_region", "abs_latitude"]], on="iso3", how="left")
note("geography joined on ISO3", len(table))

region = pd.get_dummies(table.iso_region, prefix="region").astype(int)
table = pd.concat([table, region], axis=1)

table["log_population"] = np.log10(table.population)
table["log_gdp_pc"] = np.log10(table.gdp_pc_usd)
table = table.drop(columns=["population", "gdp_pc_usd"])

region = pd.get_dummies(table.iso_region, prefix="region").astype(int)

not_predictors = [
    "iso3", "country", "wb_region", "income_group",
    "iso_region", tname, tname + "_year",
]
PRED = []
for column in table.columns:
    if column not in not_predictors and not column.endswith("_year"):
        PRED.append(column)

print(f"\nfinal table: {len(table)} countries, {len(PRED)} predictors")
print(f"target present: {table[tname].notna().sum()} rows (need 120)")
print(f"target: {tname} (R&D spending / GDP)")


table.to_csv("partC_table.csv", index=False)
pd.DataFrame(coverage).to_csv("partC_coverage.csv", index=False)
