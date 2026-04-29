"""Genererer baseline_modeller.ipynb fra strukturert kildekode.

Kjor manuelt etter endringer:
    python _bygg_notebook.py
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "baseline_modeller.ipynb"

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(r"""# Baseline-modeller for prediksjon av lakseeksportpris

**LOG650 – Forskningsprosjekt: Logistikk og kunstig intelligens**

Dette notebooket implementerer baseline-modellene som er beskrevet i prosjektplanen:

1. **Naiv baseline** – `pris(t+h) = pris(t)`
2. **SARIMA** – tradisjonell tidsseriemodell (sesongkomponent uke)
3. **XGBoost** – maskinlæring med direkte multi-horisont-prediksjon

Modellene evalueres på prognosehorisontene **4, 8 og 12 uker** med MAE og MAPE.

Datasett: `004 data/Analyseklart datasett/laks_ukentlig_features.csv`.
""")

md("## 1. Imports og oppsett")

code("""import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (11, 4)

DATA_DIR = Path("..") / "004 data" / "Analyseklart datasett"
HORISONTER = [4, 8, 12]
TEST_UKER = 104  # siste 2 ar som testset
""")

md("## 2. Last inn data")

code("""df = pd.read_csv(
    DATA_DIR / "laks_ukentlig_features.csv",
    parse_dates=["uke_start"],
).set_index("uke_start").sort_index()

print(f"Periode: {df.index.min().date()} til {df.index.max().date()}")
print(f"Antall uker: {len(df)}")
df.head()
""")

code("""fig, ax = plt.subplots()
df["eksport_pris_nok_kg"].plot(ax=ax, color="steelblue")
ax.set_title("Ukentlig eksportpris fersk norsk laks (NOK/kg)")
ax.set_xlabel("")
ax.set_ylabel("NOK/kg")
ax.grid(alpha=0.3)
plt.tight_layout()
""")

md("""## 3. Train/test-split

Kronologisk split: siste 104 uker (~2 år) holdes av som testset.
""")

code("""train = df.iloc[:-TEST_UKER].copy()
test = df.iloc[-TEST_UKER:].copy()
print(f"Train: {train.index.min().date()} til {train.index.max().date()}  ({len(train)} uker)")
print(f"Test:  {test.index.min().date()} til {test.index.max().date()}  ({len(test)} uker)")
""")

md("""## 4. Hjelpefunksjoner – evaluering""")

code("""def evaluer(y_true: pd.Series, y_pred: pd.Series, modell: str, horisont: int) -> dict:
    \"\"\"Returnerer MAE og MAPE etter at NaN er droppet.\"\"\"
    par = pd.concat([y_true, y_pred], axis=1).dropna()
    if par.empty:
        return {"modell": modell, "horisont": horisont, "n": 0, "MAE": np.nan, "MAPE": np.nan}
    return {
        "modell": modell,
        "horisont": horisont,
        "n": len(par),
        "MAE": mean_absolute_error(par.iloc[:, 0], par.iloc[:, 1]),
        "MAPE": mean_absolute_percentage_error(par.iloc[:, 0], par.iloc[:, 1]),
    }

resultater = []
prognoser = {}  # (modell, horisont) -> Series av prediksjoner indeksert pa test-datoer
""")

md("""## 5. Modell A – Naiv baseline

`pris(t+h) = pris(t)`. Definerer minimumsnivaet for ML-modellene.
""")

code("""for h in HORISONTER:
    pred = test["eksport_pris_nok_kg"].shift(h)  # bruker kjent verdi h uker tilbake
    resultater.append(evaluer(test["eksport_pris_nok_kg"], pred, "Naiv", h))
    prognoser[("Naiv", h)] = pred
""")

md("""## 6. Modell B – SARIMA

Sesongkomponent **s=52** (arlig sykl). Bruker SARIMA(1,1,1)(1,1,1,52).
Fittes pa treningsdata, deretter forecastes hele testperioden i ett – horisont vokser
med tid siden train-end. Ved sluttrapport bor denne erstattes med rullende-opphav-evaluering
for fair sammenligning med direkte XGBoost.
""")

code("""y_train = train["eksport_pris_nok_kg"].asfreq("W-MON")
y_test = test["eksport_pris_nok_kg"].asfreq("W-MON")

print("Tilpasser SARIMA(1,1,1)(1,1,1,52) ... (kan ta noen minutter)")
sarima = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)
print(sarima.summary().tables[0])
""")

code("""# Forecast hele testperioden i ett. For hver horisont h registrerer vi prediksjonen
# som ble laget h uker for hver test-uke (dvs. samme element-vis sammenligning som for naiv).
forecast_full = sarima.get_forecast(steps=len(y_test)).predicted_mean
forecast_full.index = y_test.index

for h in HORISONTER:
    # SARIMA-prediksjonen for tidspunkt t (laget h uker tilbake) er bare relevant for t > train_end + h.
    # Med single-fit + multi-step forecast er prediksjonen for t laget t-train_end skritt frem,
    # ikke h skritt. Dette er en grov tilnaerming - se kommentar over.
    pred = forecast_full.copy()
    resultater.append(evaluer(y_test, pred, "SARIMA", h))
    prognoser[("SARIMA", h)] = pred
""")

md("""## 7. Modell C – XGBoost (direkte multi-horisont)

Trener én XGBoost-modell per horisont. Mål: `eksport_pris_nok_kg.shift(-h)`.
Forklaringsvariabler: alle features unntatt fremtidige verdier.
""")

code("""# Velg features - dropp identifikatorer og rene fremtids-leakage-kolonner
feature_kols = [c for c in df.columns if c not in {
    "iso_aar", "iso_uke", "uke_kode",
    "eksport_pris_nok_kg",      # malvariabel
    "fao_global_atlantisk_tonn", # NaN i 2023+
    "fao_norge_tonn",
    "fao_eks_norge_tonn",
}]
print(f"Antall features: {len(feature_kols)}")
print(feature_kols)
""")

code("""xgb_modeller = {}

for h in HORISONTER:
    y = df["eksport_pris_nok_kg"].shift(-h)  # mal h uker frem
    X = df[feature_kols]
    data = pd.concat([X, y.rename("target")], axis=1).dropna()

    # Re-bruk samme tidsbaserte split som tidligere
    cutoff = train.index.max()
    train_mask = data.index <= cutoff
    X_tr, y_tr = data.loc[train_mask, feature_kols], data.loc[train_mask, "target"]
    X_te, y_te = data.loc[~train_mask, feature_kols], data.loc[~train_mask, "target"]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    pred = pd.Series(model.predict(X_te), index=X_te.index)

    # Test-perioden i evaluering: prediksjon for t (laget pa X[t-h]) sammenlignes
    # med faktisk pris pa t. Mal i 'data' er allerede pris(t+h), sa vi flytter
    # prediksjons-indeksen frem h uker for sammenligning mot test["eksport_pris_nok_kg"].
    pred.index = pred.index + pd.Timedelta(weeks=h)
    pred = pred.reindex(test.index)

    resultater.append(evaluer(test["eksport_pris_nok_kg"], pred, "XGBoost", h))
    prognoser[("XGBoost", h)] = pred
    xgb_modeller[h] = model
""")

md("""## 8. Sammenligning""")

code("""res = pd.DataFrame(resultater).pivot(index="horisont", columns="modell", values=["MAE", "MAPE"])
res = res.reorder_levels([1, 0], axis=1).sort_index(axis=1)
res
""")

code("""# Plot prediksjoner mot fasit for hver horisont
fig, axes = plt.subplots(len(HORISONTER), 1, figsize=(11, 9), sharex=True)
for ax, h in zip(axes, HORISONTER):
    ax.plot(test.index, test["eksport_pris_nok_kg"], label="Faktisk", color="black", lw=1.2)
    for modell, farge in [("Naiv", "tab:gray"), ("SARIMA", "tab:orange"), ("XGBoost", "tab:blue")]:
        s = prognoser[(modell, h)]
        ax.plot(s.index, s, label=modell, color=farge, alpha=0.8)
    ax.set_title(f"Horisont {h} uker")
    ax.set_ylabel("NOK/kg")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
""")

md("""## 9. Feature importance (XGBoost)

Hvilke variabler bidrar mest? Vist for horisont 4 uker.
""")

code("""h_vis = 4
imp = pd.Series(
    xgb_modeller[h_vis].feature_importances_,
    index=feature_kols,
).sort_values(ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(8, 6))
imp.plot.barh(ax=ax, color="steelblue")
ax.set_title(f"XGBoost – top features for horisont {h_vis} uker")
ax.set_xlabel("Feature importance")
plt.tight_layout()
""")

md("""## 10. Neste steg

- **Rullende-opphav-evaluering for SARIMA** for fair sammenligning ved fast horisont
- **Hyperparameter-tuning** av XGBoost (RandomizedSearchCV med TimeSeriesSplit)
- **Auto-ARIMA** (`pmdarima`) for å finne ordre automatisk
- **LightGBM** som alternativ til XGBoost (begge er nevnt i prosjektplanen)
- **FAO-data**: håndter manglende verdier 2023–2026 (forward-fill / dropp / ekstrapoler)
- **Eksogene variabler i SARIMA** (SARIMAX med valuta som regressor)
- **Konfidensintervaller** for SARIMA-prognosene
- **Feature engineering**: differanser, ratio mellom EUR/USD, akkumulert volum
""")


nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Skrev {OUT.name} ({len(cells)} celler)")
