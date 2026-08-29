# LEARNING CURVES — the last rubric gap

Add this as the final cell of Phase 11, after Cell 11.2. Runtime roughly 30–90 seconds.

```python
# ============================================================
# PHASE 11 — CELL 3 — Learning curves (rubric requirement)
# ============================================================

from sklearn.model_selection import TimeSeriesSplit, learning_curve

# TimeSeriesSplit, not KFold: ordinary cross-validation would train on
# future data to predict the past. TimeSeriesSplit always trains on an
# earlier block and validates on the block immediately after it.
tscv = TimeSeriesSplit(n_splits=4)
train_sizes = np.linspace(0.2, 1.0, 5)

# Rebuild the primary design matrix on the TRAINING period only — the
# learning curve is an internal-validation diagnostic and must not touch
# the 2018 test set.
X_lc = PRIMARY_BUILDER(train_df, ref_columns=PRIMARY_REF)
y_lc = train_yr.to_numpy(dtype=float)

# Forest reduced to 100 trees here purely for runtime: the curve refits
# it 20 times (5 sizes x 4 folds). Shape is what matters, not the last
# decimal place.
curve_models = {
    "Linear + interactions (primary)": LinearRegression(),
    "Random forest": RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1),
}

curves = {}
for name, mdl in curve_models.items():
    sizes, train_scores, val_scores = learning_curve(
        mdl, X_lc, y_lc,
        train_sizes=train_sizes,
        cv=tscv,
        scoring="neg_mean_absolute_error",   # negated: sklearn maximises
        shuffle=False,                       # never shuffle a time series
        n_jobs=1,
    )
    curves[name] = (sizes, -train_scores, -val_scores)

    tr_final = (-train_scores).mean(axis=1)[-1]
    va_final = (-val_scores).mean(axis=1)[-1]
    print(f"{name}")
    print(f"  final training MAE:   {tr_final:8.1f} MW")
    print(f"  final validation MAE: {va_final:8.1f} MW")
    print(f"  generalisation gap:   {va_final - tr_final:8.1f} MW\n")

fig, axes = plt.subplots(1, len(curves), figsize=(7 * len(curves), 5), squeeze=False)
for ax, (name, (sizes, tr, va)) in zip(axes[0], curves.items()):
    tr_m, tr_s = tr.mean(axis=1), tr.std(axis=1)
    va_m, va_s = va.mean(axis=1), va.std(axis=1)

    ax.plot(sizes, tr_m, marker="o", label="Training MAE")
    ax.plot(sizes, va_m, marker="s", label="Validation MAE")
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15)
    ax.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.15)

    ax.set_xlabel("Training set size (hours)")
    ax.set_ylabel("Mean Absolute Error (MW)")
    ax.set_title(name)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## How to read your output

**Gap between the two lines** = overfitting. A forest with `min_samples_leaf=5`
memorises individual hours, so expect its training MAE to sit well below its
validation MAE. The linear model has far fewer effective parameters, so its two
lines should nearly coincide.

**Slope of the validation line at the right-hand edge** = whether more data would
help. Still falling → four years is limiting you. Flat → the ceiling is the feature
set, not the sample size. Given that weather and calendar features cannot explain
demand shocks like public holidays or industrial scheduling, expect it to flatten.

**Both of those support choices you already made.** A small linear gap plus a large
forest gap is an additional argument for the primary specification, independent of the
R² comparison in Table 2. Report it that way.

---

## Text for the report (replaces the Figure 10 placeholder, Section 3.12)

Paste this in and substitute your actual numbers where marked.

> ### 3.13 Learning curves
>
> Figure 10 reports learning curves for the primary specification and the random
> forest, computed on the training period only using a four-fold expanding-window
> time-series split, so that every validation fold follows its training fold
> chronologically. Ordinary k-fold cross-validation would be invalid here, since it
> would train on later observations to predict earlier ones.
>
> Two features are relevant. First, the gap between training and validation error
> differs markedly between the models: the primary specification's curves converge
> to within [X] MW, whereas the random forest retains a gap of [Y] MW. The forest,
> permitted five samples per leaf, fits individual hours in a way that does not
> generalise; the linear specification, with far fewer effective parameters, does
> not. This provides support for the primary specification independent of the
> out-of-sample comparison in Table 2.
>
> Second, the validation curve [flattens / continues to decline] as training size
> approaches its maximum. [If flat:] This indicates that predictive performance is
> limited by the information content of the weather and calendar feature set rather
> than by sample size, which is consistent with the residual analysis in Section 3.1
> — the largest errors coincide with public holidays, which no weather or calendar
> feature in this model can anticipate. Additional years of data at the same feature
> resolution would not be expected to improve accuracy materially. [If declining:]
> This indicates that the four-year record is a binding constraint and that a longer
> series would likely improve accuracy.

Then update Figure 10's caption to match what you actually see, and delete the
bracketed alternatives you don't use.
