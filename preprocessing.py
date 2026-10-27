"""Feature preprocessing for the BRFSS heart-disease (MICHD) dataset.

The pipeline is fit on the training set only and then applied identically to
the test set (no information from the test set is used).  Every step can be
switched off through the ``config`` dictionary, which is how the ablation study
in ``experiments.py`` is produced.

Steps
-----
1. Drop administrative / survey-design columns (dates, ids, sampling weights)
   and raw columns that are duplicated by a cleaner CDC-computed variable.
2. Drop columns whose missing fraction exceeds ``max_nan``.
3. Split the remaining columns into categorical (<= ``max_categories``
   distinct values, plus ``_STATE``) and continuous.
4. Categorical -> one-hot encoding (NaN gets its own indicator, rare levels
   are dropped).  BRFSS answers such as 7 = "don't know" / 9 = "refused"
   thereby become their own indicators instead of being treated as numbers.
5. Continuous -> decode BRFSS special codes (88 = "none" -> 0, 77/99/777/999
   -> missing), log-transform heavy-tailed variables, clip outliers at the
   1st/99th percentile, median-impute, add a missing-value indicator and
   standardise.
6. Optional squared continuous features (captures U-shaped effects, e.g.
   BMI or sleep, that a linear model cannot represent).
"""

import numpy as np

# Survey administration, identifiers, interview dates and sampling weights:
# they describe *how* the respondent was interviewed, not their health.
ADMIN_COLUMNS = [
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE",
    "LANDLINE", "QSTVER", "QSTLANG", "_STSTR", "_STRWT", "_RAWRAKE",
    "_WT2RAKE", "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT",
]  # fmt: skip

# Raw answers with a mixed unit encoding (e.g. 1xx = per day, 2xx = per week)
# for which the CDC already provides a cleaned, computed variable.
RAW_DUPLICATE_COLUMNS = [
    "WEIGHT2", "HEIGHT3",                    # -> WTKG3, HTM4, _BMI5
    "ALCDAY5",                               # -> DROCDY3_
    "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",
    "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2", "STRENGTH",  # -> PA*_
    "EXRACT11", "EXRACT21",                  # activity type (75 codes)
    "FLSHTMY2", "HIVTSTD3",                  # month/year dates
]  # fmt: skip

# Columns treated as categorical even though they have many levels.
FORCED_CATEGORICAL = ["_STATE"]

# Special codes of continuous survey answers: value -> replacement.
SPECIAL_CODES = {
    "PHYSHLTH": {88: 0, 77: np.nan, 99: np.nan},
    "MENTHLTH": {88: 0, 77: np.nan, 99: np.nan},
    "POORHLTH": {88: 0, 77: np.nan, 99: np.nan},
    "CHILDREN": {88: 0, 77: np.nan, 99: np.nan},
    "HHADULT": {77: np.nan, 99: np.nan},
    "AVEDRNK2": {77: np.nan, 99: np.nan},
    "MAXDRNKS": {77: np.nan, 99: np.nan},
    "DRNK3GE5": {88: 0, 77: np.nan, 99: np.nan},
    "DIABAGE2": {98: np.nan, 99: np.nan},
    "DROCDY3_": {900: np.nan},
    "_DRNKWEK": {99900: np.nan},
    "MAXVO2_": {999: np.nan},
    "FC60_": {999: np.nan},
}

DEFAULT_CONFIG = {
    "drop_admin": True,  # step 1
    "max_nan": 0.9,  # step 2
    "max_categories": 15,  # step 3
    "one_hot": True,  # step 4 (False -> categorical columns kept as numbers)
    "min_level_count": 50,  # rare categorical levels are dropped
    "decode_special": True,  # step 5
    "log_transform": True,  # step 5
    "clip": True,  # step 5
    "missing_indicator": True,  # step 5
    "squares": False,  # step 6 (add z^2 of every continuous feature)
}


def read_column_names(x_train_csv_path):
    """Read the feature names from the header of x_train.csv (without 'Id')."""
    with open(x_train_csv_path) as f:
        header = f.readline().strip().split(",")
    return header[1:]


class Preprocessor:
    """Fit-on-train / transform-anything feature pipeline.

    Usage:
        prep = Preprocessor(column_names, config)
        x_train_feat = prep.fit_transform(x_train)
        x_test_feat = prep.transform(x_test)
    """

    def __init__(self, column_names, config=None):
        self.names = list(column_names)
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)

    # ------------------------------------------------------------------ fit
    def fit(self, x):
        """Learn every statistic of the pipeline from the training matrix x."""
        cfg = self.config
        name_to_idx = {n: i for i, n in enumerate(self.names)}
        x = self._decode(x)
        nan_frac = np.isnan(x).mean(axis=0)

        dropped = set()
        if cfg["drop_admin"]:
            dropped |= set(ADMIN_COLUMNS) | set(RAW_DUPLICATE_COLUMNS)
        self.cat_cols, self.num_cols = [], []
        for j, name in enumerate(self.names):
            if name in dropped or nan_frac[j] > cfg["max_nan"]:
                continue
            col = x[:, j]
            n_unique = len(np.unique(col[~np.isnan(col)]))
            if n_unique <= 1:
                continue
            if name in FORCED_CATEGORICAL or n_unique <= cfg["max_categories"]:
                self.cat_cols.append(j)
            else:
                self.num_cols.append(j)
        if not cfg["one_hot"]:  # categorical columns treated as plain numbers
            self.num_cols = sorted(self.num_cols + self.cat_cols)
            self.cat_cols = []

        # Categorical levels (NaN encoded as the sentinel -1).
        self.levels = []
        for j in self.cat_cols:
            col = np.where(np.isnan(x[:, j]), -1.0, x[:, j])
            values, counts = np.unique(col, return_counts=True)
            self.levels.append(values[counts >= cfg["min_level_count"]])

        # Continuous statistics.
        num = x[:, self.num_cols].astype(np.float64)
        self.log_mask = np.zeros(len(self.num_cols), dtype=bool)
        if cfg["log_transform"]:
            for k in range(num.shape[1]):
                c = num[:, k][~np.isnan(num[:, k])]
                if c.min() >= 0 and _skewness(c) > 2.0:
                    self.log_mask[k] = True
        num = self._log(num)
        self.low = np.nanpercentile(num, 1, axis=0)
        self.high = np.nanpercentile(num, 99, axis=0)
        if cfg["clip"]:
            num = np.clip(num, self.low, self.high)
        self.median = np.nanmedian(num, axis=0)
        self.has_nan = np.isnan(num).any(axis=0) & cfg["missing_indicator"]
        num = np.where(np.isnan(num), self.median, num)
        self.mean = num.mean(axis=0)
        self.std = num.std(axis=0) + 1e-8

        if cfg["squares"]:
            z2 = ((num - self.mean) / self.std) ** 2
            self.sq_mean = z2.mean(axis=0)
            self.sq_std = z2.std(axis=0) + 1e-8
        self.feature_names = self._build_feature_names()
        return self

    # ------------------------------------------------------------ transform
    def transform(self, x):
        """Apply the fitted pipeline; returns a float32 matrix with a bias."""
        cfg = self.config
        x = self._decode(x)
        blocks = [np.ones((x.shape[0], 1), dtype=np.float32)]  # bias column

        for j, levels in zip(self.cat_cols, self.levels):
            col = np.where(np.isnan(x[:, j]), -1.0, x[:, j])
            blocks.append((col[:, None] == levels[None, :]).astype(np.float32))

        num = self._log(x[:, self.num_cols].astype(np.float64))
        nan_mask = np.isnan(num)
        if cfg["clip"]:
            num = np.clip(num, self.low, self.high)
        num = np.where(nan_mask, self.median, num)
        z = (num - self.mean) / self.std
        blocks.append(z.astype(np.float32))
        blocks.append(nan_mask[:, self.has_nan].astype(np.float32))

        if cfg["squares"]:
            blocks.append(((z**2 - self.sq_mean) / self.sq_std).astype(np.float32))
        return np.hstack(blocks)

    def fit_transform(self, x):
        """Fit on x then transform x."""
        return self.fit(x).transform(x)

    # -------------------------------------------------------------- helpers
    def _decode(self, x):
        """Replace BRFSS special codes of continuous answers (copy of x)."""
        x = np.array(x, dtype=np.float64, copy=True)
        if not self.config["decode_special"]:
            return x
        for j, name in enumerate(self.names):
            for code, value in SPECIAL_CODES.get(name, {}).items():
                x[x[:, j] == code, j] = value
        return x

    def _log(self, num):
        """log(1 + x) on the heavy-tailed continuous columns."""
        if self.log_mask.any():
            num[:, self.log_mask] = np.log1p(num[:, self.log_mask])
        return num

    def _build_feature_names(self):
        """Human-readable name for every output column (for inspection)."""
        names = ["bias"]
        for j, levels in zip(self.cat_cols, self.levels):
            names += [f"{self.names[j]}={'NaN' if v == -1 else int(v)}" for v in levels]
        num_names = [self.names[j] for j in self.num_cols]
        names += num_names
        names += [f"{n}_missing" for n, m in zip(num_names, self.has_nan) if m]
        if self.config["squares"]:
            names += [f"{n}^2" for n in num_names]
        return names


def _skewness(c):
    """Sample skewness of a 1D array."""
    s = c.std()
    return 0.0 if s == 0 else np.mean(((c - c.mean()) / s) ** 3)
