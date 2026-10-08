"""
The original New Training Docs/MVODB/mvo4.ipynb logic, copied cell by cell into
one function so it can be run against the same inputs as the sweep engine.

This file is deliberately NOT tidied: statements are copied verbatim from the
notebook (a copy of which is in this folder, mvo4_original_MVODB.ipynb), with the
notebook cell id above each block. Only these adaptations were made, each marked
"# ADAPTED":

  1. yf.download(...) is replaced by slicing the same cached price table the
     engine uses, so both sides see identical prices. The training slice includes
     `end` (the notebook's yf.download excludes it; see README, "Dates"), and the
     evaluation slice starts at `start_swan_date` (the engine's buy_date).
  2. Cell 7335a9dc's look-ahead bug (avg = the target swan's own beta) is replaced
     by the notebook's own commented-out fix (mean over the training swans).
  3. `portfolio_beta` is passed in restricted to the chronological swan columns
     (config.SWAN_ORDER), so the notebook's `start_swan:end_swan` label slices pick
     exactly the training swans, including the merged-in Swan10D column.
  4. Inputs (dates, K, Seeds, ...) are function arguments instead of cell-14
     globals, and the hard-coded evaluation dates in cells 9bf934f6 / b45e8a54 use
     the event's peak and bottom.
  5. The notebook's positional Series lookups (selected_beta["Group"][0], and
     temp["portfolio_value"][1]) are written with .iloc. Same meaning in pandas 2.x,
     where the notebook's form only raises a FutureWarning; pandas 3 removes it.

Everything else (the K-means voting, the crash curve, the injection, the
drawdown loop, the covariance, E_2, max_sharpe(0.00), the 3% floor, the PR
calculation, and data_cleaning with its original indexing) is the notebook's code.
"""
import datetime
import math

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from sklearn import metrics
from sklearn.cluster import KMeans


# --- cell a375b8e8 (verbatim, including its row-index bug; prints removed) ---
def data_cleaning(data):
    if (data.isna()).any().any():
        missing = np.where(data.isna())
        rows = missing[0]
        cols = missing[1]
        for row in range(len(rows)):
            col = cols[row]
            if row == len(data) - 1:
                data.iloc[row, col] = data.iloc[row - 1, col] ** 2 / data.iloc[row - 2, col]
            elif row == 0:
                data.iloc[row, col] = data.iloc[row + 1, col] ** 2 / data.iloc[row + 2, col]
            else:
                data.iloc[row, col] = math.sqrt(data.iloc[row - 1, col] * data.iloc[row + 1, col])


# --- cell 7fe29e3f ---
def get_price_years_ago(data, date, time):
    year_ago = date - datetime.timedelta(days=int(time * 365))
    price_year_ago = data[data.index <= year_ago]
    return price_year_ago


# --- cell 74e1c1f8 ---
def max_drawdown_absolute(returns):
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.argmin()
    start = r.iloc[:end].argmax()
    return abs(mdd), start, end


def run_notebook_logic(portfolio_beta, market_drop_info, optdata_path, px_all, oex_all,
                       start, end, peak, bottom, start_swan, end_swan,
                       N, Seeds, Min_appear, w1, selected_stock_number=22, start_swan_date=None):
    """Returns the notebook's intermediates and PR for one configuration."""
    out = {}

    # ADAPTED (4): cell d54d1018 parameters come in as arguments.
    # ADAPTED (1): download shim over the cached price table.
    def download(tickers, s, e):
        return px_all.loc[s:e, list(tickers)]

    # --- cell a1c7e6cb ---
    selected_swans_var = portfolio_beta.loc[:, start_swan:end_swan]
    portfolio_beta = portfolio_beta[selected_swans_var.T.isnull().sum() < selected_swans_var.shape[1] - 1]

    # --- cell 6f9b6522 ---
    corr = portfolio_beta.corr()
    selected_corr = corr.loc["Swan1":end_swan, "Swan1":end_swan]
    for i in range(selected_corr.shape[1]):
        for j in range(i + 1, selected_corr.shape[0]):
            selected_corr.iloc[j, i] = np.nan

    # --- cell 70e36794 ---
    avg = selected_corr.mean(axis=1)
    weight_swan = avg / sum(avg)

    # --- cell b2d69e9c, method == "K_means_multiple" ---
    selected_beta = portfolio_beta.loc[:, start_swan:end_swan].dropna()
    Y_predic = selected_beta[end_swan].copy()
    seed_dict = {}
    for r in range(1, 1001):
        Kmean = KMeans(n_clusters=N, random_state=r, n_init=10)
        Kmean.fit(selected_beta)
        labels = Kmean.labels_
        dbscore = metrics.davies_bouldin_score(selected_beta, labels)
        seed_dict[r] = dbscore
    sorted_dict = sorted(seed_dict.items(), key=lambda x: x[1])
    out["dbi_ranking"] = sorted_dict

    stock_list = []
    max_length = 0
    for index in range(0, Seeds):
        selected_beta = portfolio_beta.loc[:, start_swan:end_swan].dropna()
        Kmean = KMeans(n_clusters=N, random_state=sorted_dict[index][0], n_init=10)
        Kmean.fit(selected_beta)
        ct = Kmean.cluster_centers_
        selected_beta["Group"] = Kmean.labels_
        res = ct.mean(axis=1)
        res[res < 0] = 0
        for i in range(N):
            Y_predic[selected_beta["Group"] == i] = res[i]
        selected_beta["avg"] = Y_predic.values
        selected_beta = selected_beta.sort_values(by=["avg"])
        first_group = selected_beta[selected_beta["Group"] == selected_beta["Group"].iloc[0]]  # ADAPTED (5)
        if first_group.shape[0] < 10:
            second_group = selected_beta[selected_beta["Group"] == selected_beta["Group"].iloc[first_group.shape[0]]]  # ADAPTED (5)
            preselected_stock = pd.concat([first_group, second_group], axis=0, ignore_index=False)
        else:
            preselected_stock = first_group
        if preselected_stock.shape[0] > max_length:
            max_length = preselected_stock.shape[0]
        stock_list.extend(preselected_stock.index.values)

    values, counts = np.unique(stock_list, return_counts=True)
    count_dict = {}
    preselected_stock = []
    for t in range(0, len(values)):
        count_dict[values[t]] = counts[t]
    sorted_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    for p in range(0, min(selected_stock_number, len(sorted_count))):
        if sorted_count[p][1] >= Min_appear:
            preselected_stock.append(sorted_count[p][0])
    preselected_stock = sorted(preselected_stock)
    selected_beta = selected_beta.loc[preselected_stock]
    selected_beta["avg"] = selected_beta.loc[:, start_swan:end_swan].mean(axis=1)
    preselected_stock = selected_beta

    # --- cell 7335a9dc --- ADAPTED (2): the notebook's commented-out correct line
    preselected_stock["avg"] = preselected_stock.loc[:, start_swan:end_swan].mean(axis=1)
    out["preselected"] = list(preselected_stock.index)

    # --- cell 8e194840 ---
    selected_stock = preselected_stock
    selected_stock_list = selected_stock.index.values
    swan_beta = preselected_stock.loc[selected_stock_list, "avg"]

    # --- cell 2bd97cf4 --- ADAPTED (1)
    yahooDataFrame = download(list(selected_stock_list), start, end)
    yahooDataFrame = yahooDataFrame[selected_stock_list]

    # --- cell b807776f ---
    total_day = yahooDataFrame.shape[0]
    t1 = int((np.array(market_drop_info.loc[:end_swan, "T1"]) * np.array(weight_swan)).sum())
    t2 = int(round(market_drop_info.loc[:end_swan, "T2"].mean(), 0))
    dd = market_drop_info.loc[:end_swan, "Actual DD"].mean()
    t0 = int(total_day - t1 - t2)
    out.update(t1=t1, t2=t2, dd=dd, t0=t0, total_day=total_day)

    # --- cell 451fc9c4 ---
    md = []
    j = 0
    inc = dd / (t1 - 1)
    for i in range(t1):
        md.append(round(j, 4))
        j = j + inc
    base = (t2 + 1) ** (1 / (dd * 100))
    l1 = []
    for i in range(t2):
        l1.append(math.log((i + 2), base) / 100)
    for i in range(t2):
        md.append(round((dd - l1[i]), 4))
    for i in range(t0):
        md.append(0)
    market_drop = pd.DataFrame(md)

    # --- cell 3be4c870 ---
    swan_drop_pct = np.array(swan_beta) * np.array(market_drop)
    yahooDataFrame = pd.DataFrame((1 - swan_drop_pct) * np.array(yahooDataFrame), index=yahooDataFrame.index,
                                  columns=yahooDataFrame.columns)
    yahooDataFrame = yahooDataFrame.dropna(how="all", axis=1)
    selected_stock_list = yahooDataFrame.columns.values
    selected_stock_number = yahooDataFrame.shape[1]

    # --- cell 668e84bf ---
    data_cleaning(yahooDataFrame)

    # --- cell 47a07252 ---
    one_year_ago_data = get_price_years_ago(yahooDataFrame, end, 1)
    one_year_ago_date = one_year_ago_data.index.values[-1]
    stock_date = yahooDataFrame.index.values
    week = pd.to_datetime(stock_date).strftime("%W").values
    week_1 = np.roll(week, 1)
    week_difference = np.where(week != week_1)
    week_difference = np.delete(week_difference, 0, None)
    weekly_return = yahooDataFrame.iloc[week_difference - 1].pct_change().dropna()
    log_weekly_return = np.log(1 + weekly_return)
    one_year_weekly_return = log_weekly_return.iloc[log_weekly_return.index >= one_year_ago_date]

    # --- cell 25f95798 ---
    daily_return = yahooDataFrame.pct_change().dropna()
    stock_return_one_year = daily_return.loc[one_year_ago_date:, :]
    DD_range = 90
    total_range = 90
    DD_end = yahooDataFrame.iloc[yahooDataFrame.index <= one_year_ago_date]
    DD_start = yahooDataFrame.iloc[yahooDataFrame.index <= end - datetime.timedelta(total_range)]
    max_DD = [-1] * selected_stock_number
    if (DD_start.shape[0] != 1) and (DD_end.shape[0] != 1):
        DD_start = DD_start.index[-1]
        DD_end = DD_end.index[-1]
        while DD_start > DD_end:
            DD_start = yahooDataFrame.iloc[yahooDataFrame.index <= end - datetime.timedelta(total_range)]
            DD_mid = yahooDataFrame.iloc[yahooDataFrame.index <= end - datetime.timedelta(DD_range + total_range)]
            if (DD_start.shape[0] != 1) and (DD_mid.shape[0] != 1):
                DD_start = DD_start.index[-1]
                DD_mid = DD_mid.index[-1]
                price = yahooDataFrame.loc[DD_start:DD_mid, :]
                for i in range(len(yahooDataFrame.columns)):
                    max_DD[i] = max(max_DD[i], max_drawdown_absolute(stock_return_one_year.iloc[:, i])[0])
                total_range = total_range + 1
            else:
                break

    # --- cell 389215e9 ---
    opt = pd.read_excel(optdata_path, index_col=0).loc[preselected_stock.index]
    R_fwd = opt["R_fwd_30_MA"]
    fall = pd.Series(max_DD, index=preselected_stock.index)

    # --- cell 3fea5d8b ---
    cov = pd.DataFrame(np.cov(np.exp(one_year_weekly_return) - 1, rowvar=False, bias=False) * 51,
                       index=one_year_weekly_return.columns, columns=one_year_weekly_return.columns)
    E_2 = R_fwd / fall
    out.update(fall=fall, E_2=E_2, cov=cov)

    # --- cell b876261e ---
    ef = EfficientFrontier(E_2 / 100, cov, weight_bounds=(0, w1))
    raw_weights = ef.max_sharpe(0.00)
    cleaned_weights = ef.clean_weights()

    # --- cell 4e90df30 ---
    weight = pd.DataFrame(cleaned_weights.values(), index=list(cleaned_weights), columns=["weight"])
    weight = weight[weight.weight >= 0.030]
    weight.weight = weight.weight / sum(weight.weight)
    out["weights"] = weight.weight

    # --- cells 11a8c553, 3cac375a --- ADAPTED (1): the notebook downloads from start_swan_date
    # (2020-02-18 for Swan 9, the day before the peak) and buys at that day's close.
    if start_swan_date is None:
        start_swan_date = peak
    OEX = oex_all.loc[start_swan_date:bottom + datetime.timedelta(days=5)].to_frame()
    stock_price = px_all.loc[start_swan_date:bottom + datetime.timedelta(days=5), list(weight.index)]
    stock_price = stock_price[weight.index]

    # --- cell 43ba6091 ---
    injected_result = pd.DataFrame(index=stock_price.index)
    S1 = np.array(10000 * weight.T) / np.array(stock_price.iloc[0, :]).T
    injected_result["portfolio_value"] = stock_price @ S1.T
    S2 = 10000 / float(OEX.iloc[0, 0])
    injected_result["oex"] = S2 * OEX

    # --- cell b45e8a54 --- ADAPTED (4): event dates instead of "2020-02-19", "2020-03-23"
    temp = injected_result.loc[[peak, bottom]]
    Portfolio_DD = abs(temp["portfolio_value"].iloc[1] / temp["portfolio_value"].iloc[0] - 1)  # ADAPTED (5)
    DD = abs(temp["oex"].iloc[1] / temp["oex"].iloc[0] - 1)  # ADAPTED (5)
    PR = 1 - (Portfolio_DD / DD)
    out.update(PR=PR, Portfolio_DD=Portfolio_DD, DD=DD)
    return out
