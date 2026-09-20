import re
from typing import Optional
import numpy as np
import pandas as pd


BOT_UA_PATTERN = (
    r"headlesschrome|scrapy|python-|urllib|curl/|wget|go-http|node-fetch"
)

def build_features(
    events: pd.DataFrame,
    cookies: pd.DataFrame,
    *,
    with_target: Optional[bool] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Собирает таблицу финальных признаков

    Параметры
    ---------
    events : pd.DataFrame
        Сырые события. Обязательные колонки:
        cookie_id, event_ts, event_name, item_id, item_category,
        item_location, platform, seller_type, user_agent,
        search_query, search_page, pointer_x, pointer_y.
    cookies : pd.DataFrame
        Куки с окном наблюдения. Обязательные колонки:
        cookie_id, cookie_created_at, window_start_ts, window_end_ts.
        Опционально: target (тогда with_target=True по умолчанию).
    with_target : bool, optional
        Добавлять ли колонку target в результат.
        По умолчанию - True, если target есть в cookies, иначе False.
    verbose : bool
        Печатать ли размеры промежуточных таблиц.

    Возвращает
    ----------
    pd.DataFrame
        Таблица признаков. Колонка cookie_id + набор признаков
        (+ target, если with_target).
    """
    if with_target is None:
        with_target = "target" in cookies.columns

    events = _prepare_events(events)
    events_window = _filter_events_by_window(events, cookies)

    if verbose:
        print(f"Событий в окне: {len(events_window)}")
        
    final_feats = _normalized_count_features(events_window)

    final_feats = final_feats.merge(_user_agent_features(events_window), on="cookie_id", how="left")
    final_feats = final_feats.merge(_platform_features(events_window), on="cookie_id", how="left")
    final_feats = final_feats.merge(_seller_features(events_window), on="cookie_id", how="left")
    final_feats = final_feats.merge(_interval_features(events_window), on="cookie_id", how="left")
    final_feats = final_feats.merge(_age_features(cookies), on="cookie_id", how="left")
    final_feats = final_feats.merge(_search_features(events_window), on="cookie_id", how="left")
    final_feats = final_feats.merge(_event_type_features(events_window), on="cookie_id", how="left")
    final_feats = final_feats.merge(_ngrams_features(events_window), on="cookie_id", how="left")
    final_feats = final_feats.merge(_pointer_features(events_window), on="cookie_id", how="left")

    if with_target and "target" in cookies.columns:
        final_feats = final_feats.merge(
            cookies[["cookie_id", "target"]], on="cookie_id", how="left"
        )

    if verbose:
        n_features = len(final_feats.columns) - 1 - int(with_target)
        print(f"Итого признаков: {n_features}")

    return final_feats

def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Приводит типы и создаёт служебные колонки"""
    events = events.copy()
    events["event_ts"] = pd.to_datetime(events["event_ts"])
    events["platform_clean"] = (
        events["platform"].str.lower().replace({"desktop": "web", "iphone": "ios"})
    )
    events["seller_type_clean"] = events["seller_type"].fillna("unknown")
    events["has_pointer"] = (
        events["pointer_x"].notna() & events["pointer_y"].notna()
    )
    return events


def _filter_events_by_window(events: pd.DataFrame, cookies: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только события, попавшие в окно наблюдения куки"""
    cols = ["cookie_id", "window_start_ts", "window_end_ts"]
    merged = events.merge(cookies[cols], on="cookie_id", how="inner")

    mask = (
        (merged["event_ts"] >= merged["window_start_ts"])
        & (merged["event_ts"] <= merged["window_end_ts"])
    )
    return merged.loc[mask].drop(columns=["window_start_ts", "window_end_ts"])

def _user_agent_features(events: pd.DataFrame) -> pd.DataFrame:
    ua = events[["cookie_id", "user_agent"]].copy()
    ua["ua"] = ua["user_agent"].fillna("").str.lower()

    ua["is_bot_ua"] = ua["ua"].str.contains(BOT_UA_PATTERN, regex=True)
    ua["is_mobile_app"] = ua["ua"].str.contains("okhttp", regex=False)

    g = ua.groupby("cookie_id")
    out = pd.DataFrame(index=g.size().index)

    out["ratio_is_bot_ua"] = g["is_bot_ua"].mean()
    out["is_pure_bot_ua"] = (out["ratio_is_bot_ua"] == 1.0).astype(int)
    out["ratio_is_mobile_app"] = g["is_mobile_app"].mean()
    out["ua_length_median"] = g.apply(lambda x: x["ua"].str.len().median())

    return out.reset_index()

def _platform_features(events: pd.DataFrame) -> pd.DataFrame:
    feats = (
        events.groupby("cookie_id")["platform_clean"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    feats = feats.rename(columns={
        "web": "platform_web_ratio",
        "android": "platform_android_ratio",
        "ios": "platform_ios_ratio",
    })
    for col in ["platform_web_ratio", "platform_android_ratio", "platform_ios_ratio"]:
        if col not in feats.columns:
            feats[col] = 0.0
    return feats[["cookie_id", "platform_web_ratio",
                  "platform_android_ratio", "platform_ios_ratio"]]

def _seller_features(events: pd.DataFrame) -> pd.DataFrame:
    feats = (
        events.groupby("cookie_id")["seller_type_clean"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    feats = feats.rename(columns={
        "private": "seller_private_ratio",
        "pro": "seller_pro_ratio",
        "unknown": "seller_unknown_ratio",
    })
    for col in ["seller_private_ratio", "seller_pro_ratio", "seller_unknown_ratio"]:
        if col not in feats.columns:
            feats[col] = 0.0
    return feats[["cookie_id", "seller_private_ratio",
                  "seller_pro_ratio", "seller_unknown_ratio"]]

def _interval_features(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["cookie_id", "event_ts"]).copy()

    ev["delta"] = (
        ev.groupby("cookie_id")["event_ts"]
          .diff()
          .dt.total_seconds()
    )

    g = ev.groupby("cookie_id")
    out = pd.DataFrame(index=g.size().index)

    out["median_interval"] = g["delta"].median()
    out["mean_interval"]   = g["delta"].mean()
    out["std_interval"]    = g["delta"].std()
    out["min_interval"]    = g["delta"].min()
    out["max_interval"]    = g["delta"].max()
    out["p10_interval"]    = g["delta"].quantile(0.10)
    out["p90_interval"]    = g["delta"].quantile(0.90)

    def iqr(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 2:
            return np.nan
        return x.quantile(0.75) - x.quantile(0.25)

    out["iqr_interval"] = g["delta"].apply(iqr)

    out["iqr_over_median"] = out["iqr_interval"] / out["median_interval"].replace(0, np.nan)
    out["std_over_median"] = out["std_interval"] / out["median_interval"].replace(0, np.nan)
    out["p90_over_p10"]    = out["p90_interval"] / out["p10_interval"].replace(0, np.nan)
    out["cv_interval"]     = out["std_interval"] / out["mean_interval"].replace(0, np.nan)

    def ratio_gt(x: pd.Series, threshold: float) -> float:
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        return (x > threshold).mean()

    out["ratio_gt_60s"]  = g["delta"].apply(lambda x: ratio_gt(x, 60))
    out["ratio_gt_300s"] = g["delta"].apply(lambda x: ratio_gt(x, 300))

    return out.reset_index()

def _normalized_count_features(events: pd.DataFrame) -> pd.DataFrame:
    g = events.groupby("cookie_id")
    out = pd.DataFrame(index=g.size().index)

    out["event_count"] = g["event_name"].size()
    out["unique_items"] = g["item_id"].nunique()
    out["unique_categories"] = g["item_category"].nunique()
    out["unique_locations"] = g["item_location"].nunique()
    out["unique_event_types"] = g["event_name"].nunique()

    q = events[events["search_query"].fillna("").str.strip() != ""]
    out["unique_search_queries"] = (
        q.groupby("cookie_id")["search_query"].nunique()
        .reindex(out.index, fill_value=0)
    )

    n = g.size().clip(lower=1)

    out["items_per_event"]       = out["unique_items"]          / n
    out["categories_per_event"]  = out["unique_categories"]     / n
    out["locations_per_event"]   = out["unique_locations"]      / n
    out["queries_per_event"]     = out["unique_search_queries"] / n
    out["event_types_per_event"] = out["unique_event_types"]    / n

    return out.reset_index()

def _age_features(cookies: pd.DataFrame) -> pd.DataFrame:
    age_days = ((cookies["window_start_ts"] - cookies["cookie_created_at"]).dt.total_seconds() / 86400)

    return pd.DataFrame({
        "cookie_id": cookies["cookie_id"].values,
        "cookie_age_log": np.log1p(age_days.clip(lower=0)).values,
    })

def _search_features(events: pd.DataFrame) -> pd.DataFrame:
    search = events[events["search_query"].fillna("").str.strip() != ""]

    feats = (
        events.groupby("cookie_id")
        .size()
        .reset_index(name="total_events")
        .merge(
            search.groupby("cookie_id").agg(
                search_events_count=("search_query", "size"),
                unique_search_queries=("search_query", "nunique"),
                max_search_page=("search_page", "max"),
                mean_search_page=("search_page", "mean"),
            ).reset_index(),
            on="cookie_id",
            how="left",
        )
    )

    fill_cols = [
        "search_events_count", "unique_search_queries",
        "max_search_page", "mean_search_page",
    ]
    feats[fill_cols] = feats[fill_cols].fillna(0)
    feats["search_query_ratio"] = feats["search_events_count"] / feats["total_events"]

    return feats[[
        "cookie_id", "search_query_ratio", "unique_search_queries",
        "max_search_page", "mean_search_page",
    ]]

def _event_type_features(events: pd.DataFrame) -> pd.DataFrame:
    ratios = (
        events.groupby("cookie_id")["event_name"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .add_prefix("ratio_")
        .reset_index()
    )

    expected_events = [
        "item_view", "photo_swipe", "favorite_add", "login",
        "search_results_view", "contact_phone_show", "contact_chat_open",
        "contact_message_sent", "seller_page_view", "captcha_shown",
    ]
    for e in expected_events:
        col = f"ratio_{e}"
        if col not in ratios.columns:
            ratios[col] = 0.0

    flags = (
        events.assign(
            has_favorite_add=events["event_name"] == "favorite_add",
            has_login=events["event_name"] == "login",
        )
        .groupby("cookie_id")[["has_favorite_add", "has_login"]]
        .any()
        .astype(int)
        .reset_index()
    )

    item_view   = events[events["event_name"] == "item_view"].groupby("cookie_id").size()
    photo_swipe = events[events["event_name"] == "photo_swipe"].groupby("cookie_id").size()
    ratio = (
        (item_view / (photo_swipe + 1))
        .fillna(0)
        .reset_index(name="item_view_to_photo_ratio")
    )

    feats = ratios.merge(flags, on="cookie_id", how="left")
    feats = feats.merge(ratio, on="cookie_id", how="left")
    return feats

def _ngrams_features(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["cookie_id", "event_ts"])

    rows = []
    for cookie_id, sub in ev.groupby("cookie_id", sort=False):
        seq = sub["event_name"].tolist()
        n_events = len(seq)
        n_big = n_events - 1
        n_tri = n_events - 2

        row = {"cookie_id": cookie_id}

        # Короткая сессия: переходов нет, оставляем NaN
        if n_events < 3:
            row.update({
                "bigrams_per_transition": np.nan,
                "unique_trigrams": 0,
                "trigrams_per_transition": np.nan,
            })
            rows.append(row)
            continue

        bigrams  = list(zip(seq[:-1], seq[1:]))
        trigrams = list(zip(seq[:-2], seq[1:-1], seq[2:]))

        n_unique_bigrams  = len(set(bigrams))
        n_unique_trigrams = len(set(trigrams))

        row["bigrams_per_transition"]  = n_unique_bigrams / n_big
        row["unique_trigrams"]         = n_unique_trigrams
        row["trigrams_per_transition"] = n_unique_trigrams / n_tri

        rows.append(row)

    return pd.DataFrame(rows)

def _pointer_features(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["cookie_id", "event_ts"])

    rows = []
    for cookie_id, sub in ev.groupby("cookie_id", sort=False):
        mask = sub["pointer_x"].notna() & sub["pointer_y"].notna()

        row = {"cookie_id": cookie_id}

        if mask.sum() < 2:
            row.update({
                "ptr_std_x": np.nan,
                "ptr_std_y": np.nan,
                "ptr_range_x": np.nan,
                "ptr_range_y": np.nan,
                "ptr_mean_step": np.nan,
            })
            rows.append(row)
            continue

        p = sub.loc[mask, ["pointer_x", "pointer_y"]]
        x = p["pointer_x"].to_numpy()
        y = p["pointer_y"].to_numpy()

        # Разброс координат
        row["ptr_std_x"]   = float(np.std(x))
        row["ptr_std_y"]   = float(np.std(y))
        row["ptr_range_x"] = float(x.max() - x.min())
        row["ptr_range_y"] = float(y.max() - y.min())

        # Средний шаг между последовательными точками
        steps = np.hypot(np.diff(x), np.diff(y))
        row["ptr_mean_step"] = float(steps.mean())

        rows.append(row)

    return pd.DataFrame(rows)