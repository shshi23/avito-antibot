import re
from typing import Optional

import numpy as np
import pandas as pd


BOT_PATTERNS = re.compile(
    r"bot|crawler|spider|python|curl|wget|scrapy|selenium"
)

def build_features(
    events: pd.DataFrame,
    cookies: pd.DataFrame,
    *,
    with_target: Optional[bool] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Собирает таблицу признаков: одна строка = одна cookie_id.

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
        По умолчанию — True, если target есть в cookies, иначе False.
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

    base = _basic_counts(events_window)

    base = base.merge(_user_agent_features(events_window), on="cookie_id", how="left")
    base = base.merge(_platform_features(events_window), on="cookie_id", how="left")
    base = base.merge(_seller_features(events_window), on="cookie_id", how="left")
    base = base.merge(_time_features(events_window), on="cookie_id", how="left")
    base = base.merge(_age_features(cookies), on="cookie_id", how="left")
    base = base.merge(_search_features(events_window), on="cookie_id", how="left")
    base = base.merge(_pointer_features(events_window), on="cookie_id", how="left")
    base = base.merge(_event_type_features(events_window), on="cookie_id", how="left")

    # target подтягиваем в самом конце
    if with_target and "target" in cookies.columns:
        base = base.merge(
            cookies[["cookie_id", "target"]], on="cookie_id", how="left"
        )

    if verbose:
        n_features = len(base.columns) - 1 - int(with_target)
        print(f"Итого признаков: {n_features}")

    return base

def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Приводит типы и создаёт служебные колонки один раз."""
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


def _filter_events_by_window(
    events: pd.DataFrame, cookies: pd.DataFrame
) -> pd.DataFrame:
    """Оставляет только события, попавшие в окно наблюдения куки."""
    cols = ["cookie_id", "window_start_ts", "window_end_ts"]
    merged = events.merge(cookies[cols], on="cookie_id", how="inner")

    mask = (
        (merged["event_ts"] >= merged["window_start_ts"])
        & (merged["event_ts"] <= merged["window_end_ts"])
    )
    return merged.loc[mask].drop(columns=["window_start_ts", "window_end_ts"])

def _basic_counts(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby("cookie_id")
        .agg(
            event_count=("event_name", "size"),
            unique_items=("item_id", "nunique"),
            unique_categories=("item_category", "nunique"),
            unique_locations=("item_location", "nunique"),
            unique_event_types=("event_name", "nunique"),
        )
        .reset_index()
    )

def _user_agent_features(events: pd.DataFrame) -> pd.DataFrame:
    ua = events[["cookie_id", "user_agent"]].copy()
    ua["ua_clean"] = ua["user_agent"].fillna("").str.lower()
    ua["ua_len"] = ua["ua_clean"].str.len()
    ua["is_headless"] = ua["ua_clean"].str.contains("headless", regex=False)
    ua["is_bot"] = ua["ua_clean"].str.contains(BOT_PATTERNS)

    feats = ua.groupby("cookie_id").agg(
        ua_length=("ua_len", "median"),
        has_headless=("is_headless", "any"),
        has_bot_pattern=("is_bot", "any"),
        n_unique_ua=("user_agent", "nunique"),
    ).reset_index()

    feats[["has_headless", "has_bot_pattern"]] = (
        feats[["has_headless", "has_bot_pattern"]].astype(int)
    )
    return feats


def _platform_features(events: pd.DataFrame) -> pd.DataFrame:
    feats = (
        events.groupby("cookie_id")["platform_clean"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    feats = feats.rename(
        columns={
            "web": "platform_web_ratio",
            "android": "platform_android_ratio",
            "ios": "platform_ios_ratio",
        }
    )
    for col in ["platform_web_ratio", "platform_android_ratio", "platform_ios_ratio"]:
        if col not in feats.columns:
            feats[col] = 0.0
    return feats[
        ["cookie_id", "platform_web_ratio", "platform_android_ratio", "platform_ios_ratio"]
    ]


def _seller_features(events: pd.DataFrame) -> pd.DataFrame:
    feats = (
        events.groupby("cookie_id")["seller_type_clean"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    feats = feats.rename(
        columns={
            "private": "seller_private_ratio",
            "pro": "seller_pro_ratio",
            "unknown": "seller_unknown_ratio",
        }
    )
    for col in ["seller_private_ratio", "seller_pro_ratio", "seller_unknown_ratio"]:
        if col not in feats.columns:
            feats[col] = 0.0
    return feats[
        ["cookie_id", "seller_private_ratio", "seller_pro_ratio", "seller_unknown_ratio"]
    ]


def _time_features(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.sort_values(["cookie_id", "event_ts"]).copy()
    ev["hour"] = ev["event_ts"].dt.hour
    ev["time_diff_sec"] = (
        ev.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
    )

    return ev.groupby("cookie_id").agg(
        night_activity_ratio=("hour", lambda x: ((x >= 0) & (x < 6)).mean()),
        day_activity_ratio=("hour", lambda x: ((x >= 6) & (x < 24)).mean()),
        hour_std=("hour", "std"),
        median_interval=("time_diff_sec", "median"),
        mean_interval=("time_diff_sec", "mean"),
        min_interval=("time_diff_sec", "min"),
        ratio_instant_01=("time_diff_sec", lambda x: (x < 0.1).mean()),
    ).reset_index()


def _age_features(cookies: pd.DataFrame) -> pd.DataFrame:
    age_days = (
        (cookies["window_start_ts"] - cookies["cookie_created_at"]).dt.total_seconds()
        / 86400
    )
    return pd.DataFrame(
        {
            "cookie_id": cookies["cookie_id"].values,
            "cookie_age_log": np.log1p(age_days.clip(lower=0)).values,
            "is_created_in_window": (
                cookies["cookie_created_at"] >= cookies["window_start_ts"]
            ).astype(int).values,
        }
    )


def _search_features(events: pd.DataFrame) -> pd.DataFrame:
    search = events[events["search_query"].notna()]

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
        "search_events_count",
        "unique_search_queries",
        "max_search_page",
        "mean_search_page",
    ]
    feats[fill_cols] = feats[fill_cols].fillna(0)
    feats["search_query_ratio"] = (
        feats["search_events_count"] / feats["total_events"]
    )

    return feats[
        [
            "cookie_id",
            "search_query_ratio",
            "unique_search_queries",
            "max_search_page",
            "mean_search_page",
        ]
    ]


def _pointer_features(events: pd.DataFrame) -> pd.DataFrame:
    feats = events.groupby("cookie_id").agg(
        pointer_ratio=("has_pointer", "mean"),
        has_any_pointer=("has_pointer", "any"),
        pointer_x_range=("pointer_x", np.ptp),
        pointer_y_range=("pointer_y", np.ptp),
    ).reset_index()

    feats[["pointer_ratio", "pointer_x_range", "pointer_y_range"]] = (
        feats[["pointer_ratio", "pointer_x_range", "pointer_y_range"]].fillna(0.0)
    )
    feats["has_any_pointer"] = feats["has_any_pointer"].astype(int)
    return feats


def _event_type_features(events: pd.DataFrame) -> pd.DataFrame:
    ratios = (
        events.groupby("cookie_id")["event_name"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .add_prefix("ratio_")
        .reset_index()
    )

    # гарантируем, что все ожидаемые колонки есть (даже если событие не встретилось)
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
            has_captcha=events["event_name"] == "captcha_shown",
        )
        .groupby("cookie_id")[["has_favorite_add", "has_login", "has_captcha"]]
        .any()
        .astype(int)
        .reset_index()
    )

    item_view = (
        events[events["event_name"] == "item_view"].groupby("cookie_id").size()
    )
    photo_swipe = (
        events[events["event_name"] == "photo_swipe"].groupby("cookie_id").size()
    )
    ratio = (
        (item_view / (photo_swipe + 1))
        .fillna(0)
        .reset_index(name="item_view_to_photo_ratio")
    )

    feats = ratios.merge(flags, on="cookie_id", how="left")
    feats = feats.merge(ratio, on="cookie_id", how="left")
    return feats