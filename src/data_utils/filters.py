"""
filters.py
-------------
Reusable metadata filter helpers.

Responsibilities:
- Build reusable filter expressions
"""

import polars as pl


def by_gender(g):
    return pl.col("gender_label") == g


def by_age(a):
    return pl.col("age_label") == a
