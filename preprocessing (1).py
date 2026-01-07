#!/usr/bin/env python3
"""
generate_diverse_40k.py

Usage:
    python generate_diverse_40k.py \
        --input dataset/train.csv \
        --out_selected dataset/train_selected_40000.csv \
        --out_rest dataset/train_rest.csv \
        --target_n 40000 \
        --seed 42

This script creates a robust diverse subset for fine-tuning:
 - fills product_description with item_name where missing
 - computes auxiliary features (unit_cat, price quantiles, bullet counts, ipq presence, image availability)
 - stratified proportional sampling across unit_cat × price_q
 - force-includes rare/missing-case examples and price extremes
 - writes selected and remaining CSVs
"""

import os
import ast
import argparse
import math
import numpy as np
import pandas as pd

# ---------------------------- helpers ----------------------------
def safe_literal_eval(s):
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return s
    s_strip = str(s).strip()
    if s_strip == "[]" or s_strip == "":
        return []
    try:
        v = ast.literal_eval(s_strip)
        if isinstance(v, list):
            return v
        return []
    except Exception:
        # fallback: split by comma but keep short list
        if ',' in s_strip:
            parts = [p.strip() for p in s_strip.split(',') if p.strip()]
            return parts
        return []

def fill_description_with_itemname(df, desc_col='product_description', item_col='item_name'):
    missing_mask = df[desc_col].isna() | (df[desc_col].astype(str).str.strip().str.lower() == "none")
    df.loc[missing_mask, desc_col] = df.loc[missing_mask, item_col].astype(str)
    return df

def compute_features(df,
                     bullet_col='bullet_points_list',
                     image_col='image_link',
                     ipq_col='ipq',
                     value_col='value',
                     price_col='price',
                     item_col='item_name',
                     desc_col='product_description'):
    # unit category
    if 'unit' in df.columns:
        df['unit_cat'] = df['unit'].fillna("__missing__").astype(str).str.lower().str.strip()
        df.loc[df['unit_cat'] == "", 'unit_cat'] = "__missing__"
    else:
        df['unit_cat'] = "__missing__"

    # ipq present
    df['ipq_present'] = ~ (df[ipq_col].isna() | (df[ipq_col].astype(str).str.lower() == "none"))

    # bullet list and count
    df['bullet_list'] = df.get(bullet_col, pd.Series([[]] * len(df))).apply(safe_literal_eval)
    df['bullet_count'] = df['bullet_list'].apply(len)
    def bullet_cat(n):
        if n == 0:
            return "0"
        if n == 1:
            return "1"
        if 2 <= n <= 4:
            return "2-4"
        if 5 <= n <= 9:
            return "5-9"
        return "10+"
    df['bullet_cat'] = df['bullet_count'].apply(bullet_cat)

    # image availability
    df['image_avail'] = ~ (df[image_col].isna() | (df[image_col].astype(str).str.strip().str.lower() == "none") | (df[image_col].astype(str).str.strip() == ""))
    df['image_avail'] = df['image_avail'].astype(bool)

    # description missing (after fill)
    df['description_missing'] = df[desc_col].isna() | (df[desc_col].astype(str).str.strip().str.lower() == "none")
    df['description_missing'] = df['description_missing'].astype(bool)

    # item_name length
    df['item_name_len_words'] = df[item_col].astype(str).apply(lambda x: len(x.split()))

    # price numeric
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df['price_isnull'] = df[price_col].isna()

    # price quantile bins (deciles) for rows with positive price
    valid = df[price_col] > 0
    if valid.sum() > 0:
        try:
            df.loc[valid, 'price_q'] = pd.qcut(df.loc[valid, price_col], q=10, labels=[f"q{i}" for i in range(10)])
            # convert to object so assignments are flexible
            df['price_q'] = df['price_q'].astype(object)
        except Exception:
            df.loc[valid, 'price_q'] = pd.cut(df.loc[valid, price_col], bins=10, labels=[f"q{i}" for i in range(10)])
            df['price_q'] = df['price_q'].astype(object)
        df.loc[~valid, 'price_q'] = "missing_price"
    else:
        df['price_q'] = "missing_price"

    # value q (if numeric)
    df[value_col] = pd.to_numeric(df.get(value_col, pd.Series([np.nan]*len(df))), errors='coerce')
    valid_v = df[value_col].notna()
    if valid_v.sum() > 0:
        bins = min(10, valid_v.sum())
        if bins >= 2:
            df.loc[valid_v, 'value_q'] = pd.qcut(df.loc[valid_v, value_col], q=bins, labels=[f"vq{i}" for i in range(bins)])
        else:
            df.loc[valid_v, 'value_q'] = "vq0"
    else:
        df['value_q'] = "missing_value"

    return df

def stratified_sample(df, target_n, group_cols, min_per_group=1, random_state=42):
    groups = df.groupby(group_cols)
    sizes = groups.size().reset_index(name='gsize')
    total = len(df)
    sizes['desired'] = (sizes['gsize'] / total * target_n).round().astype(int)

    # ensure at least min_per_group for present groups
    sizes.loc[(sizes['gsize'] > 0) & (sizes['desired'] < min_per_group), 'desired'] = min_per_group

    # adjust to match target_n
    diff = target_n - sizes['desired'].sum()
    if diff > 0:
        sizes = sizes.sort_values('gsize', ascending=False).reset_index(drop=True)
        idx = 0
        while diff > 0:
            sizes.at[idx % len(sizes), 'desired'] += 1
            idx += 1
            diff -= 1
    elif diff < 0:
        sizes = sizes.sort_values(['desired', 'gsize'], ascending=[False, True]).reset_index(drop=True)
        idx = 0
        while diff < 0:
            if sizes.at[idx, 'desired'] > min_per_group:
                sizes.at[idx, 'desired'] -= 1
                diff += 1
            idx = (idx + 1) % len(sizes)

    rng = np.random.default_rng(random_state)
    selection_indices = []
    sizes_lookup = sizes.set_index(group_cols)['desired'].to_dict()

    for grp_vals, group_df in groups:
        desired = sizes_lookup.get(grp_vals, 0)
        if desired <= 0:
            continue
        if len(group_df) <= desired:
            selection_indices.extend(group_df.index.tolist())
        else:
            sampled = group_df.sample(n=desired, random_state=random_state).index.tolist()
            selection_indices.extend(sampled)

    # fix length
    if len(selection_indices) > target_n:
        selection_indices = rng.choice(selection_indices, size=target_n, replace=False).tolist()
    if len(selection_indices) < target_n:
        remaining = list(set(df.index.tolist()) - set(selection_indices))
        add_needed = target_n - len(selection_indices)
        if add_needed > 0 and len(remaining) > 0:
            extra = rng.choice(remaining, size=add_needed, replace=False).tolist()
            selection_indices.extend(extra)

    assert len(selection_indices) == target_n
    return sorted(selection_indices)

def force_include_cases(df_all, selected_df, target_n, k_per_case=5, random_state=42, rare_unit_threshold=50):
    rng = np.random.default_rng(random_state)
    selected_idx = set(selected_df.index.tolist())
    to_add_idx = []

    # Cases to guarantee
    cases = [
        ("bullet_cat", "0"),
        ("bullet_cat", "1"),
        ("ipq_present", False),
        ("description_missing", True),
        ("image_avail", False)
    ]

    # rare units
    unit_counts = df_all['unit_cat'].value_counts()
    rare_units = unit_counts[unit_counts <= rare_unit_threshold].index.tolist()
    for ru in rare_units:
        cases.append(("unit_cat", ru))

    for col, val in cases:
        if col not in df_all.columns:
            continue
        if isinstance(val, (list, tuple, set)):
            candidates = df_all[df_all[col].isin(val)].index.tolist()
        else:
            candidates = df_all[df_all[col] == val].index.tolist()
        # filter out already selected or already planned to add
        candidates = [i for i in candidates if i not in selected_idx and i not in to_add_idx]
        if len(candidates) == 0:
            continue
        take_n = min(k_per_case, len(candidates))
        chosen = list(rng.choice(candidates, size=take_n, replace=False))
        to_add_idx.extend(chosen)

    # Add extremes (top & bottom prices)
    n_extreme = min(100, max(1, int(0.002 * len(df_all))))  # up to 0.2% or 100
    # ensure unique indices not already selected or planned
    topk = [i for i in df_all.nlargest(n_extreme, 'price').index.tolist() if i not in selected_idx and i not in to_add_idx]
    botk = [i for i in df_all.nsmallest(n_extreme, 'price').index.tolist() if i not in selected_idx and i not in to_add_idx]
    to_add_idx.extend(topk)
    to_add_idx.extend(botk)

    if len(to_add_idx) == 0:
        return selected_df.copy()

    new_selected_idx = list(selected_idx) + to_add_idx

    # trim if over target: drop random from previously selected (not forced)
    if len(new_selected_idx) > target_n:
        prev_only = [i for i in selected_idx if i not in to_add_idx]
        n_drop = len(new_selected_idx) - target_n
        if len(prev_only) <= n_drop:
            drop_candidates = new_selected_idx
        else:
            drop_candidates = prev_only
        drop_idx = list(rng.choice(drop_candidates, size=n_drop, replace=False))
        new_selected_idx = [i for i in new_selected_idx if i not in drop_idx]

    # if still less (unlikely), fill with random remaining
    if len(new_selected_idx) < target_n:
        remaining = list(set(df_all.index.tolist()) - set(new_selected_idx))
        add_needed = target_n - len(new_selected_idx)
        if add_needed > 0 and len(remaining) > 0:
            extra = list(rng.choice(remaining, size=add_needed, replace=False))
            new_selected_idx.extend(extra)

    assert len(new_selected_idx) == target_n, f"final len {len(new_selected_idx)} != {target_n}"
    new_selected_df = df_all.loc[sorted(new_selected_idx)].copy()
    return new_selected_df

# ---------------------------- main ----------------------------
def main(args):
    print("Loading data from", args.input)
    df = pd.read_csv(args.input)
    print("Total rows:", len(df))

    # Fill descriptions
    print("Filling missing product_description with item_name where needed...")
    df = fill_description_with_itemname(df, desc_col='product_description', item_col='item_name')

    # Compute features
    print("Computing features...")
    df = compute_features(df)

    # Restrict to rows with valid positive price for supervised training
    df_trainable = df[df['price'].notna() & (df['price'] > 0)].copy()
    excluded = len(df) - len(df_trainable)
    print(f"Excluded {excluded} rows with missing or nonpositive price from sampling pool.")

    # Choose stratification groups
    group_cols = ['unit_cat', 'price_q']
    print("Performing stratified sampling on groups:", group_cols)

    target = min(args.target_n, len(df_trainable))
    selected_idx = stratified_sample(df_trainable, target, group_cols, min_per_group=args.min_per_group, random_state=args.seed)
    selected_df = df_trainable.loc[selected_idx].copy()

    # Force include missing/rare cases and extremes
    print(f"Forcing inclusion of special cases (k_per_case={args.k_per_case}) and extremes...")
    selected_df = force_include_cases(df_trainable, selected_df, target, k_per_case=args.k_per_case, random_state=args.seed, rare_unit_threshold=args.rare_unit_threshold)

    # Prepare rest (drop selected indices from full df)
    rest_df = df.drop(index=selected_df.index).copy()

    # Diagnostics
    print("\n--- Diagnostics for selected subset ---")
    print("Size:", len(selected_df))
    print("\nunit_cat counts (top 15):")
    print(selected_df['unit_cat'].value_counts().head(15))
    print("\nprice_q distribution:")
    print(selected_df['price_q'].value_counts().sort_index())
    print("\nbullet_cat distribution:")
    print(selected_df['bullet_cat'].value_counts())
    print("\nipq_present fraction:")
    print(selected_df['ipq_present'].value_counts(normalize=True))
    print("\nimage_avail fraction:")
    print(selected_df['image_avail'].value_counts(normalize=True))
    print("\nnumber of examples where description was replaced by item_name (post-fill):", ((df['product_description'] == df['item_name']).sum()))

    # Save
    os.makedirs(os.path.dirname(args.out_selected) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_rest) or ".", exist_ok=True)
    selected_df.to_csv(args.out_selected, index=False)
    rest_df.to_csv(args.out_rest, index=False)
    print(f"\nSaved selected subset to: {args.out_selected}")
    print(f"Saved remaining rows to: {args.out_rest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="file3.csv", help="Path to input CSV")
    parser.add_argument("--out_selected", type=str, default="train_on_40000.csv", help="Output CSV for selected subset")
    parser.add_argument("--out_rest", type=str, default="C:/Users/murtu/OneDrive/Documents/Amazon ML Challenge Preprocessing/train_rest.csv", help="Output CSV for remaining rows")
    parser.add_argument("--target_n", type=int, default=40000, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_per_group", type=int, default=1, help="Minimum per stratified group")
    parser.add_argument("--k_per_case", type=int, default=10, help="Number to force include per special case (no bullets, missing ipq, etc.)")
    parser.add_argument("--rare_unit_threshold", type=int, default=50, help="Units with frequency <= threshold considered rare and forced")
    args = parser.parse_args()
    main(args)
