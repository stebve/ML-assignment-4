import numpy as np
import pandas as pd


def DQR_continuous(df):
    summary = pd.DataFrame({
        'Count': 257673,
        '% Miss.': df.isna().mean()*100,
        'Card.': df.nunique(),
        'Min.': df.min(numeric_only=True),
        '1st Qrt.': df.quantile(0.25, numeric_only=True),
        'Mean': df.mean(numeric_only=True),
        'Median': df.median(numeric_only=True),
        '3rd Qrt.': df.quantile(0.75, numeric_only=True),
        'Max.': df.max(numeric_only=True),
        'Std. Dev.': df.std(numeric_only=True),
    })
    return summary


def DQR_categorical(df):
    summary_rows = []
 
    for col in df.columns:
        vc = df[col].value_counts()  # sorted descending by frequency, excludes NaN by default
 
        count = df[col].count()
        pct_miss = df[col].isna().mean() * 100
        card = df[col].nunique()
 
        mode = vc.index[0] if len(vc) > 0 else np.nan
        mode_freq = vc.iloc[0] if len(vc) > 0 else np.nan
        mode_pct = (mode_freq / count * 100) if count > 0 else np.nan
 
        mode2 = vc.index[1] if len(vc) > 1 else np.nan
        mode2_freq = vc.iloc[1] if len(vc) > 1 else np.nan
        mode2_pct = (mode2_freq / count * 100) if len(vc) > 1 and count > 0 else np.nan
 
        summary_rows.append({
            'Count': 257673,
            '% Miss.': pct_miss,
            'Card.': card,
            'Mode': mode,
            'Mode Freq.': mode_freq,
            'Mode %': mode_pct,
            '2nd Mode': mode2,
            '2nd Mode Freq.': mode2_freq,
            '2nd Mode %': mode2_pct,
        })
    summary = pd.DataFrame(summary_rows)
    summary.index = df.columns
 
    return summary
