"""
Common file I/O utilities for PL Analyzer
"""
import numpy as np
import pandas as pd
import streamlit as st
import io


def load_spectrum_file(uploaded_file):
    """
    Load a spectrum file (CSV, TXT, Excel) and return (wavelength, intensity) arrays.
    Tries multiple parsing strategies automatically.
    """
    if uploaded_file is None:
        return None, None

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, header=None)
        else:
            raw = uploaded_file.read().decode('utf-8', errors='replace')
            uploaded_file.seek(0)
            # Try common delimiters
            for sep in [',', '\t', ' ', ';']:
                try:
                    df = pd.read_csv(io.StringIO(raw), sep=sep, header=None, comment='#', engine='python')
                    df = df.dropna(how='all')
                    # Check we got at least 2 numeric columns
                    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().sum() > 3]
                    if len(numeric_cols) >= 2:
                        break
                except Exception:
                    continue

        # Drop fully-NaN rows/cols
        df = df.dropna(how='all').reset_index(drop=True)

        # Find first fully numeric row
        start_row = 0
        for i, row in df.iterrows():
            try:
                [float(v) for v in row.values if str(v).strip() not in ('', 'nan')]
                start_row = i
                break
            except ValueError:
                continue

        df = df.iloc[start_row:].reset_index(drop=True)

        # Take first two numeric columns
        numeric_data = df.apply(pd.to_numeric, errors='coerce')
        valid_cols = numeric_data.columns[numeric_data.notna().sum() > 3].tolist()

        wavelength = numeric_data[valid_cols[0]].dropna().values.astype(float)
        intensity = numeric_data[valid_cols[1]].dropna().values.astype(float)

        # Align lengths
        min_len = min(len(wavelength), len(intensity))
        wavelength = wavelength[:min_len]
        intensity = intensity[:min_len]

        # Sort by wavelength
        idx = np.argsort(wavelength)
        wavelength = wavelength[idx]
        intensity = intensity[idx]

        return wavelength, intensity

    except Exception as e:
        st.error(f"파일 로딩 오류 ({uploaded_file.name}): {e}")
        return None, None


def load_multiple_files(uploaded_files):
    """
    Load multiple spectrum files. Returns list of dicts with keys: name, wavelength, intensity
    """
    results = []
    for f in uploaded_files:
        wl, inten = load_spectrum_file(f)
        if wl is not None:
            results.append({
                'name': f.name,
                'wavelength': wl,
                'intensity': inten
            })
    return results


def interpolate_to_common_grid(spectra_list, wl_min=None, wl_max=None, n_points=1000):
    """
    Interpolate all spectra to a common wavelength grid.
    """
    if not spectra_list:
        return None, []

    # Determine common range
    if wl_min is None:
        wl_min = max(s['wavelength'].min() for s in spectra_list)
    if wl_max is None:
        wl_max = min(s['wavelength'].max() for s in spectra_list)

    common_wl = np.linspace(wl_min, wl_max, n_points)
    interp_intensities = []

    for s in spectra_list:
        inten_interp = np.interp(common_wl, s['wavelength'], s['intensity'])
        interp_intensities.append(inten_interp)

    return common_wl, interp_intensities


def to_excel_download(dfs_dict):
    """
    Given a dict of {sheet_name: dataframe}, return Excel bytes for download.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dfs_dict.items():
            safe_name = sheet_name[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()
