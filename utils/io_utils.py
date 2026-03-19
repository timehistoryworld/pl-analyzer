"""
Common file I/O utilities for PL Analyzer
캐싱 전략:
  - UploadedFile 객체 자체는 캐시 키로 쓸 수 없으므로
    파일 bytes + 파일명을 키로 사용하는 래퍼 패턴을 사용합니다.
  - 같은 파일을 다시 업로드하거나, 다른 탭으로 이동했다가 돌아와도
    재파싱 없이 캐시에서 즉시 반환합니다.
"""
import numpy as np
import pandas as pd
import streamlit as st
import io


# ── 내부 파싱 함수 (bytes 기반, 캐시 가능) ──────────────────────────────────

@st.cache_data(show_spinner=False)
def _parse_spectrum_bytes(file_bytes: bytes, filename: str):
    """
    실제 파싱 로직. 인자가 bytes + str이므로 st.cache_data가 완벽하게 해시 가능.
    같은 파일(내용 동일)은 재실행 없이 캐시에서 반환.
    """
    filename_lower = filename.lower()
    try:
        if filename_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes), header=None)
        else:
            raw = file_bytes.decode('utf-8', errors='replace')
            df = None
            for sep in [',', '\t', ' ', ';']:
                try:
                    candidate = pd.read_csv(io.StringIO(raw), sep=sep,
                                            header=None, comment='#', engine='python')
                    candidate = candidate.dropna(how='all')
                    numeric_cols = [
                        c for c in candidate.columns
                        if pd.to_numeric(candidate[c], errors='coerce').notna().sum() > 3
                    ]
                    if len(numeric_cols) >= 2:
                        df = candidate
                        break
                except Exception:
                    continue
            if df is None:
                return None, None

        df = df.dropna(how='all').reset_index(drop=True)

        # 첫 번째 완전히 숫자인 행 찾기
        start_row = 0
        for i, row in df.iterrows():
            try:
                [float(v) for v in row.values if str(v).strip() not in ('', 'nan')]
                start_row = i
                break
            except ValueError:
                continue

        df = df.iloc[start_row:].reset_index(drop=True)

        numeric_data = df.apply(pd.to_numeric, errors='coerce')
        valid_cols = numeric_data.columns[numeric_data.notna().sum() > 3].tolist()
        if len(valid_cols) < 2:
            return None, None

        wavelength = numeric_data[valid_cols[0]].dropna().values.astype(float)
        intensity  = numeric_data[valid_cols[1]].dropna().values.astype(float)

        min_len   = min(len(wavelength), len(intensity))
        wavelength = wavelength[:min_len]
        intensity  = intensity[:min_len]

        idx        = np.argsort(wavelength)
        wavelength = wavelength[idx]
        intensity  = intensity[idx]

        return wavelength, intensity

    except Exception as e:
        return None, None


# ── 공개 API ─────────────────────────────────────────────────────────────────

def load_spectrum_file(uploaded_file):
    """
    UploadedFile → (wavelength, intensity) 배열 반환.
    내부적으로 bytes를 읽어 캐시 함수에 넘기므로,
    같은 파일은 Streamlit 세션 내내 재파싱하지 않습니다.
    """
    if uploaded_file is None:
        return None, None
    try:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)   # 다른 곳에서 다시 읽을 수 있도록 리셋
        wl, inten = _parse_spectrum_bytes(file_bytes, uploaded_file.name)
        if wl is None:
            st.error(f"파일 파싱 실패: {uploaded_file.name}")
        return wl, inten
    except Exception as e:
        st.error(f"파일 로딩 오류 ({uploaded_file.name}): {e}")
        return None, None


def load_multiple_files(uploaded_files):
    """
    여러 UploadedFile → list of dict(name, wavelength, intensity).
    각 파일 개별적으로 캐시 적용됩니다.
    """
    results = []
    for f in uploaded_files:
        wl, inten = load_spectrum_file(f)
        if wl is not None:
            results.append({
                'name':       f.name,
                'wavelength': wl,
                'intensity':  inten,
            })
    return results


@st.cache_data(show_spinner=False)
def interpolate_to_common_grid(names_key: tuple,
                                wavelengths: tuple,
                                intensities: tuple,
                                wl_min=None, wl_max=None, n_points=1000):
    """
    여러 스펙트럼을 공통 파장 그리드로 보간.
    numpy array는 캐시 키로 바로 쓸 수 없으므로 tuple로 변환해서 넘깁니다.

    사용 예:
        wls   = tuple(s['wavelength'].tobytes() for s in spectra)
        ints  = tuple(s['intensity'].tobytes()  for s in spectra)
        names = tuple(s['name'] for s in spectra)
        common_wl, interp_list = interpolate_to_common_grid(names, wls, ints)
    """
    wl_arrays    = [np.frombuffer(w, dtype=float) for w in wavelengths]
    inten_arrays = [np.frombuffer(i, dtype=float) for i in intensities]

    if wl_min is None:
        wl_min = max(w.min() for w in wl_arrays)
    if wl_max is None:
        wl_max = min(w.max() for w in wl_arrays)

    common_wl      = np.linspace(wl_min, wl_max, n_points)
    interp_list    = [np.interp(common_wl, wl, inten)
                      for wl, inten in zip(wl_arrays, inten_arrays)]
    return common_wl, interp_list


def interpolate_spectra(spectra_list, wl_min=None, wl_max=None, n_points=1000):
    """
    spectra_list(dict 리스트) → (common_wl, interp_intensities).
    내부적으로 캐시 가능한 형태로 변환해서 호출합니다.
    """
    if not spectra_list:
        return None, []
    names = tuple(s['name']                    for s in spectra_list)
    wls   = tuple(s['wavelength'].tobytes()    for s in spectra_list)
    ints  = tuple(s['intensity'].tobytes()     for s in spectra_list)
    return interpolate_to_common_grid(names, wls, ints, wl_min, wl_max, n_points)


@st.cache_data(show_spinner=False)
def to_excel_download(sheets: tuple) -> bytes:
    """
    ((sheet_name, df), ...) 형태의 tuple → Excel bytes.
    tuple로 받아야 캐시 키 해싱이 가능합니다.

    사용 예:
        excel = to_excel_download((
            ("Sheet1", df1),
            ("Sheet2", df2),
        ))
        st.download_button(..., data=excel)
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in sheets:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()


def to_excel_download_dict(dfs_dict: dict) -> bytes:
    """
    기존 dict 인터페이스 유지용 래퍼.
    내부에서 tuple로 변환해 캐시 함수 호출.
    """
    return to_excel_download(tuple(dfs_dict.items()))
