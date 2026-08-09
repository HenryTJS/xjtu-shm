#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据加载器：统一加载应变、声发射、光纤数据，并进行时间同步

仅负责加载和对齐，不做全局归一化（归一化由 OnlineNormalizer 在线完成）。
"""
import os
import numpy as np
import pandas as pd

from .config import BASE_DIR


class DataLoader:
    """数据加载器：统一加载应变、声发射、光纤数据，并进行时间同步"""

    def __init__(self, group_id):
        self.group_id = str(group_id).zfill(3)
        self.data_dir = os.path.join(BASE_DIR, self.group_id)
        self.strain = self.ae = self.fo = self.synced_data = None

    def _detect_encoding(self, file_path):
        for enc in ['utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    f.readline()
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return 'latin-1'

    def _find_file(self, name_patterns):
        for p in name_patterns:
            fp = os.path.join(self.data_dir, p)
            if os.path.exists(fp):
                return fp
        return None

    def _unify_columns(self, df, col_map_rules):
        df.columns = df.columns.str.strip()
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace(' ', '').replace('(', '').replace(')', '').replace('（', '').replace('）', '').replace('_', '')
            for pattern, target in col_map_rules:
                if pattern in cl:
                    col_map[col] = target
                    break
        for i, (pattern, target) in enumerate([('time', 'time'), ('strain', 'strain')]):
            if target not in col_map.values() and len(df.columns) > i:
                col_map[df.columns[i]] = target
        return df.rename(columns=col_map)

    def load_strain(self):
        fp = self._find_file([f'{self.group_id}应变.csv', f'{self.group_id}应变.xlsx'])
        if fp is None:
            print(f'  [警告] 未找到应变文件: {self.data_dir}')
            return None
        if fp.endswith('.csv'):
            encoding = self._detect_encoding(fp)
            with open(fp, 'r', encoding=encoding) as f:
                sep = '\t' if '\t' in f.readline() else ','
            df = pd.read_csv(fp, sep=sep, encoding=encoding, engine='python')
        else:
            df = pd.read_excel(fp)
        df = self._unify_columns(df, [
            (c, 'time') for c in ['时间', 'time', 'timestamp']
        ] + [
            (c, 'strain') for c in ['应变', 'strain', '幅值']
        ])
        df = df[['time', 'strain']].copy()
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['strain'] = pd.to_numeric(df['strain'], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        self.strain = df
        return df

    def load_ae(self):
        fp = os.path.join(self.data_dir, f'{self.group_id}声发射.csv')
        if not os.path.exists(fp):
            print(f'  [警告] 未找到声发射文件: {fp}')
            return None
        df = pd.read_csv(fp, encoding='utf-8-sig', engine='python')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        self.ae = df
        return df

    def load_fo(self):
        fp = self._find_file([f'{self.group_id}光纤.csv', f'{self.group_id}光纤.xlsx'])
        if fp is None:
            print(f'  [警告] 未找到光纤文件: {self.data_dir}')
            return None
        df = pd.read_csv(fp, encoding='utf-8-sig', engine='python') if fp.endswith('.csv') else pd.read_excel(fp)
        df.columns = df.columns.str.strip()
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace(' ', '').replace('(', '').replace(')', '').replace('（', '').replace('）', '').replace('_', '')
            if any(t in cl for t in ['时间', 'time', 'timestamp']):
                col_map[col] = 'time'
            elif cl.startswith('s') and len(col.strip()) <= 3:
                col_map[col] = col.strip()
            elif 'fiber' in cl and 's' in cl:
                col_map[col] = col.split('_')[-1]
        if 'time' not in col_map.values() and len(df.columns) > 0:
            col_map[df.columns[0]] = 'time'
        df = df.rename(columns=col_map)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        self.fo = df
        return df

    def load_all(self):
        print(f'\n>>> 加载组 {self.group_id} 数据')
        self.load_strain(); self.load_ae(); self.load_fo()
        return self

    def sync_timeline(self):
        if self.strain is None:
            return None
        timeline = self.strain[['time']].copy().sort_values('time').drop_duplicates('time').reset_index(drop=True)
        timeline['time'] = timeline['time'].astype(float)
        result = timeline.merge(self.strain, on='time', how='left')
        if self.fo is not None:
            fo_t = self.fo[['time']].copy().sort_values('time').drop_duplicates('time').reset_index(drop=True)
            fo_t['time'] = fo_t['time'].astype(float)
            td = np.diff(np.sort(result['time'].unique()))
            tol = np.median(td) * 1.5 if len(td) > 0 else 1.0
            result = pd.merge_asof(result.sort_values('time'), self.fo.sort_values('time').astype({'time': float}),
                                   on='time', direction='nearest', tolerance=tol)
            fo_cols = [c for c in result.columns if c.startswith('s') and len(c) <= 3]
            if fo_cols:
                result[fo_cols] = result[fo_cols].ffill()
        if self.ae is not None:
            ae_time = np.linspace(result['time'].min(), result['time'].max(), len(self.ae))
            target = result['time'].values
            ae_vals = self.ae.values.T
            interp = np.array([np.interp(target, ae_time, col) for col in ae_vals]).T
            for i, col in enumerate(self.ae.columns):
                result[f'ae_{col}'] = interp[:, i]
        self.synced_data = result
        return result
