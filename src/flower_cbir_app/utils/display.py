
from __future__ import annotations

import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def render_debug_bundle(bundle: dict):
    images = bundle.get('images', {})
    if images:
        keys = list(images.keys())
        cols = st.columns(min(3, len(keys)))
        for idx, key in enumerate(keys):
            with cols[idx % len(cols)]:
                st.image(images[key], caption=key, use_container_width=True)
    plots = bundle.get('plots', {})
    for name, payload in plots.items():
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(payload.get('x', list(range(len(payload['y'])))), payload['y'])
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    tables = bundle.get('tables', {})
    for name, payload in tables.items():
        st.markdown(f"**{name}**")
        st.dataframe(pd.DataFrame([payload]), use_container_width=True, hide_index=True)


def make_feature_config_dataframe(catalog: List, feature_state: dict) -> pd.DataFrame:
    rows = []
    for feature in catalog:
        state = feature_state[feature.key]
        rows.append({
            'group': feature.group,
            'feature': feature.name,
            'enabled': state['enabled'],
            'distance': state['distance'],
            'weight': state['weight'],
            'retrieval': not feature.is_meta,
            'default_on': feature.enabled_by_default,
            'dimension': feature.output_dim_display,
        })
    return pd.DataFrame(rows)
