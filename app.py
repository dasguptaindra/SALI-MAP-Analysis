# app.py - Advanced SAS Map Streamlit app (Simplified)
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

st.set_page_config(page_title="Advanced SAS Map (SALI) — Streamlit", layout="wide")
st.title("🧭 Advanced SAS Map Generator — SALI / Activity Cliffs")

# ---------- Sidebar: Upload & Params ----------
st.sidebar.header("Input & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Fingerprint type selection
fingerprint_type = st.sidebar.selectbox(
    "Fingerprint Type", 
    ["ECFP4", "ECFP6", "MACCS"], 
    index=0
)

# Conditional parameters based on fingerprint type
if fingerprint_type.startswith("ECFP"):
    radius = st.sidebar.slider("Morgan radius", 1, 4, 2 if fingerprint_type == "ECFP4" else 3)
    n_bits = st.sidebar.selectbox("Fingerprint size (bits)", [512, 1024, 2048], index=2)
else:  # MACCS
    radius = None
    n_bits = 167  # MACCS has fixed size

color_by = st.sidebar.selectbox("Color by", ["SALI", "MaxActivity"])
top_n = st.sidebar.number_input("Top cliffs to highlight (top SALI)", min_value=1, max_value=1000, value=10)
max_pairs_plot = st.sidebar.number_input("Max pairs to plot", min_value=2000, max_value=200000, value=10000, step=1000)

# ---------- Functions ----------
def compute_fingerprints(smiles_list, fp_type, radius, n_bits):
    """Compute fingerprints for a list of SMILES"""
    fps = []
    valid_idx = []
    invalid_smiles = []
    
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            invalid_smiles.append(s)
            continue
            
        try:
            if fp_type == "ECFP4":
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_bits)
            elif fp_type == "ECFP6":
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=n_bits)
            elif fp_type == "MACCS":
                fp = MACCSkeys.GenMACCSKeys(m)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
            
            fps.append(fp)
            valid_idx.append(i)
        except Exception as e:
            invalid_smiles.append(f"{s} (error: {str(e)})")
            continue
    
    return fps, valid_idx, invalid_smiles

def compute_similarity_matrix(fps):
    """Compute pairwise Tanimoto similarity matrix"""
    n = len(fps)
    sim_matrix = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                try:
                    s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s
                except Exception:
                    sim_matrix[i, j] = 0.0
                    sim_matrix[j, i] = 0.0
    
    return sim_matrix

# ---------- Main UI ----------
if uploaded_file is None:
    st.info("📁 Upload a CSV file containing columns: SMILES and an activity (e.g. pIC50).")
    st.stop()

# Read file
try:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Molecules", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.write("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Column selectors
cols = list(df.columns)
smiles_col = st.selectbox("SMILES column", cols, index=0)
activity_col = st.selectbox("Activity column", cols, index=1 if len(cols)>1 else 0)
id_col_opt = ["None"] + cols
id_col = st.selectbox("Optional ID column", id_col_opt, index=0)

# Generate button
if st.button("🚀 Generate SAS map and analyze"):
    st.info("Processing — this may take a while for large datasets.")
    
    # Basic filter and validation
    df_clean = df.dropna(subset=[smiles_col, activity_col]).copy()
    if len(df_clean) == 0:
        st.error("No valid data after removing rows with missing SMILES or activity values.")
        st.stop()
        
    # Parse activities
    try:
        activities = df_clean[activity_col].astype(float).values
    except Exception as e:
        st.error(f"Activity column conversion to float failed: {e}")
        st.stop()

    ids = df_clean[id_col].astype(str).values if id_col != "None" else np.array([f"Mol_{i+1}" for i in range(len(df_clean))])
    smiles_list = df_clean[smiles_col].astype(str).values

    # Step 1: Compute fingerprints
    st.write(f"### Step 1: Computing {fingerprint_type} fingerprints...")
    fps, valid_idx, invalid_smiles = compute_fingerprints(smiles_list, fingerprint_type, radius, n_bits)
    
    if invalid_smiles:
        st.warning(f"{len(invalid_smiles)} invalid SMILES found and excluded.")
        with st.expander("Show invalid SMILES"):
            for bad_smiles in invalid_smiles[:10]:
                st.write(bad_smiles)
            if len(invalid_smiles) > 10:
                st.write(f"... and {len(invalid_smiles) - 10} more")
    
    # Keep only valid entries
    activities = activities[valid_idx]
    ids = ids[valid_idx]
    smiles_list = smiles_list[valid_idx]
    n = len(fps)
    
    if n < 2:
        st.error("Need at least 2 valid molecules to compute pairs.")
        st.stop()

    st.success(f"✅ Fingerprints computed for {n} molecules using {fingerprint_type}.")

    # Step 2: Compute similarity matrix
    st.write("### Step 2: Computing pairwise Tanimoto similarities...")
    sim_matrix = compute_similarity_matrix(fps)
    st.success(f"✅ Similarity matrix computed for {n} molecules.")

    # Step 3: Build pairs and compute SALI
    st.write("### Step 3: Building pair list and computing SALI...")
    pairs = []
    eps_distance = 1e-2
    
    total_pairs = n * (n - 1) // 2
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    pair_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if pair_count % 1000 == 0:
                progress_text.text(f"Processing pairs: {pair_count:,}/{total_pairs:,}")
                progress_bar.progress(min(pair_count / total_pairs, 1.0))
            
            sim = sim_matrix[i, j]
            act_diff = float(abs(activities[i] - activities[j]))
            max_val = float(max(activities[i], activities[j]))
            distance = max(1.0 - sim, eps_distance)
            sali = act_diff / distance
            
            pairs.append({
                "Mol1_idx": i, "Mol2_idx": j,
                "Mol1_ID": ids[i], "Mol2_ID": ids[j],
                "SMILES1": smiles_list[i], "SMILES2": smiles_list[j],
                "Activity1": activities[i], "Activity2": activities[j],
                "Similarity": sim, "Activity_Diff": act_diff,
                "MaxActivity": max_val, "SALI": sali
            })
            pair_count += 1
    
    progress_text.empty()
    progress_bar.empty()
    
    if not pairs:
        st.error("No valid pairs were generated. Check your data.")
        st.stop()
        
    pairs_df = pd.DataFrame(pairs)
    st.success(f"✅ Created {len(pairs_df):,} molecular pairs.")

    # Mark top N SALI cliffs
    top_n_use = min(int(top_n), len(pairs_df))
    pairs_df["is_top_cliff"] = False
    if top_n_use > 0:
        top_idxs = pairs_df.nlargest(top_n_use, "SALI").index
        pairs_df.loc[top_idxs, "is_top_cliff"] = True

    # ---------- RESULTS VISUALIZATION ----------
    st.markdown("---")
    st.header("📊 Results Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["SAS Map", "Statistics", "Top Cliffs"])
    
    with tab1:
        st.subheader("SAS Activity Landscape Map")
        
        # Optionally subsample for plotting
        plot_df = pairs_df
        if len(pairs_df) > max_pairs_plot:
            st.warning(f"Too many pairs ({len(pairs_df):,}) — subsampling {max_pairs_plot:,} for plotting.")
            top_df = pairs_df[pairs_df["is_top_cliff"]]
            other_df = pairs_df[~pairs_df["is_top_cliff"]].sample(n=max_pairs_plot - len(top_df), random_state=42)
            plot_df = pd.concat([top_df, other_df], ignore_index=True)

        plot_df = plot_df.copy()
        plot_df["marker_size"] = np.where(plot_df["is_top_cliff"], 10, 6)
        plot_df["symbol"] = np.where(plot_df["is_top_cliff"], "diamond", "circle")

        color_col = color_by
        
        # Create the plot
        fig = px.scatter(
            plot_df,
            x="Similarity",
            y="Activity_Diff",
            color=color_col,
            size="marker_size",
            symbol="symbol",
            hover_data=["Mol1_ID", "Mol2_ID", "Similarity", "Activity_Diff", "SALI"],
            title=f"SAS Map ({fingerprint_type}) — colored by {color_by} (top {top_n_use} cliffs highlighted)",
            width=1000,
            height=650,
        )
        fig.update_traces(marker=dict(opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics for the plot
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pairs Plotted", len(plot_df))
        with col2:
            st.metric("Average Similarity", f"{plot_df['Similarity'].mean():.3f}")
        with col3:
            st.metric("Average Activity Diff", f"{plot_df['Activity_Diff'].mean():.3f}")
        with col4:
            st.metric("Max SALI", f"{plot_df['SALI'].max():.3f}")
    
    with tab2:
        st.subheader("Statistical Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # SALI distribution
            fig_sali = px.histogram(pairs_df, x="SALI", nbins=50, 
                                  title="SALI Distribution")
            st.plotly_chart(fig_sali, use_container_width=True)
            
            # Similarity distribution
            fig_sim = px.histogram(pairs_df, x="Similarity", nbins=50,
                                 title="Similarity Distribution")
            st.plotly_chart(fig_sim, use_container_width=True)
        
        with col2:
            # Activity difference distribution
            fig_act = px.histogram(pairs_df, x="Activity_Diff", nbins=50,
                                 title="Activity Difference Distribution")
            st.plotly_chart(fig_act, use_container_width=True)
            
            # Summary statistics table
            st.subheader("Summary Statistics")
            stats_df = pairs_df[['Similarity', 'Activity_Diff', 'SALI']].describe()
            st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        st.subheader(f"Top {top_n_use} Activity Cliffs")
        
        top_cliffs = pairs_df.nlargest(top_n_use, "SALI").reset_index(drop=True)
        
        # Display top cliffs in an expandable table
        with st.expander("View All Top Cliffs", expanded=True):
            st.dataframe(top_cliffs[['Mol1_ID', 'Mol2_ID', 'Similarity', 
                                   'Activity_Diff', 'SALI', 'MaxActivity']], 
                       use_container_width=True)
        
        # Show top 5 cliffs with more details
        st.subheader("Top 5 Most Significant Cliffs")
        for i, (idx, row) in enumerate(top_cliffs.head(5).iterrows()):
            with st.expander(f"Cliff #{i+1}: {row['Mol1_ID']} vs {row['Mol2_ID']} (SALI: {row['SALI']:.2f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{row['Mol1_ID']}**")
                    st.code(f"SMILES: {row['SMILES1']}")
                    st.write(f"Activity: {row['Activity1']:.3f}")
                with col2:
                    st.write(f"**{row['Mol2_ID']}**")
                    st.code(f"SMILES: {row['SMILES2']}")
                    st.write(f"Activity: {row['Activity2']:.3f}")
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Similarity", f"{row['Similarity']:.3f}")
                with col4:
                    st.metric("Activity Difference", f"{row['Activity_Diff']:.3f}")
                with col5:
                    st.metric("SALI", f"{row['SALI']:.3f}")

    # ---------- DOWNLOAD SECTION ----------
    st.markdown("---")
    st.header("📥 Download Results")
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Full pairs data as CSV
        csv_data = pairs_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Pairs (CSV)",
            data=csv_data,
            file_name=f"SAS_pairs_full_{fingerprint_type}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Top cliffs only
        top_cliffs_csv = top_cliffs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Top Cliffs Only (CSV)",
            data=top_cliffs_csv,
            file_name=f"top_{top_n}_cliffs_{fingerprint_type}.csv",
            mime="text/csv"
        )

    st.success("🎉 Analysis complete! Use the download buttons above to save your results.")
