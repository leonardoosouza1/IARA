import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np
import pandas as pd
import sys
import json
import glob
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.train.trainer_liquid import TrainerConfig, MultiAssetDataset
from src.model.iara_liquid import IARA

# Configuração da Página
st.set_page_config(layout="wide", page_title="SACI Neural Dashboard", page_icon="🧠")

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    h1 {color: #FF4B4B;}
    .stMetric {background-color: #0E1117; padding: 10px; border-radius: 5px; border: 1px solid #262730;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 SACI-Brain: Neural Inspector v6.11")

# --- Helpers ---
@st.cache_resource
def load_model(device='cpu', use_lab=False):
    base_dir = "checkpoints_lab" if use_lab else "checkpoints"
    path = f"{base_dir}/best_model.pth"
    if not Path(path).exists():
        # Fallback to interrupted if best not found
        path = f"{base_dir}/interrupted_model.pth"
    
    try:
        # Load com segurança para TrainerConfig
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Config
        config_data = checkpoint.get('config', None)
        if isinstance(config_data, TrainerConfig):
             config = config_data
        elif isinstance(config_data, dict):
             config = TrainerConfig(**config_data)
        else:
             config = TrainerConfig()
             
        # Model
        model = IARA(config).to(device)
        
        # State Dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Clean keys
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        return model, config
    except Exception as e:
        st.error(f"Erro ao carregar modelo ({path}): {e}")
        return None, None

@st.cache_resource
def get_dataset(_config, ticker="EURUSD"):
    # Cache do dataset para não recarregar sempre
    # Use legacy mode to access specific tickers
    return MultiAssetDataset(
        seq_len=_config.seq_len,
        stride=1,
        tickers=[ticker],
        augment=False,
        target_lookahead=5
    )

def get_latest_log_file():
    log_files = glob.glob("logs/training_history_*.jsonl")
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime) if 'os' in locals() else sorted(log_files)[-1]

def load_training_history():
    log_file = get_latest_log_file()
    if not log_file:
        return pd.DataFrame()
    
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'epoch' in entry: # Filter only epoch stats
                    # Flatten nested metrics
                    flat_entry = {
                        'epoch': entry['epoch'],
                        'train_loss': entry['train'].get('loss'),
                        'train_acc': entry['train'].get('accuracy'),
                        'val_loss': entry['val'].get('loss'),
                        'val_auc': entry['val'].get('auc'),
                        # Handle old logs vs new logs format
                        'lr': entry.get('metrics', {}).get('learning_rate', 0),
                        'memory_chronic': entry.get('memory', {}).get('chronic', 0),
                        'memory_mastered': entry.get('memory', {}).get('mastered', 0)
                    }
                    data.append(flat_entry)
            except:
                pass
    return pd.DataFrame(data)

# --- Hooks ---
class HookManager:
    def __init__(self):
        self.activations = {}
        self.attention = None
        
    def hook_cfc(self, name):
        def hook(module, input, output):
            # Output: [batch, hidden]
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(output.detach().cpu())
        return hook
    
    def hook_attn(self, module, input, output):
        self.attention = output.detach().cpu()

# --- Main Logic ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sidebar Configuration
st.sidebar.header("Configuration")
model_source = st.sidebar.radio("Model Source", ["Production (Full)", "Lab (Experimental)"], index=0)
is_lab = model_source == "Lab (Experimental)"

# Load Model
model, config = load_model(device, use_lab=is_lab)

# --- TABS ---
tab_live, tab_neu, tab_mkt = st.tabs(["🔥 Live Training", "🧠 Neural Neuroscience", "📉 Market Context"])

# --- TAB 1: Live Training ---
with tab_live:
    st.header("Monitoramento de Treino em Tempo Real")
    
    if st.button("🔄 Refresh Logs"):
        st.rerun()
        
    df_history = load_training_history()
    
    if not df_history.empty:
        # Metrics
        latest = df_history.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epoch", int(latest['epoch']))
        c2.metric("Val AUC", f"{latest.get('val_auc', 0):.4f}", delta=f"{latest.get('val_auc', 0) - 0.5:.4f}")
        c3.metric("Train Loss", f"{latest.get('train_loss', 0):.4f}")
        c4.metric("Learning Rate", f"{latest.get('lr', 0):.2e}")
        
        # Plots
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            fig_loss = px.line(df_history, x='epoch', y=['train_loss', 'val_loss'], title="Loss Curve (Focal Loss)")
            st.plotly_chart(fig_loss, use_container_width=True)
            
        with col_plot2:
            fig_auc = px.line(df_history, x='epoch', y=['val_auc', 'train_acc'], title="Validation AUC & Accuracy")
            st.plotly_chart(fig_auc, use_container_width=True)
            
        with st.expander("Memory Consistency (Selective Memory Strategy)"):
            if 'memory_chronic' in df_history.columns:
                 fig_mem = px.area(df_history, x='epoch', y=['memory_chronic', 'memory_mastered'], title="Curriculum Learning State")
                 st.plotly_chart(fig_mem, use_container_width=True)
    else:
        st.info("Nenhum log de treino encontrado. O treino começou?")

# --- TAB 2: Neuroscience ---
with tab_neu:
    if model:
        # Sidebar Controls
        st.sidebar.header("Inference Config")
        ticker = st.sidebar.selectbox("Asset", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"])
        
        # Dataset Loading
        ds = get_dataset(config, ticker)
        if len(ds) > 0:
            sample_idx = st.sidebar.slider("Sample Index (Time)", 0, len(ds)-1, len(ds)-1)
            
            # Inference Data
            x, y, w = ds[sample_idx]
            x_in = x.unsqueeze(0).to(device) # [1, seq, feat]
            
            # Feature Padding (Hack for 16->18 mismatch)
            current_dim = x_in.shape[-1]
            target_dim = config.input_dim
            
            if current_dim < target_dim:
                st.toast(f"Padding input {current_dim} -> {target_dim} (Adding Context/Trend Zeros)", icon="🔧")
                padding = torch.zeros(1, x_in.shape[1], target_dim - current_dim, device=device)
                x_in_padded = torch.cat([x_in, padding], dim=-1)
                x_in = x_in_padded
            
            # Hooks
            hooks = HookManager()
            handles = []
            
            # Register Hooks
            for i, layer in enumerate(model.cfc_layers):
                h = layer.register_forward_hook(hooks.hook_cfc(f"Layer {i+1} (Liquid)"))
                handles.append(h)
            
            if hasattr(model, 'self_attention'):
                h = model.self_attention.register_forward_hook(hooks.hook_attn)
                handles.append(h)
            
            # Forward
            with torch.no_grad():
                logits, _ = model(x_in)
                prob = torch.sigmoid(logits).item()
                
            # Remove Hooks
            for h in handles: h.remove()
            
            # --- Visualization Area ---
            c_prob, c_label, c_conf = st.columns(3)
            c_prob.metric("Model Probability", f"{prob:.2%}")
            c_label.metric("Ground Truth", "UP" if y.item() > 0.5 else "DOWN")
            c_conf.metric("Status", "CONFIDENT" if abs(prob-0.5) > 0.1 else "UNCERTAIN")
            
            # 1. Feature Importance (Weights Proxy)
            st.subheader("🕵️ Feature Attribution (O que o modelo olhou?)")
            
            # Try to get feature names from norm_params or default
            feature_names = [f"Feat {i}" for i in range(target_dim)]
            try:
                norm_params = torch.load("data/cache/norm_params.pt", weights_only=False)
                if 'feature_cols' in norm_params:
                     feature_names = norm_params['feature_cols']
            except:
                pass
                
            # Input Projection Weights (First Layer Importance)
            # Sum of absolute weights connected to each feature
            input_weights = model.input_proj.weight.abs().mean(dim=0).detach().cpu().numpy()
            
            # Adjust if dimension mismatch (model holds 18 weights, names might be 18)
            if len(input_weights) == len(feature_names):
                df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': input_weights})
                df_imp = df_imp.sort_values('Importance', ascending=True)
                
                fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title="Relevância Global das Features (Camada de Entrada)")
                st.plotly_chart(fig_imp, use_container_width=True)
            
            # 2. Neuron Activities
            st.subheader("🧠 Atividade dos Neurônios (Dinâmica Líquida)")
            if hooks.activations:
                # Use only the last layer or aggregate
                last_layer_name = list(hooks.activations.keys())[-1]
                activity_list = hooks.activations[last_layer_name]
                act_tensor = torch.stack(activity_list).squeeze(1).T.numpy() # [hidden, seq]
                
                # Heatmap
                fig_neuro = px.imshow(
                    act_tensor,
                    aspect='auto',
                    color_continuous_scale='Magma',
                    labels=dict(x="Tempo (Barras)", y="Neurônio ID"),
                    title=f"Mapa de Disparo - {last_layer_name}"
                )
                st.plotly_chart(fig_neuro, use_container_width=True)
                
            # 3. Attention Flow
            st.subheader("👁️ Fractal Attention Focus")
            if hooks.attention is not None:
                 attn_mag = torch.norm(hooks.attention.squeeze(0), dim=1).numpy()
                 # Smoothing
                 attn_series = pd.Series(attn_mag).rolling(5).mean().fillna(0)
                 
                 fig_attn = px.area(
                     x=range(len(attn_mag)), 
                     y=attn_series,
                     title="Atenção Temporal (Picos indicam momentos decisivos)",
                 )
                 st.plotly_chart(fig_attn, use_container_width=True)

        else:
            st.warning("Dataset vazio para este ticker.")
    else:
        st.error("Carregamento do modelo falhou.")

# --- TAB 3: Market ---
with tab_mkt:
    st.subheader("Input Feature Heatmap")
    if 'x_in' in locals():
        feat_data = x_in.squeeze(0).cpu().numpy().T
        fig_feat = px.imshow(feat_data, aspect='auto', title="Raw Input Features", origin='lower')
        st.plotly_chart(fig_feat, use_container_width=True)
