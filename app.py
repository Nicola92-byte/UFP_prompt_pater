# app.py  ────────────────────────────────────────────────────────────
import streamlit as st
import tempfile, os

# Importiamo sia le singole funzioni che la pipeline completa
import agente_calcolo as agent     # contiene ancora generate_sf() & calculate_ufp()
from agente_calcolo import run_pipeline

# ─────────────────────────  CONFIG  ────────────────────────────────
st.set_page_config(
    page_title="Function Point Estimator",
    layout="wide",
    page_icon=":chart_with_upwards_trend:"
)

# ─────────────────────────  CSS  ───────────────────────────────────
st.markdown("""
<style>
body           {background-color:#f0f8ff; font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;}
.main          {background-color:#e6f2ff; padding:20px; border-radius:10px;}
.stButton>button{
    background-color:#007acc; color:white; border:none; border-radius:5px;
    padding:10px 20px; font-size:16px; font-weight:600;}
.stButton>button:hover{background-color:#005f99;}
.header        {text-align:center; padding:20px;}
.instructions  {text-align:center; font-size:18px; padding:10px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────  HEADER  ────────────────────────────────
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.image("img1_calcolo.png", width=150)
st.markdown("<h1>Function Point Estimator</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:20px;'>Calcola i Function Point secondo gli standard IFPUG in modo semplice e intuitivo!</p>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  INSTRUCTIONS  ─────────────────────────
st.markdown("<div class='instructions'>", unsafe_allow_html=True)
st.markdown("""
### Istruzioni:
1. Carica un file **.docx** contenente l'Analisi Requisiti Utente (ARU).
2. Premi **➊ Genera Specifica Funzionale** per eseguire in un solo step la generazione della SF e il calcolo UFP.
3. Se vuoi, premi **➋ Mostra Report UFP** per rivedere il risultato.
""", unsafe_allow_html=True)
st.image("img2_software.png", width=80)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  UTILS  ─────────────────────────────────
def _make_temp_copy(file) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    tmp.write(file.getvalue())
    tmp.close()
    return tmp.name

# ─────────────────────────  SESSION STATE  ────────────────────────
if "sf_text" not in st.session_state:
    st.session_state.sf_text      = None
    st.session_state.ufp_report   = None
    st.session_state.pre_analysis = None
    st.session_state.ufp_info     = None

# ─────────────────────────  FILE UPLOAD  ──────────────────────────
uploaded = st.file_uploader("📄 Scegli un file .docx", type="docx")

if uploaded:
    tmp_path = _make_temp_copy(uploaded)
    st.success("File caricato con successo!")

    # -------- STEP 1: genera SF + calcola UFP ----------------------
    if st.button("➊ Genera Specifica Funzionale"):
        with st.spinner("Esecuzione pipeline (SF + UFP)…"):
            try:
                # Esegue l'intera pipeline: SF + UFP
                sf_text, ufp_report, pre_analysis, ufp_info = run_pipeline(tmp_path)
                # Salva in session state
                st.session_state.sf_text      = sf_text
                st.session_state.ufp_report   = ufp_report
                st.session_state.pre_analysis = pre_analysis
                st.session_state.ufp_info     = ufp_info
                st.success("Pipeline completata: SF e UFP pronti!")
            except Exception as e:
                st.error(f"Errore durante l'esecuzione: {e}")

    # Se abbiamo già generato la SF, la mostriamo
    if st.session_state.sf_text:
        st.markdown("### 📄 Specifica Funzionale")
        st.write(st.session_state.sf_text)

        # ----- STEP 2: (opzionale) mostra report UFP ---------------
        if st.button("➋ Mostra Report UFP"):
            st.markdown("### 📊 Report UFP")
            st.write(st.session_state.ufp_report)

        # (Facoltativo) Mostra anche pre-analysis e ufp_info
        # st.markdown("**Pre-analysis:**")
        # st.write(st.session_state.pre_analysis)
        # st.markdown("**UFP info:**")
        # st.write(st.session_state.ufp_info)

    # Pulizia file temporaneo
    os.remove(tmp_path)
