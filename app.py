# app.py  ────────────────────────────────────────────────────────────
import streamlit as st
import tempfile, os
import agente_calcolo as agent     # contiene generate_sf() & calculate_ufp()

# ─────────────────────────  CONFIG  ────────────────────────────────
st.set_page_config(page_title="Function Point Estimator",
                   layout="wide",
                   page_icon=":chart_with_upwards_trend:")

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
st.markdown("<p style='font-size:20px;'>Calcola i Function Point secondo gli standard IFPUG in modo semplice e intuitivo!</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  INSTRUCTIONS  ─────────────────────────
st.markdown("<div class='instructions'>", unsafe_allow_html=True)
st.markdown("""
### Istruzioni:
1. Carica un file **.docx** contenente l'Analisi Requisiti Utente (ARU).
2. Premi **➊ Genera Specifica Funzionale** per ottenere la SF.
3. Dopo la verifica, premi **➋ Calcola UFP** per il conteggio.
""", unsafe_allow_html=True)
st.image("img2_software.png", width=80)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  UTILS  ─────────────────────────────────
def _make_temp_copy(file) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    tmp.write(file.getvalue()); tmp.close()
    return tmp.name

# ─────────────────────────  SESSION STATE  ────────────────────────
if "sf_text" not in st.session_state:
    st.session_state.sf_text  = None
    st.session_state.req_text = None

# ─────────────────────────  FILE UPLOAD  ──────────────────────────
uploaded = st.file_uploader("📄 Scegli un file .docx", type="docx")

if uploaded:
    tmp_path = _make_temp_copy(uploaded)
    st.success("File caricato con successo!")

    # -------- STEP 1: genera SF -----------------------------------
    if st.button("➊ Genera Specifica Funzionale"):
        with st.spinner("Generazione SF in corso…"):
            try:
                sf, req = agent.generate_sf(tmp_path)
                st.session_state.sf_text  = sf
                st.session_state.req_text = req
                st.success("Specifica Funzionale generata!")
            except Exception as e:
                st.error(f"Errore durante la generazione: {e}")

    # mostra SF se presente
    if st.session_state.sf_text:
        st.markdown("### 📄 Specifica Funzionale")
        st.write(st.session_state.sf_text)

        # ----- STEP 2: calcola UFP --------------------------------
        if st.button("➋ Calcola UFP"):
            with st.spinner("Calcolo UFP…"):
                try:
                    report = agent.calculate_ufp(
                        st.session_state.sf_text,
                        st.session_state.req_text
                    )
                    st.markdown("### 📊 Report UFP")
                    st.write(report)
                except Exception as e:
                    st.error(f"Errore nel calcolo UFP: {e}")

    # rimuovi file temporaneo
    os.remove(tmp_path)
