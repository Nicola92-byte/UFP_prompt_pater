"""ufp_agents.py
================
Due agenti distinti – uno genera la Specifica Funzionale (SF) dal documento ARU,
l'altro calcola gli Unadjusted Function Point (UFP) a partire dalla SF.
Le utility originali (FAISS, clamp, Agile‑context, ecc.) restano invariate.
"""
import os
import re
import pickle
import logging
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Funzioni di estrazione proprietarie
from estrazione_damas_wave import get_functional_requirements
from estrazione_dati_utili_wave import parse_aru_docx

###############################################################################
# ENV & OpenAI
###############################################################################
load_dotenv()
openai.api_type    = os.getenv("OPENAI_API_TYPE")
openai.api_base    = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key     = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME    = os.getenv("DEPLOYMENT_NAME")
if not openai.api_key or not DEPLOYMENT_NAME:
    raise ValueError("OPENAI_API_KEY / DEPLOYMENT_NAME mancante nel .env")

###############################################################################
# Logging
###############################################################################
logger = logging.getLogger("UFP_Agents")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO)
    fh = logging.FileHandler("app.log"); fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)

###############################################################################
# PDF manuale & FAISS (copia invariata rispetto allo script originale)
###############################################################################
def read_pdf_and_chunk(pdf_path: str, chunk_size: int = 500):
    logger.info("Lettura PDF %s", pdf_path)
    with open(pdf_path, "rb") as f:
        pages = [p.extract_text().strip() for p in PdfReader(f).pages if p.extract_text()]
    big = "\n".join(pages)
    return [big[i:i+chunk_size] for i in range(0, len(big), chunk_size)]


def build_faiss_index(chunks, model, idx_path="manual_FP_calc.index", emb_path="manual_FP_calc.npy"):
    if os.path.exists(idx_path) and os.path.exists(emb_path):
        logger.info("Caricamento FAISS index da cache")
        return faiss.read_index(idx_path)
    logger.info("Creazione FAISS index nuovo")
    emb = model.encode(chunks)
    idx = faiss.IndexFlatL2(emb.shape[1]); idx.add(emb.astype("float32"))
    faiss.write_index(idx, idx_path); np.save(emb_path, emb)
    return idx


def retrieve_context(query, idx, chunks, model, k=2):
    qemb = model.encode([query]).astype("float32")
    _, ids = idx.search(qemb, k)
    ctx = "\n".join(chunks[i] for i in ids[0])
    return ctx[:2000]


def get_manual_chunks(pdf_path: str, chunk_size=500, cache="manual_chunks.pkl"):
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    ch = read_pdf_and_chunk(pdf_path, chunk_size)
    pickle.dump(ch, open(cache, "wb")); return ch

###############################################################################
# Clamp & Agile helpers (immutati)
###############################################################################
CLAMP_MIN, CLAMP_MAX = 20, 200

def clamp_range(answer: str, lo=CLAMP_MIN, hi=CLAMP_MAX):
    m = re.search(r"Totale UFP\s*=\s*(\d+)", answer)
    if m:
        old = int(m.group(1)); new = max(min(old, hi), lo)
        if new != old:
            answer = re.sub(r"Totale UFP\s*=\s*\d+", f"Totale UFP = {new}", answer)
            logger.info("Clamp UFP %d→%d", old, new)
    return answer

def quick_pre_analysis(req_text):
    lines = req_text.splitlines()
    # contiamo quante "RF"
    rf_count = sum(1 for ln in lines if "RF" in ln)
    return f"Trovati {rf_count} requisiti con label 'RF'."

AGILE_KWS = {"product backlog","sprint","agile","metodologia agile"}

def is_agile(text: str):
    t = text.lower(); return any(kw in t for kw in AGILE_KWS)

def adjust_for_agile(answer: str, req: str, factor=0.4):
    if not is_agile(req):
        return answer
    m = re.search(r"Totale UFP\s*=\s*(\d+)", answer)
    if not m:
        return answer
    old = int(m.group(1)); new = int(old * factor)
    logger.info("Agile context → UFP %d→%d", old, new)
    return re.sub(r"Totale UFP\s*=\s*\d+", f"Totale UFP = {new}", answer)

###############################################################################
# Agent 1 – Generatore di Specifica Funzionale
###############################################################################
PROMPT_SF_TEMPLATE = """

Hai l'obiettivo di convertire un documento di Analisi Requisiti Utente (ARU) in un documento di Specifica Funzionale (SF) utile a un successivo calcolo dei function point secondo standard IFPUG. Il documento deve essere il più lungo e completo possibile. Considera che deve essere almeno 3-4 pagine. Se non riesci a dare l'output in un'unica risposta dividi in più risposte. 
Ecco il Prompt di Estrazione per Documento di Specifica Funzionale e Non Funzionale 
1.	Introduzione
Estrarre la sezione che descrive il contesto di riferimento, gli obiettivi del progetto, il committente e i vincoli temporali e di pianificazione. 
2.	Descrizione Generale del Sistema
Estrarre una panoramica generale del sistema, includendo l'ambito del progetto, gli utenti principali e le interfacce con altri sistemi.
 
3.	Definizione dei Boundary del Sistema
Definire i confini del sistema, distinguendo tra ciò che è interno e ciò che è esterno, specificando gli archivi logici interni (ILF) e gli archivi logici esterni (EIF).
Requisiti Funzionali
 
Estrarre e organizzare tutti i requisiti funzionali presenti nel documento ARU, includendo per ogni requisito:
- ID del requisito
- Titolo del requisito
- Descrizione dettagliata del requisito: includere i processi elementari, le funzionalità richieste e le regole di business pertinenti.
- Priorità del requisito
- Dipendenze: indicare eventuali dipendenze con altri requisiti o moduli del sistema.
 
Dettagli sui Dati e le Transazioni:
• Data Function (Funzioni di Dato): Specificare dettagliatamente gli Archivi Logici Interni (ILF) e gli Archivi Logici Esterni (EIF), descrivendo le entità e gli attributi principali.
• Transaction Function (Funzioni di Transazione): Definire le Entrate Esterne (EI), Uscite Esterne (EO) e Consultazioni Esterne (EQ), includendo la descrizione dei processi e delle logiche di elaborazione associate.
 
4.	Requisiti Non Funzionali
Estrarre e catalogare i requisiti non funzionali, includendo dettagli su:
• Prestazioni: specifiche richieste di performance.
• Sicurezza: requisiti relativi alla protezione dei dati e alla sicurezza del sistema.
• Usabilità: requisiti sull'interfaccia utente e l'esperienza d'uso.
• Affidabilità: standard di affidabilità e disponibilità richiesti.
• Manutenibilità: requisiti per la manutenzione e l'aggiornamento del sistema.
• Portabilità: necessità di operare in diversi ambienti o piattaforme.
 
5.	Regole di Business
Documentare tutte le regole di business che influenzano le operazioni e i calcoli effettuati dal sistema.
 
6.	Eccezioni e Condizioni Speciali
Descrivere eventuali eccezioni, condizioni speciali o situazioni non standard che il sistema deve gestire.
 
7.	Report e Output del Sistema
Documentare tutti i report e gli output generati dal sistema, specificando il contenuto, il formato e la frequenza.
 
8.	Interfacce Utente
Dettagliare le interfacce utente, includendo le schermate e le interazioni previste, per identificare correttamente le Entrate Esterne (EI).
 
9.	Processi di Interfacciamento con Altri Sistemi
Descrivere come il sistema interagisce con altri sistemi, identificando le interfacce esterne e il flusso di dati tra sistemi.
 
10.	Casi d'Uso e Scenari Operativi
Fornire una descrizione dei casi d'uso principali e degli scenari operativi che il sistema deve supportare.
 
11.	Dettagli sull'Architettura del Sistema
Fornire una panoramica dell'architettura del sistema, descrivendo i componenti principali e le loro interazioni.
 
12.	Allegati e Appendici
Estrarre eventuali allegati e appendici, includendo:
- Glossario: definizioni di termini tecnici e acronimi.
- Diagrammi e Modelli: diagrammi di flusso, modelli di dati e altri diagrammi tecnici.
- Prototipi e Schermate: bozze o esempi di schermate dell'interfaccia utente.
 
13.	Note
Assicurarsi di mantenere l'ordine e l'organizzazione originale dei contenuti durante l'estrazione.
Ogni sezione del documento di specifica deve essere chiaramente separata e etichettata, seguendo la struttura delineata sopra.
"""

def agent_generate_sf(aru_text: str, summary: str = "", ufp_info: str = "") -> str:

    messages = [
        {"role":"system","content":"Sei un analista senior di Specifiche Funzionali."},
        {"role":"user",  "content": f"""
        Hai l'obiettivo di convertire un documento di Analisi Requisiti Utente (ARU) in un documento di Specifica Funzionale (SF) utile a un successivo calcolo dei function point secondo standard IFPUG. 
        
        [CONTESTO AGGIUNTIVO]
        Ecco un riepilogo dell'analisi preliminare:
        {summary}
        
        Ecco informazioni estratte automaticamente utili per il conteggio dei Function Point:
        {ufp_info}
        
        [REQUISITI FUNZIONALI]
        {aru_text}
        
        Ora segui il Prompt seguente per generare il documento SF completo:
        {PROMPT_SF_TEMPLATE}
        """}
    ]
    resp = openai.ChatCompletion.create(engine=DEPLOYMENT_NAME, messages=messages, max_tokens=6000, temperature=0.0)
    sf = resp.choices[0].message.content.strip()
    logger.info("Specifiche Funzionali generate (agent 1)")
    return sf

###############################################################################
# Agent 2 – Calcolo UFP
###############################################################################
PROMPT_UFP_TEMPLATE = """
[SPECIFICA FUNZIONALE]
{sf}

Sei un esperto analista di Function Point certificato IFPUG con esperienza nella stima di progetti di varie dimensioni. Utilizza la specifica funzionale che hai appena creato. Il tuo compito è analizzare questi requisiti e calcolare gli Unadjusted Function Point (UFP) secondo lo standard IFPUG 4.3.1. Segui attentamente questi passaggi:
  
1. Analisi preliminare:
  - Leggi attentamente il documento delle spedifica funzionale.
  - Riassumi in 3-5 frasi la portata e il contesto del progetto, inclusa una stima preliminare delle sue dimensioni (piccolo, medio, grande).
  - Identifica e elenca tutti i processi elementari e i gruppi di dati logici menzionati esplicitamente.
 
2. Identificazione e classificazione delle funzioni:
  - Identifica e classifica le funzioni di tipo dati (ILF e EIF), limitandoti a quelle chiaramente definite nei requisiti.
  - Identifica e classifica le funzioni di tipo transazionale (EI, EO, EQ), considerando solo i processi elementari distinti.
  - Per ogni funzione identificata, fornisci una breve giustificazione della classificazione e spiega perché non potrebbe essere classificata diversamente.
 
3. Valutazione della complessità:
  - Per ogni funzione di tipo dati, determina il numero di RET e DET, giustificando chiaramente le tue decisioni.
  - Per ogni funzione transazionale, determina il numero di FTR e DET, giustificando chiaramente le tue decisioni.
  - Usa le matrici di complessità fornite nel documento IFPUG per determinare la complessità di ciascuna funzione.
 
4. Calcolo degli UFP (Unadjusted Function Points):
  - Assegna il valore di UFP appropriato a ciascuna funzione in base alla sua complessità.
  - Somma tutti i valori UFP per ottenere il totale.
  - Fornisci una breve analisi del totale UFP in relazione alla tua stima preliminare delle dimensioni del progetto.
 
5. Presentazione dei risultati:
  - Fornisci una tabella riassuntiva con il conteggio dettagliato per tipo di funzione.
  - Presenta il calcolo finale dei Function Point, mostrando chiaramente tutti i passaggi.
 
6. Analisi di sensibilità:
  - Identifica le 3-5 decisioni che hanno avuto il maggior impatto sul conteggio finale.
  - Spiega come cambierebbe il risultato se queste decisioni fossero state prese diversamente.
 
7. Verifica finale:
   - Rivedi il tuo conteggio complessivo e assicurati che sia coerente con la portata e la complessità del progetto come descritto nei requisiti.
   - Se noti incongruenze, rivedi i passaggi precedenti e giustifica eventuali modifiche.
 
Ricorda di essere preciso e coerente in tutte le tue valutazioni. Giustifica chiaramente ogni decisione che ha un impatto significativo sul conteggio finale. Se ci sono ambiguità nei requisiti, esplicita le tue assunzioni e spiega come queste influenzano il conteggio.

"""

def agent_calculate_ufp(sf_text: str, requirements_text: str) -> str:
    messages = [
        {"role":"system","content":"Sei un analista Function Point IFPUG esperto."},
        {"role":"user",  "content": PROMPT_UFP_TEMPLATE.format(sf=sf_text)}
    ]
    resp = openai.ChatCompletion.create(engine=DEPLOYMENT_NAME, messages=messages, max_tokens=4000, temperature=0.0)
    answer = resp.choices[0].message.content.strip()
    answer = clamp_range(answer)
    answer = adjust_for_agile(answer, requirements_text)
    logger.info("Report UFP generato (agent 2)")
    return answer

###############################################################################
# Agenti per app.py
###############################################################################

# --- all’interno di agente_calcolo.py ---------------------------

def generate_sf(docx_path: str) -> tuple[str, str]:
    """
    Ritorna:
        sf_text   – Specifica Funzionale generata (markdown o testo)
        req_text  – testo requisiti (serve dopo per Agile / clamp)
    """
    # aru_text            = get_functional_requirements(docx_path, use_regex=True)
    aru_text = get_functional_requirements(docx_path, use_regex=True)
    sf_text             = agent_generate_sf(aru_text)
    return sf_text, aru_text


def calculate_ufp(sf_text: str, requirements_text: str) -> str:
    """
    Usa Agent 2 e restituisce il report UFP in markdown.
    """
    return agent_calculate_ufp(sf_text, requirements_text)



###############################################################################
# Pipeline completa
###############################################################################
PDF_MANUAL_PATH = r"C:\Users\A395959\PycharmProjects\UFP_estimator\Function_Point_calcManual.pdf"


def run_pipeline(docx_path: str):
    # 0) Estrai testo ARU
    logger.info("Estrazione ARU da %s", docx_path)
    aru_text            = get_functional_requirements(docx_path, use_regex=True)
    pre_analysis        = quick_pre_analysis(aru_text)
    ufp_info, _, summary      = parse_aru_docx(docx_path)

    # 1) Agent 1 – Specifiche Funzionali
    sf_text = agent_generate_sf(aru_text, summary=summary, ufp_info=ufp_info)

    with open("specifica_funzionale.md", "w", encoding="utf-8") as f:
        f.write(sf_text)

    # 2) Agent 2 – Calcolo UFP
    ufp_report = agent_calculate_ufp(sf_text, aru_text)
    with open("ufp_report.md", "w", encoding="utf-8") as f:
        f.write(ufp_report)

    return sf_text, ufp_report, pre_analysis, ufp_info

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    DOCX_PATH = r"C:\Users\A395959\PycharmProjects\UFP_estimator\ARU_dir\ARU - STL 2023 Wave 1.docx"
    sf, report, pre, info = run_pipeline(DOCX_PATH)
    print("\n=== PRE‑ANALISI ===\n", pre)
    print("\n=== INFO UFP (dal docx) ===\n", info)
    print("\n=== SPECIFICA FUNZIONALE ===\n", sf)
    print("\n=== REPORT UFP ===\n", report)
