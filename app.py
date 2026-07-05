import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
from zoneinfo import ZoneInfo
from datetime import datetime
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --bg-deep:      #0F0A0E;
        --bg-panel:     #1A0F17;
        --bg-card:      #22101D;
        --accent-red:   #E8305A;
        --accent-gold:  #D4A840;
        --accent-rose:  #FF6B8A;
        --text-primary: #F0E8EE;
        --text-muted:   #9A7E8E;
        --border:       #4A1E35;
    }

    .stApp {
        background: linear-gradient(135deg, #0F0A0E 0%, #1A0D15 50%, #100810 100%);
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 820px; }

    h1, h2, h3, h4, h5, h6, p, label,
    .stMarkdown, .stText { color: var(--text-primary) !important; }

    .hero-banner {
        background: linear-gradient(120deg, #1E0D18 0%, #180A14 60%, #120810 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 36px 32px 28px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% -20%, rgba(232,48,90,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: clamp(22px, 4vw, 34px);
        font-weight: 700;
        color: var(--accent-red) !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 0 0 20px rgba(232,48,90,0.5), 0 0 60px rgba(232,48,90,0.2);
        margin: 0 0 6px;
    }
    .hero-sub {
        font-size: 13px;
        color: var(--text-muted) !important;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .hero-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-red), transparent);
        margin: 20px auto 0;
        width: 60%;
        opacity: 0.5;
    }
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--accent-red) !important;
        margin-bottom: 14px;
        padding-left: 4px;
    }
    .input-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 28px 28px 20px;
        margin-bottom: 24px;
    }
    .stNumberInput > label {
        font-size: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
    .stNumberInput input {
        background-color: #150810 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 17px !important;
        padding: 10px 14px !important;
    }
    .stNumberInput input:focus {
        border-color: var(--accent-red) !important;
        box-shadow: 0 0 0 2px rgba(232,48,90,0.15) !important;
    }
    .unit-badge {
        display: inline-block;
        background: rgba(232,48,90,0.08);
        border: 1px solid rgba(232,48,90,0.25);
        color: var(--accent-red) !important;
        border-radius: 6px;
        font-size: 11px;
        font-family: 'Share Tech Mono', monospace;
        padding: 2px 8px;
        margin-left: 6px;
        vertical-align: middle;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7A1530 0%, #A01E40 50%, #C42550 100%) !important;
        color: #F0E8EE !important;
        border: 1px solid rgba(232,48,90,0.40) !important;
        border-radius: 10px !important;
        height: 52px !important;
        width: 100% !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 15px !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        box-shadow: 0 0 20px rgba(232,48,90,0.2) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #931A3A 0%, #C42550 50%, #E8305A 100%) !important;
        box-shadow: 0 0 30px rgba(232,48,90,0.40) !important;
        transform: translateY(-1px) !important;
    }
    .result-card {
        background: linear-gradient(135deg, #1A0E10 0%, #1C0A14 100%);
        border: 1px solid #5C1A30;
        border-radius: 14px;
        padding: 32px 28px;
        text-align: center;
        margin-top: 28px;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(212,168,64,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .result-label {
        font-size: 11px;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: var(--text-muted) !important;
        margin-bottom: 10px;
    }
    .result-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: clamp(36px, 6vw, 56px);
        font-weight: 700;
        color: var(--accent-gold) !important;
        text-shadow: 0 0 24px rgba(212,168,64,0.55), 0 0 60px rgba(212,168,64,0.2);
        line-height: 1.1;
    }
    .result-unit {
        font-size: 14px;
        color: var(--text-muted) !important;
        letter-spacing: 2px;
        margin-top: 6px;
    }
    .result-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #5C1A30, transparent);
        margin: 20px 0 16px;
    }
    .result-note {
        font-size: 12px;
        color: var(--text-muted) !important;
        letter-spacing: 1px;
    }
    .stars {
        font-size: 26px;
        margin-top: 6px;
        letter-spacing: 4px;
    }
    .history-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px 12px;
        margin-top: 8px;
    }
    .history-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #2A1020;
        font-family: 'Share Tech Mono', monospace;
        font-size: 13px;
    }
    .history-row:last-child { border-bottom: none; }
    .history-num { color: var(--text-muted) !important; min-width: 28px; }
    .history-params { color: var(--text-muted) !important; font-size: 11px; flex: 1; padding: 0 12px; }
    .history-val { color: var(--accent-gold) !important; font-weight: 700; font-size: 15px; }
    .history-time { color: #5A2535 !important; font-size: 10px; min-width: 55px; text-align: right; }
    .stAlert { border-radius: 10px !important; }
    .footer {
        text-align: center;
        margin-top: 44px;
        padding-top: 20px;
        border-top: 1px solid #2A1020;
    }
    .footer p {
        font-size: 11px !important;
        color: #5A2535 !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  ANN MODEL  (4-input version — update if needed)
# ─────────────────────────────────────────────
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        return self.model(x)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    m = ANN()
    m.load_state_dict(torch.load("model.pkl", map_location="cpu"))
    m.eval()
    sc = joblib.load("scaler.pkl")
    return m, sc

with st.spinner(""):
    try:
        model, scaler = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        model_error = str(e)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def quality_stars(score):
    full  = int(round(score))
    full  = max(1, min(10, full))
    stars = "★" * min(full, 5) + "☆" * max(0, 5 - full)
    return stars

def quality_label(score):
    if score >= 7:   return "Excellent 🍷"
    elif score >= 6: return "Good"
    elif score >= 5: return "Average"
    else:            return "Below Average"

# ─────────────────────────────────────────────
#  PDF GENERATOR
# ─────────────────────────────────────────────
def generate_pdf(history):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm)

    red  = colors.HexColor("#E8305A")
    gold = colors.HexColor("#D4A840")
    muted = colors.HexColor("#9A7E8E")

    title_style = ParagraphStyle("title", fontName="Helvetica-Bold",
        fontSize=18, textColor=red, alignment=TA_CENTER, spaceAfter=4)
    sub_style = ParagraphStyle("sub", fontName="Helvetica",
        fontSize=9, textColor=muted, alignment=TA_CENTER, spaceAfter=16)
    section_style = ParagraphStyle("section", fontName="Helvetica-Bold",
        fontSize=10, textColor=red, spaceBefore=14, spaceAfter=6)
    normal_style = ParagraphStyle("normal", fontName="Helvetica",
        fontSize=9, textColor=colors.HexColor("#333333"), spaceAfter=4)

    story = []
    story.append(Paragraph("🍷 WINE QUALITY PREDICTION REPORT", title_style))
    story.append(Paragraph("AI Wine Analysis  |  ML Quality Scoring  |  Sommelier AI", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=red, spaceAfter=16))

    now = datetime.datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%d %B %Y, %I:%M:%S %p")

    story.append(Paragraph(f"Report Generated : {now}", normal_style))
    story.append(Paragraph(f"Total Predictions : {len(history)}", normal_style))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("PREDICTION HISTORY", section_style))

    table_data = [[
        "#", "Time", "Volatile Acidity", "Citric Acid",
        "Sulphates", "Alcohol (%)", "Quality Score"
    ]]
    for i, row in enumerate(history, 1):
        table_data.append([
            str(i), row["time"],
            f"{row['VA']:.3f}", f"{row['CA']:.3f}",
            f"{row['SU']:.2f}", f"{row['AL']:.1f}",
            f"{row['output']:.2f}",
        ])

    col_widths = [0.8*cm, 1.8*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]
    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#2A0A18")),
        ("TEXTCOLOR",     (0,0),(-1,0), red),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,0), 8),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("BOTTOMPADDING", (0,0),(-1,0), 8),
        ("TOPPADDING",    (0,0),(-1,0), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#1A0F17"), colors.HexColor("#22101D")]),
        ("TEXTCOLOR",     (0,1),(-2,-1), colors.HexColor("#F0E8EE")),
        ("TEXTCOLOR",     (-1,1),(-1,-1), gold),
        ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1),(-1,-1), 8),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#4A1E35")),
        ("TOPPADDING",    (0,1),(-1,-1), 6),
        ("BOTTOMPADDING", (0,1),(-1,-1), 6),
    ]))
    story.append(tbl)

    if history:
        outputs = [r["output"] for r in history]
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("SUMMARY STATISTICS", section_style))
        avg = sum(outputs) / len(outputs)
        summary_data = [
            ["Metric", "Value"],
            ["Highest Quality Score", f"{max(outputs):.2f}"],
            ["Lowest Quality Score",  f"{min(outputs):.2f}"],
            ["Average Quality Score", f"{avg:.2f}"],
            ["Grade",                 quality_label(avg)],
        ]
        stbl = Table(summary_data, colWidths=[8*cm, 5*cm])
        stbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#2A0A18")),
            ("TEXTCOLOR",  (0,0),(-1,0), red),
            ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0),(-1,0), 9),
            ("BACKGROUND", (0,1),(-1,-1), colors.HexColor("#1A0F17")),
            ("TEXTCOLOR",  (0,1),(0,-1), colors.HexColor("#F0E8EE")),
            ("TEXTCOLOR",  (1,1),(1,-1), gold),
            ("FONTNAME",   (0,1),(-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,1),(-1,-1), 9),
            ("ALIGN",      (0,0),(-1,-1), "CENTER"),
            ("GRID",       (0,0),(-1,-1), 0.5, colors.HexColor("#4A1E35")),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ]))
        story.append(stbl)

    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
        color=colors.HexColor("#4A1E35")))
    story.append(Paragraph(
        "Wine Quality AI  |  ML Analysis  |  Sommelier Intelligence",
        sub_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🍷 Wine Quality Predictor</p>
        <p class="hero-sub">AI Sommelier · ML Analysis · Red Wine Intelligence</p>
        <div class="hero-divider"></div>
    </div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Model failed to load — ensure `model.pkl` & `scaler.pkl` are in the working directory.\n\n`{model_error}`")
    st.stop()

# ─────────────────────────────────────────────
#  INPUT SECTION
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">⬡ Chemical Parameters</p>', unsafe_allow_html=True)
st.markdown('<div class="input-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Volatile Acidity** <span class='unit-badge'>g/dm³</span>", unsafe_allow_html=True)
    v1 = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0,
        value=0.50, step=0.01, format="%.3f", label_visibility="collapsed")

    st.markdown("**Citric Acid** <span class='unit-badge'>g/dm³</span>", unsafe_allow_html=True)
    v2 = st.number_input("Citric Acid", min_value=0.0, max_value=1.0,
        value=0.25, step=0.01, format="%.3f", label_visibility="collapsed")

with col2:
    st.markdown("**Sulphates** <span class='unit-badge'>g/dm³</span>", unsafe_allow_html=True)
    v3 = st.number_input("Sulphates", min_value=0.3, max_value=2.0,
        value=0.60, step=0.01, format="%.2f", label_visibility="collapsed")

    st.markdown("**Alcohol** <span class='unit-badge'>% vol</span>", unsafe_allow_html=True)
    v4 = st.number_input("Alcohol (%)", min_value=8.0, max_value=15.0,
        value=10.0, step=0.1, format="%.1f", label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────
predict_clicked = st.button("🍷  ANALYSE WINE QUALITY")

if predict_clicked:
    try:
        data_scaled = scaler.transform([[v1, v2, v3, v4]])
        data = torch.tensor(data_scaled, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(data).item()

        prediction = round(max(1.0, min(10.0, prediction)), 2)
        stars = quality_stars(prediction)
        label = quality_label(prediction)

        st.session_state.history.append({
            "time": datetime.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S"),
            "VA": v1, "CA": v2, "SU": v3, "AL": v4,
            "output": prediction,
        })

        st.markdown(f"""
            <div class="result-card">
                <p class="result-label">Predicted Wine Quality Score</p>
                <p class="result-value">{prediction:.2f}</p>
                <div class="stars">{stars}</div>
                <p class="result-unit">{label} &nbsp;·&nbsp; Scale 1–10</p>
                <div class="result-divider"></div>
                <p class="result-note">
                    VA = {v1:.3f} g/dm³ &nbsp;|&nbsp;
                    CA = {v2:.3f} g/dm³ &nbsp;|&nbsp;
                    SU = {v3:.2f} g/dm³ &nbsp;|&nbsp;
                    AL = {v4:.1f}%
                </p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Inference failed: {e}")

# ─────────────────────────────────────────────
#  HISTORY SECTION
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<p class="section-label" style="margin-top:36px;">⬡ Prediction History</p>', unsafe_allow_html=True)
    st.markdown('<div class="history-card">', unsafe_allow_html=True)

    for i, row in enumerate(reversed(st.session_state.history), 1):
        idx = len(st.session_state.history) - i + 1
        lbl = quality_label(row['output'])
        st.markdown(f"""
            <div class="history-row">
                <span class="history-num">#{idx}</span>
                <span class="history-params">
                    VA={row['VA']:.3f} &nbsp; CA={row['CA']:.3f} &nbsp; SU={row['SU']:.2f} &nbsp; AL={row['AL']:.1f}%
                </span>
                <span class="history-val">{row['output']:.2f} · {lbl}</span>
                <span class="history-time">{row['time']}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Buttons row ──
    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])

    with col_a:
        pdf_bytes = generate_pdf(st.session_state.history)
        fname = f"wine_quality_report_{datetime.datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d_%H%M%S')}.pdf"
        st.markdown("""
        <style>
        .stDownloadButton > button {
            background: linear-gradient(90deg, #4A1E35 0%, #6B2848 50%, #8C3260 100%) !important;
            color: #D4A840 !important;
            border: 1px solid rgba(212,168,64,0.4) !important;
            border-radius: 10px !important;
            height: 52px !important;
            width: 100% !important;
            font-family: 'Share Tech Mono', monospace !important;
            font-size: 15px !important;
            letter-spacing: 3px !important;
            text-transform: uppercase !important;
            font-weight: 700 !important;
            box-shadow: 0 0 20px rgba(212,168,64,0.15), inset 0 0 20px rgba(212,168,64,0.04) !important;
            transition: all 0.25s ease !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(90deg, #6B2848 0%, #8C3260 50%, #AE3C78 100%) !important;
            box-shadow: 0 0 35px rgba(212,168,64,0.30), inset 0 0 20px rgba(212,168,64,0.08) !important;
            transform: translateY(-2px) !important;
            border-color: #D4A840 !important;
        }
        .stDownloadButton > button:active {
            transform: translateY(0) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.download_button(
            label="📄  DOWNLOAD PDF REPORT",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
        )

    with col_b:
        if st.button("🗑️  CLEAR"):
            st.session_state.history = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
    <div class="footer">
        <p>Wine Quality AI · ML Analysis · Sommelier Intelligence</p>
    </div>
""", unsafe_allow_html=True)