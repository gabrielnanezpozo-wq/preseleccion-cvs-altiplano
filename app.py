import streamlit as st
import pypdf
import pandas as pd
from google import genai

# Configuración de la pantalla de la App
st.set_page_config(page_title="Confecciones Altiplano - Preselección IA", layout="wide")

st.title("🏢 Confecciones Altiplano S.A.")
st.subheader("Módulo de Preselección Algorítmica con IA (Evaluación Ciega)")

# Barra lateral para ingresar la clave API
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Ingresa tu Gemini API Key:", type="password")
    st.markdown("---")
    st.info("Esta aplicación procesa lotes de CVs en PDF, aplica anonimización ciega y revela únicamente el contacto del rango 80% - 100%.")

# Carga de archivos PDF
archivos_pdf = st.file_uploader("Carga los archivos PDF de los CVs", type=["pdf"], accept_multiple_files=True)

PROMPT_MAESTRO = """
Actúa como un Arquitecto de Datos y Especialista en IA Ética y RRHH, especializado en la mitigación de sesgos algorítmicos y optimización de flujos de reclutamiento para la industria de manufactura y operaciones.

Tu objetivo es procesar un lote de CVs para la preselección automatizada del puesto de Supervisor de Línea de Producción Textil en Planta Arequipa (Confecciones Altiplano S.A.), ejecutando los PUNTOS 1 AL 3 y generando de manera directa la lista de contacto del PUNTO 4.

CONTEXTO TÉCNICO Y OBJETIVO ÉTICO:
- Empresa: Confecciones Altiplano S.A. (~900 operarios).
- Meta ética: Mitigar el sesgo histórico de contratación (histórico 92% hombres vs. 35% postulantes mujeres calificadas) mediante una evaluación basada 100% en mérito técnico y competencias operativas reales.

REGLAS DE EVALUACIÓN Y PARÁMETROS AVANZADOS DE RRHH:

1. CUMPLE REQUISITOS MÍNIMOS (Filtro Duro - Pasa / No Pasa):
   - Mínimo 3 años de experiencia en líneas de producción, manufactura o gestión de operarios de planta realizando labores del puesto o afines.
   - FORMACIÓN ACADÉMICA Y CLÁUSULA DE HOMOLOGACIÓN POR TRAYECTORIA EXPERTA:
     * Opción A: Grado Técnico Titulado o Bachiller/Titulado Universitario en áreas afines.
     * Opción B (Homologación por Trayectoria Empírica Avanzada): Si el candidato NO cuenta con grado académico formal pero demuestra 3 o más años de trayectoria práctica (especialmente en jefatura de módulos, calibración de maquinaria, resolución de cuellos de botella y guía operativa de taller), NO SE LE ASIGNA 0 NI SE LE DESCARTA. Se homologa su maestría práctica como equivalente técnico directo y pasa a la fase de scoring.
   - REGULARIZACIÓN DE CARGOS: Evalúa las funciones reales descritas (control de mermas, balanceo de líneas, gestión de personal, tareo, mantenimiento de maquinaria) y no únicamente la palabra exacta "Supervisor" (acepta "Encargado", "Coordinador", "Líder de Módulo", "Jefe de Grupo", "Técnico Mecánico de Costura").
   - GESTIÓN DE MOVILIDAD GEOGRÁFICA:
     * Residencia en Arequipa -> APROBADO DIRECTO (Local).
     * Residencia en otra provincia (ej. Lambayeque, Lima, Cusco) -> APROBADO CONDICIONAL si el CV incluye disponibilidad de mudanza O si presenta un alto perfil técnico (Exp >= 4 años o Grado técnico/univ/empírico).
     * DESCARTE DIRECTO: Aplicar por residencia ÚNICAMENTE si existe negativa explícita a mudarse o si no cumple con la experiencia técnica mínima.
   - REGLA DE DESCARTE AUTOMÁTICO (Puntaje Técnico = 0):
     * Aplicar ÚNICAMENTE si el candidato tiene menos de 3 años de experiencia total O proviene de un rubro totalmente ajeno sin transferencia operativa (ej. Agronomía, Ventas minoristas).

2. ANONIMIZACIÓN OPERATIVA EN EVALUACIÓN (Capa Ciega):
   - Asigna a cada postulación un código neutro correlativo: "Candidato_001", "Candidato_002", "Candidato_003", etc.
   - Mide la experiencia acumulada total. NO penalices por lagunas laborales ni por la ausencia de títulos universitarios si el dominio del oficio en planta está demostrado.

3. PUNTAJE TÉCNICO Y SCORING OBJETIVO (Base 100 Puntos):

   - EXPERIENCIA OPERATIVA Y JEFATURA DE TALLER (Máx 50 pts):
     * De 3 a 5 años de experiencia relevante: 30 pts.
     * De 6 a 8 años de experiencia en planta/módulos: 40 pts.
     * Más de 9 años de experiencia sénior en jefatura de módulo, taller o planta: 50 pts.

   - DIVERSIFICACIÓN DE COMPETENCIAS PRÁCTICAS Y OPERATIVAS (Máx 25 pts):
     * Pilar 1: Eficiencia y Maquinaria (Hasta 10 pts): Calibración/mantenimiento de máquinas, solución de cuellos de botella y balanceo de líneas.
     * Pilar 2: Control de Procesos y Calidad (Hasta 10 pts): Control de mermas, fichas técnicas, métricas de producción, 5S o Lean Manufacturing.
     * Pilar 3: Liderazgo y Normativa de Planta (Hasta 5 pts): Gestión de operarios/tareo, resolución de conflictos de taller, SST (Seguridad Industrial) o certificaciones verificables.

   - FORMACIÓN ACADÉMICA / HOMOLOGACIÓN PRÁCTICA (Máx 25 pts):
     * Titulado Universitario / Bachiller: 25 pts.
     * Técnico Titulado: 20 pts.
     * Perfil Empírico Avanzado (Sin título pero > 5 años de trayectoria práctica comprobada en taller/módulos): 15 pts.
     * Perfil Empírico Básico (3 a 4 años en taller sin título): 10 pts.

REGLAS ESTRICTAS DE CORTES DE SELECCIÓN:
- Prioridad Top (80 a 100 pts): Pasa a lista de contacto directo.
- Validar Mudanza (80 a 100 pts + Provincia): Pasa a lista de contacto directo con indicación de confirmar reubicación.
- En Reserva / Apto para Evaluación Técnica en Planta (70 a 79 pts): Se mantiene en banco de talento en la tabla para prueba de campo (ideal para perfiles empíricos sénior).
- Descartado (< 70 pts o No cumple Filtro Duro): Eliminado automáticamente.

FORMATO Y ESTRUCTURA OBLIGATORIA DE LA RESPUESTA:

En una sola respuesta, entrega OBLIGATORIAMENTE dos secciones:

SECCIÓN 1: TABLA RESUMEN ANÓNIMA
| Código Candidato | Cumple Requisitos MÍNIMOS | Puntaje Técnico (0-100) | Diferencial Competitivo (Punto Fuerte) | Resultado y Acción Sugerida |
| :---: | :---: | :---: | :--- | :--- |

SECCIÓN 2: DIRECTORIO DE CONTACTO DIRECTO (SOLO CANDIDATOS 80% - 100%)
Debajo de la tabla, genera automáticamente el desglose de los postulantes de Prioridad Top con el siguiente formato:

📌 CANDIDATOS HABILITADOS PARA CONTACTO E INTERVENCIÓN HUMANA (80% - 100%):
- [Código Candidato]: [Nombre Real / Nombre del Archivo Original] | Puntaje: [X pts] | Acción: Citar a entrevista presencial y prueba técnica en planta.
- [Código Candidato]: [Nombre Real / Nombre del Archivo Original] | Puntaje: [X pts] | Acción: Contactar para confirmar disponibilidad de reubicación a Arequipa.

(Si un candidato tiene menos de 80 pts o está Descartado, queda estrictamente prohibido incluirlo en este directorio de contacto).
"""

if st.button("🚀 Procesar Lote de CVs", type="primary"):
    if not api_key:
        st.error("Por favor, ingresa tu API Key en la barra lateral.")
    elif not archivos_pdf:
        st.warning("Por favor, sube al menos un archivo PDF.")
    else:
        st.info("Extrayendo texto de los PDFs y ejecutando evaluación algorítmica...")
        
        # 1. Extraer texto de cada PDF
        lote_texto = []
        for idx, pdf in enumerate(archivos_pdf, start=1):
            codigo_id = f"Candidato_{idx:03d}"
            reader = pypdf.PdfReader(pdf)
            texto_pdf = ""
            for page in reader.pages:
                texto_pdf += page.extract_text() or ""
            
            expediente = f"\n--- INICIO EXPEDIENTE: {codigo_id} (Archivo Real: {pdf.name}) ---\n{texto_pdf}\n--- FIN EXPEDIENTE ---\n"
            lote_texto.append(expediente)

        # 2. Conectar con la API de Gemini
        try:
            client = genai.Client(api_key=api_key)
            prompt_final = PROMPT_MAESTRO + "\n\nTEXTO DE LOS CVs DE ENTRADA:\n" + "".join(lote_texto)
            
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt_final,
            )
            
            # 3. Mostrar resultados
            st.success("¡Evaluación completada con éxito!")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"Ocurrió un error al procesar con la IA: {e}")