import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

# 1. Inicialização da Aplicação FastAPI
app = FastAPI(
    title="VinhoAI - Sistema Inteligente de Avaliação Enológica",
    description="API FastAPI para predição da qualidade de vinhos a partir de parâmetros físico-químicos.",
    version="1.0.0"
)

# Caminhos absolutos/relativos robustos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo", "modelo_vinho_pipeline.joblib")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Configurar diretório estático e templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# 2. Carregamento do modelo serializado com joblib (apenas no carregamento do módulo)
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Modelo não encontrado em: {MODEL_PATH}. Execute o notebook trabalho_final_experimento_ml.ipynb primeiro.")

model_pipeline = joblib.load(MODEL_PATH)
print("Pipeline de Machine Learning carregado com sucesso via joblib!")

# 3. Definição do Schema de Entrada com Validação Estrita (Pydantic)
# Faixas plausíveis baseadas em parâmetros reais de enologia (Cortez et al., 2009)
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=3.0, le=18.0, description="Acidez Fixa (g(tartárico)/dm³)")
    volatile_acidity: float = Field(..., ge=0.05, le=2.0, description="Acidez Volátil (g(acético)/dm³)")
    citric_acid: float = Field(..., ge=0.0, le=1.5, description="Ácido Cítrico (g/dm³)")
    residual_sugar: float = Field(..., ge=0.5, le=30.0, description="Açúcar Residual (g/dm³)")
    chlorides: float = Field(..., ge=0.01, le=1.0, description="Cloretos (g(NaCl)/dm³)")
    free_sulfur_dioxide: float = Field(..., ge=1.0, le=100.0, description="Dióxido de Enxofre Livre (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=5.0, le=350.0, description="Dióxido de Enxofre Total (mg/dm³)")
    density: float = Field(..., ge=0.980, le=1.050, description="Densidade (g/cm³)")
    pH: float = Field(..., ge=2.5, le=4.5, description="pH (Potencial Hidrogeniônico)")
    sulphates: float = Field(..., ge=0.2, le=2.5, description="Sulfatos (g(K2SO4)/dm³)")
    alcohol: float = Field(..., ge=7.0, le=17.0, description="Teor Alcoólico (% vol.)")

    @field_validator('total_sulfur_dioxide')
    @classmethod
    def validate_sulfur(cls, v, values):
        # Validação cruzada de domínio enológico: SO2 total não pode ser menor que SO2 livre
        free_so2 = values.data.get('free_sulfur_dioxide')
        if free_so2 is not None and v < free_so2:
            raise ValueError("O Dióxido de Enxofre Total não pode ser menor que o Dióxido de Enxofre Livre.")
        return v

# 4. Rotas da Aplicação
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renderiza a página web inicial com o formulário de predição."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
async def health_check():
    """Endpoint de checagem de saúde da API."""
    return {"status": "online", "model_loaded": True, "app": "VinhoAI"}

@app.post("/predict")
async def predict(features: WineFeatures):
    """
    Consome o modelo serializado para prever a qualidade do vinho.
    Recebe os 11 parâmetros físico-químicos e retorna a classe e probabilidade calibrada.
    """
    try:
        # Montar DataFrame com os nomes exatos esperados pelo Pipeline
        input_data = pd.DataFrame([{
            'fixed acidity': features.fixed_acidity,
            'volatile acidity': features.volatile_acidity,
            'citric acid': features.citric_acid,
            'residual sugar': features.residual_sugar,
            'chlorides': features.chlorides,
            'free sulfur dioxide': features.free_sulfur_dioxide,
            'total sulfur dioxide': features.total_sulfur_dioxide,
            'density': features.density,
            'pH': features.pH,
            'sulphates': features.sulphates,
            'alcohol': features.alcohol
        }])

        # Realizar a predição usando o pipeline exportado
        prediction_class = int(model_pipeline.predict(input_data)[0])
        probabilities = model_pipeline.predict_proba(input_data)[0]
        prob_premium = float(probabilities[1])
        prob_standard = float(probabilities[0])

        if prediction_class == 1:
            classe_formatada = "Vinho de Alta Qualidade (Premium)"
            badge_color = "success"
            interpretacao = (
                "O perfil físico-químico apresenta características ideais de maturação e equilíbrio enológico, "
                "com destaque para proporção adequada entre teor alcoólico, acidez volátil controlada e sulfatos preservados. "
                "Recomendado para linha de vinhos finos de guarda / reserva."
            )
        else:
            classe_formatada = "Vinho de Qualidade Padrão / Comercial"
            badge_color = "secondary"
            interpretacao = (
                "O perfil físico-químico é característico de vinhos comerciais para consumo imediato. "
                "Pode apresentar acidez volátil mais elevada ou menor equilíbrio alcoólico em relação à acidez total."
            )

        return {
            "status": "success",
            "prediction_class": prediction_class,
            "classe_formatada": classe_formatada,
            "probabilidade_premium": round(prob_premium * 100, 2),
            "probabilidade_padrao": round(prob_standard * 100, 2),
            "badge_color": badge_color,
            "interpretacao": interpretacao,
            "input_recebido": features.model_dump()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento da predição: {str(e)}")

# Handler amigável para erros de validação Pydantic
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": f"Erro de validação química: {str(exc)}"}
    )
