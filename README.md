# VinhoAI - Classificação Preditiva de Qualidade Enológica

**Trabalho Final da Disciplina de Machine Learning**  
**Pós-Graduação em Inteligência Artificial Generativa e Aplicações com LLMs – PUC Minas**  
**Professor:** Maurício Rodrigues da Silva  
**Aluno:** Vinicius Beltran (`vinicius.beltran@proton.me`)  
**Repositório GitHub:** [https://github.com/ViniciusBelt/vinho-ai-machine-learning](https://github.com/ViniciusBelt/vinho-ai-machine-learning)

---

## 1. Visão Geral do Projeto

O **VinhoAI** é uma solução completa de Ciência de Dados e Engenharia de Machine Learning desenvolvida para prever e certificar se um lote de vinho tinto atinge os critérios de **Alta Qualidade (Vinho Premium, nota sensorial ≥ 7)** ou se enquadra na categoria de **Qualidade Padrão / Comercial (< 7)** exclusivamente a partir de parâmetros físico-químicos laboratoriais mensuráveis.

Inspirado no projeto DiabetesAI da disciplina, este projeto implementou do zero:
1. **Notebook de Experimentos (`trabalho_final_experimento_ml.ipynb`):** Executado do início ao fim com EDA (4 gráficos detalhados), pipelines scikit-learn anti-vazamento, validação cruzada estratificada em 5 folds, comparação de 4 modelos (Regressão Logística, SVM, Random Forest e XGBoost), matrizes de confusão, curvas ROC comparativas e exportação via `joblib`.
2. **Interface Gráfica Interativa (Opção A - FastAPI + Web):** Backend assíncrono em FastAPI (`app.py`), validação estrita com Pydantic, frontend moderno e responsivo (`index.html` + `style.css`), botões de preenchimento rápido (presets), exibição de probabilidade calibrada e tratamento completo de erros amigáveis.
3. **Relatório Técnico em PDF (`relatorio_final_machine_learning.pdf`):** Documento formal de 8 páginas (atendendo ao requisito mínimo de 5 páginas) diagramado com ReportLab contendo Capa, Introdução, Dataset, Metodologia, Resultados, Prints reais da Interface e Conclusão.

---

## 2. Estrutura de Arquivos

```text
trabalho_final/
├── app.py                                   # Servidor FastAPI com endpoints REST e validações
├── templates/
│   └── index.html                          # Frontend Web HTML5 + JS interativo
├── static/
│   └── style.css                           # Identidade visual enológica e responsiva
├── modelo/
│   ├── modelo_vinho_pipeline.joblib         # Pipeline campeão serializado (Scaler + Random Forest)
│   └── metadata.json                       # Estatísticas e metadados das features
├── img/                                     # Figuras da EDA, avaliação e prints da tela
│   ├── eda_1_target_dist.png
│   ├── eda_2_correlation_matrix.png
│   ├── eda_3_key_features_dist.png
│   ├── eda_4_domain_bivariate.png
│   ├── eval_confusion_matrices.png
│   ├── eval_roc_curves.png
│   ├── screenshot_main.png                 # Print da tela principal limpa
│   ├── screenshot_success.png              # Print da predição válida de sucesso
│   └── screenshot_error.png                # Print do caso negativo com erro tratado
├── trabalho_final_experimento_ml.ipynb      # Notebook executado de ponta a ponta
├── relatorio_final_machine_learning.pdf     # Relatório técnico completo de 8 páginas
├── requirements.txt                         # Dependências do projeto
└── README.md                                # Instruções detalhadas de execução
```

---

## 3. Como Executar a Aplicação Web

### Passo 1: Instalação das Dependências
No terminal (PowerShell ou Bash), a partir da pasta raiz do projeto:

```bash
pip install -r trabalho_final/requirements.txt
```

### Passo 2: Iniciar o Servidor FastAPI
Execute o Uvicorn apontando para o módulo `trabalho_final.app`:

```bash
uvicorn trabalho_final.app:app --host 127.0.0.1 --port 8000 --reload
```

### Passo 3: Acessar no Navegador
Abra seu navegador em:
- **Interface Web:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Documentação Interativa (Swagger/OpenAPI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Endpoint de Saúde (Health Check):** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 4. Como Executar o Notebook de Experimentos

O notebook `trabalho_final_experimento_ml.ipynb` já se encontra totalmente executado com todas as saídas, tabelas e gráficos embutidos. Caso deseje reexecutá-lo do zero:

1. Abra o Jupyter Lab ou VS Code / Google Colab.
2. Selecione o kernel Python configurado.
3. Clique em **Kernel > Restart & Run All**.
4. O notebook executará em aproximadamente 15 a 20 segundos, revalidando todas as métricas e gerando o modelo `.joblib`.

---

## 5. Resumo do Desempenho dos Modelos

Avaliação no conjunto de teste independente (320 amostras):

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Regressão Logística | 81.25% | 40.00% | 79.07% | 0.5312 | 0.8824 |
| Support Vector Machine (SVM) | 84.69% | 45.83% | 76.74% | 0.5739 | 0.9015 |
| **Random Forest (Campeão)** | **91.25%** | **64.15%** | **79.07%** | **0.7083** | **0.9375** |
| XGBoost Classifier | 92.19% | 68.75% | 76.74% | 0.7253 | 0.9242 |

**Modelo Selecionado:** O **Random Forest** foi eleito o modelo campeão definitivo pela maior área sob a curva ROC (**0.9375**), excelente equilíbrio com Recall de **79.07%** (capturou 34 de 43 vinhos nobres do teste) e alta robustez contra ruídos e outliers enológicos.
