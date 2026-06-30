# SpecGuard

Classificação de violência contra a mulher em imagens via **perfil radial da FFT 2D** com **Random Forest** e **SVM**.

## Funcionamento

O pipeline transforma uma imagem em 60 características numéricas e usa dois modelos treinados para classificá-la:

```
Imagem → Gaussian Blur → FFT 2D → Perfil Radial (20 anéis × 3 canais) → 60 features → RF + SVM → resultado
```

### Etapas em detalhe

1. **Redimensionamento** — a imagem é redimensionada para 150×150 pixels
2. **Normalização** — os pixels são divididos por 255 (valores entre 0 e 1)
3. **Gaussian Blur** — filtro 5×5 suaviza a imagem e remove ruído de alta frequência
4. **FFT 2D** — cada canal RGB é transformado para o domínio da frequência com `np.fft.fft2`
5. **Espectro deslocado** — `fftshift` centraliza a frequência zero no meio da matriz
6. **Log do espectro** — `log(|F| + 1)` comprime a escala para visualização e análise
7. **Perfil radial** — 20 anéis concêntricos medem a energia média em cada faixa de frequência, repetido para R, G e B → 60 características
8. **Classificação** — os 60 valores alimentam Random Forest e SVM lado a lado
9. **Visualização** — matplotlib gera um gráfico com 3 painéis: imagem original, espectro com anéis, curvas energia×frequência

### Por que FFT + perfil radial?

A FFT revela padrões de frequência invisíveis no domínio espacial. O perfil radial resume o espectro inteiro em 60 valores que capturam como a energia se distribui das baixas (centro) às altas (bordas) frequências. Imagens de violência e não violência tendem a ter distribuições diferentes — o modelo aprende a distingui-las.

## Pré-requisitos

- Python 3.9+
- pip

## Como usar

```bash
# 1. clone o repositório
git clone git@github.com:1g0rX/SpecGuard.git
cd SpecGuard

# 2. crie e ative o ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 3. instale as dependências
pip install -r requirements.txt

# 4. execute
python specguard.py
```

O navegador abre automaticamente em `http://127.0.0.1:5000`.  
Pressione `Ctrl+C` no terminal para encerrar.

Os modelos já vêm treinados em `models/rf_model.pkl` e `models/svm_model.pkl`.

## Re-treinar

Pela interface web, vá na seção **Treinamento**, selecione a pasta que contém as subpastas `violencia/` e `nao_violencia/` com as imagens, e clique em **Treinar modelos**. O servidor processa as imagens, treina novos modelos e exibe as métricas na tela.

A estrutura esperada da pasta selecionada é:

```
pasta_selecionada/
├── violencia/
│   ├── img1.jpg
│   └── ...
└── nao_violencia/
    ├── img1.jpg
    └── ...
```

## Dataset

O modelo foi treinado com **24 imagens** (12 violência, 12 não violência). Cada imagem original é aumentada para 50 versões via rotação, zoom, deslocamento e espelhamento, totalizando 900 amostras de treino.

## Modelos

| Modelo | Kernel | Características |
|--------|--------|----------------|
| Random Forest | N árvores (grid search 1-96) | 60 features (20 anéis × 3 canais) |
| SVM | RBF | 60 features (20 anéis × 3 canais) |

## Arquivos essenciais

```
SpecGuard/
├── specguard.py            # entrada: sobe servidor + abre navegador
├── app.py                  # servidor Flask com API de predição
├── models/
│   ├── rf_model.pkl        # Random Forest treinado
│   └── svm_model.pkl       # SVM treinado
├── templates/
│   └── index.html          # interface web
├── requirements.txt        # dependências
└── README.md
```

## API

### `POST /api/predict`
Envia uma imagem (`multipart/form-data`, campo `image`), retorna JSON com classificação e gráfico:

```json
{
  "rf": { "classe": "violencia", "confianca": 0.94 },
  "svm": { "classe": "violencia", "confianca": 0.87 },
  "plot": "data:image/png;base64,..."
}
```

### `POST /api/retrain`
Envia os arquivos do dataset (`multipart/form-data`, campo `files[]` com `webkitRelativePath`), retorna JSON com métricas e matrizes de confusão (base64):

```json
{
  "rf": {
    "acuracia": 0.87,
    "precisao": 0.85,
    "revocacao": 0.90,
    "f1": 0.87,
    "arvores": 30,
    "matriz": [[120, 15], [10, 155]]
  },
  "svm": {
    "acuracia": 0.82,
    "precisao": 0.80,
    "revocacao": 0.86,
    "f1": 0.83,
    "matriz": [[115, 20], [12, 153]]
  },
  "imagens_originais": 24,
  "treino": 900,
  "teste": 300,
  "plots": {
    "rf": "data:image/png;base64,...",
    "svm": "data:image/png;base64,..."
  }
}
```

Após o treinamento, a interface exibe:
- **Matriz de confusão** para cada modelo
- **Acurácia, precisão, revocação e F1-score**
- Número de árvores usadas (RF)
- Quantidade de imagens originais, amostras de treino e de teste

## Tecnologias

- **Python** — Flask, numpy, scikit-learn, OpenCV, TensorFlow/Keras, matplotlib, joblib
- **Frontend** — HTML + CSS + JavaScript puros
