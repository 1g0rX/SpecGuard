# SpecGuard 🔬🛡️

Classificação de violência contra a mulher em imagens via **Transformada de Fourier 2D** e **Random Forest / SVM**.

## Estrutura

```
SpecGuard/
├── src/
│   ├── rf_classifier.py      # Random Forest + FFT
│   └── svm_classifier.py     # SVM RBF + FFT
├── data/
│   └── originais/
│       ├── violencia/        # 12 imagens
│       └── nao_violencia/    # 12 imagens
├── results/                  # Métricas e gráficos (gerado)
├── requirements.txt
└── README.md
```

## Como usar

```bash
# 1. ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 2. dependências
pip install -r requirements.txt

# 3. treinar modelos (primeira vez apenas)
python src/train_models.py

# 4. executar a interface
python specguard.py
```

A interface abre automaticamente no navegador em `http://127.0.0.1:5000`.
Pressione `Ctrl+C` no terminal para encerrar.

## Pipeline

`Imagem → FFT 2D → baixas frequências (32×32×3) → Classificador → Resultado`

## Resultados

| Classificador | Acurácia |
|---|---|
| RF + FFT | **~87%** |
| SVM + FFT | ~72% |
