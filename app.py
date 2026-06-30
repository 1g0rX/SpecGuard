import os
import io
import base64
import numpy as np
from pathlib import Path
import joblib

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "static" / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

classes = ['violencia', 'nao_violencia']
img_size = (150, 150)

print("Carregando modelos...")
rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
svm_model = joblib.load(MODELS_DIR / "svm_model.pkl")
print("Modelos carregados.")

def perfil_radial(magnitude, n_aneis=20):
    centro = magnitude.shape[0] // 2
    raio_max = min(centro, magnitude.shape[1] // 2) - 1
    passo = raio_max / n_aneis
    perfil = []
    y, x = np.indices(magnitude.shape)
    r = np.sqrt((x - centro)**2 + (y - centro)**2)
    for i in range(n_aneis):
        r_min = i * passo
        r_max = (i + 1) * passo
        mascara = (r >= r_min) & (r < r_max)
        energia = magnitude[mascara].mean()
        perfil.append(energia)
    return np.array(perfil)

def extrair_caracteristicas(imagem_flat):
    img_3d = imagem_flat.reshape((150, 150, 3))
    feats = []
    for canal in range(3):
        f = np.fft.fft2(img_3d[:, :, canal])
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        feats.extend(perfil_radial(magnitude))
    return np.array(feats)

def gerar_plot_espectro(img_array):
    """Gera visualização: espectro com círculos + perfil radial."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Imagem original
    axes[0].imshow(img_array)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')

    # 2. Espectro com círculos (média dos 3 canais)
    espectro_medio = np.zeros((150, 150))
    for canal in range(3):
        f = np.fft.fft2(img_array[:, :, canal])
        fshift = np.fft.fftshift(f)
        espectro_medio += np.log(np.abs(fshift) + 1)
    espectro_medio /= 3

    centro = 75
    vmin = np.percentile(espectro_medio, 5)
    vmax = np.percentile(espectro_medio, 95)
    axes[1].imshow(espectro_medio, cmap='inferno', vmin=vmin, vmax=vmax)

    for i in range(1, 6):
        raio = i * 74 // 5
        circle = plt.Circle((centro, centro), raio, fill=False,
                          color='white', linewidth=0.8, linestyle='--', alpha=0.6)
        axes[1].add_patch(circle)
    axes[1].set_title('Espectro FFT (anéis concêntricos)')
    axes[1].axis('off')

    # 3. Perfil radial
    perfis = []
    for canal in range(3):
        f = np.fft.fft2(img_array[:, :, canal])
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        perfis.append(perfil_radial(magnitude))

    cores_perfil = ['#ff4444', '#44ff44', '#4488ff']
    labels = ['Canal R', 'Canal G', 'Canal B']
    for perfil, cor, label in zip(perfis, cores_perfil, labels):
        freqs = np.linspace(0, 100, len(perfil))
        axes[2].plot(freqs, perfil, color=cor, linewidth=2, label=label)

    axes[2].set_xlabel('Frequência (%)')
    axes[2].set_ylabel('Energia Média')
    axes[2].set_title('Perfil Radial (energia × frequência)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def processar_imagem(caminho):
    img = load_img(caminho, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_original = img_array.copy()
    img_blur = cv2.GaussianBlur(img_array, (5, 5), 0)
    features = extrair_caracteristicas(img_blur.flatten())
    return img_original, features.reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.after_request
def add_cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return resp

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'ok': True})

    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Arquivo vazio'}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'):
        ext = '.jpg'
    path = UPLOADS_DIR / f"upload{ext}"
    file.save(str(path))

    try:
        img_array, features = processar_imagem(path)
    except Exception as e:
        return jsonify({'error': f'Erro ao processar imagem: {str(e)}'}), 400

    pred_rf = rf_model.predict(features)[0]
    proba_rf = rf_model.predict_proba(features)[0]
    pred_svm = svm_model.predict(features)[0]
    proba_svm = svm_model.predict_proba(features)[0] if hasattr(svm_model, 'predict_proba') else None

    plot_b64 = gerar_plot_espectro(img_array)

    return jsonify({
        'rf': {
            'classe': classes[int(pred_rf)],
            'confianca': float(max(proba_rf)),
        },
        'svm': {
            'classe': classes[int(pred_svm)],
            'confianca': float(max(proba_svm)) if proba_svm is not None else None,
        },
        'plot': plot_b64,
    })

def run_server(port=5000, debug=False):
    app.run(host='0.0.0.0', port=port, debug=debug)

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    run_server(debug=False)
