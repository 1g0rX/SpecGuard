import os, io, base64
import numpy as np
from pathlib import Path
import joblib

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "static" / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

classes = ['violencia', 'nao_violencia']
img_size = (150, 150)

# --- lazy model loading ---
rf_model = None
svm_model = None

def carregar_modelos():
    global rf_model, svm_model
    rf_path = MODELS_DIR / "rf_model.pkl"
    svm_path = MODELS_DIR / "svm_model.pkl"
    if rf_path.exists() and svm_path.exists():
        rf_model = joblib.load(rf_path)
        svm_model = joblib.load(svm_path)
        return True
    return False

# --- FFT / feature extraction ---
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

# --- training ---
def treinar_modelos(dataset_dir):
    """Treina modelos a partir de dataset_dir/violencia/ e dataset_dir/nao_violencia/."""
    X, y = [], []
    for label, cls in enumerate(classes):
        pasta = dataset_dir / cls
        if not pasta.exists():
            return None, f"Pasta '{cls}' nao encontrada em {dataset_dir}"
        fnames = sorted(os.listdir(pasta))
        if not fnames:
            return None, f"Pasta '{cls}' esta vazia"
        for fname in fnames:
            img = load_img(pasta / fname, target_size=img_size)
            X.append(img_to_array(img).flatten() / 255.0)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    n_por_classe = sum(y == 0)
    print(f"  Imagens carregadas: violencia={n_por_classe}, nao_violencia={len(y)-n_por_classe}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    datagen = ImageDataGenerator(
        preprocessing_function=lambda img: cv2.GaussianBlur(img, (5,5), 0),
        rotation_range=30, zoom_range=0.2,
        width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip=True, fill_mode='nearest'
    )

    def gerar_aumentadas(imgs, lbls, n=50):
        todas_X, todas_y = [], []
        for img_f, lb in zip(imgs, lbls):
            x = img_f.reshape((1, 150, 150, 3))
            for batch in datagen.flow(x, batch_size=1):
                feats = extrair_caracteristicas(batch[0].flatten())
                todas_X.append(feats)
                todas_y.append(lb)
                if len(todas_X) % n == 0:
                    break
        return np.array(todas_X), np.array(todas_y)

    X_train_aug, y_train_aug = gerar_aumentadas(X_train, y_train)
    X_test_aug, y_test_aug = gerar_aumentadas(X_test, y_test)

    print(f"  Treino: {X_train_aug.shape[0]} amostras, Teste: {X_test_aug.shape[0]} amostras")

    # RF grid search
    melhor_n, melhor_acc = 0, 0
    for n in range(1, 100, 5):
        accs = []
        for rep in range(5):
            clf = RandomForestClassifier(n_estimators=n, random_state=42+rep)
            clf.fit(X_train_aug, y_train_aug)
            accs.append(accuracy_score(y_test_aug, clf.predict(X_test_aug)))
        media = np.mean(accs)
        if media > melhor_acc:
            melhor_acc = media
            melhor_n = n

    rf = RandomForestClassifier(n_estimators=melhor_n, random_state=42)
    rf.fit(X_train_aug, y_train_aug)
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
    rf_acc = accuracy_score(y_test_aug, rf.predict(X_test_aug))

    svm = SVC(kernel='rbf', random_state=42, cache_size=2000, probability=True)
    svm.fit(X_train_aug, y_train_aug)
    joblib.dump(svm, MODELS_DIR / "svm_model.pkl")
    svm_acc = accuracy_score(y_test_aug, svm.predict(X_test_aug))

    global rf_model, svm_model
    rf_model = rf
    svm_model = svm

    return {
        'rf': { 'acuracia': round(rf_acc, 4), 'arvores': melhor_n },
        'svm': { 'acuracia': round(svm_acc, 4) },
        'treino': X_train_aug.shape[0],
        'teste': X_test_aug.shape[0],
        'imagens_originais': len(X),
    }, None

# --- plot ---
def gerar_plot_espectro(img_array):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')

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

# --- routes ---
@app.route('/')
def index():
    modelos_carregados = carregar_modelos()
    return render_template('index.html', modelos_prontos=modelos_carregados)

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

    if rf_model is None or svm_model is None:
        if not carregar_modelos():
            return jsonify({'error': 'Modelos nao encontrados. Treine primeiro.'}), 400

    pred_rf = rf_model.predict(features)[0]
    proba_rf = rf_model.predict_proba(features)[0]
    pred_svm = svm_model.predict(features)[0]
    proba_svm = svm_model.predict_proba(features)[0] if hasattr(svm_model, 'predict_proba') else None
    plot_b64 = gerar_plot_espectro(img_array)
    return jsonify({
        'rf': { 'classe': classes[int(pred_rf)], 'confianca': float(max(proba_rf)) },
        'svm': { 'classe': classes[int(pred_svm)], 'confianca': float(max(proba_svm)) if proba_svm is not None else None },
        'plot': plot_b64,
    })

@app.route('/api/retrain', methods=['POST', 'OPTIONS'])
def retrain():
    if request.method == 'OPTIONS':
        return jsonify({'ok': True})

    # Recebe arquivos com caminhos relativos (webkitdirectory)
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    temp_dir = BASE_DIR / "data" / "temp_retrain"
    for cls in classes:
        (temp_dir / cls).mkdir(parents=True, exist_ok=True)

    for f in files:
        rel = f.filename  # vem como "violencia/imagem.jpg" ou "nao_violencia/img.png"
        parts = Path(rel).parts
        if len(parts) < 2:
            continue
        cls_folder = parts[0]
        if cls_folder not in classes:
            continue
        dest = temp_dir / cls_folder / Path(rel).name
        f.save(str(dest))

    # Verifica se tem imagens
    total = sum(len(os.listdir(temp_dir / cls)) for cls in classes)
    if total == 0:
        return jsonify({'error': 'Nenhuma imagem encontrada nas pastas violencia/ e nao_violencia/'}), 400

    resultado, erro = treinar_modelos(temp_dir)

    # Limpa temporarios
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    if erro:
        return jsonify({'error': erro}), 400
    return jsonify(resultado)

def run_server(port=5000, debug=False):
    app.run(host='0.0.0.0', port=port, debug=debug)

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    run_server(debug=False)
