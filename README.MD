# 🚗 AI-Powered Driver Fatigue Monitoring System  
*Sistema Inteligente de Monitoreo de Fatiga del Conductor – Classic vs IA*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

---

## 🌐 Language / Idioma
- [🇺🇸 English](#english-version)
- [🇪🇸 Español](#versión-en-español)

---

# 🇺🇸 English Version

## 📋 Project Overview
This project implements a **driver fatigue monitoring system** with two complementary approaches:

1. **Classic (rule-based) approach** – Uses handcrafted features and thresholds:
	 - Eye Aspect Ratio (EAR) for eye closure
	 - Mouth Aspect Ratio (MAR) for yawns
	 - Head pose tilt estimation
2. **AI (deep learning) approach** – Uses convolutional neural networks (CNNs):
	 - Eye CNN → probability of *eyes closed*
	 - Yawn CNN → probability of *yawn*
	 - Drowsy CNN → probability of *global drowsiness* using the full face

The goal is to **compare the classic vs AI approach** in terms of behavior and detection quality, using the same camera input.

## 🎯 Objectives
- Build a real-time fatigue detection system focused on drivers.
- Provide a **classic baseline** and an **AI-based alternative**.
- Enable side-by-side experiments and analysis (classic vs AI).

## ✨ Main Features
- Real-time face and landmark detection using MediaPipe Face Mesh.
- Classic metrics:
	- EAR, MAR and head tilt calculation.
	- Rule-based fatigue detection and CSV logging.
- AI models:
	- Eye, yawn and drowsy classifiers based on MobileNet.
	- Probability bars overlay (eyes closed, yawning, drowsy).
- Webcam-based demo, designed for driver scenario (front-facing camera).

## 🗂 Project Structure (simplified)
```text
.
├── main.py              # Classic fatigue detector (rule-based)
├── ai_main.py           # AI-based fatigue detector (CNNs)
├── requirements.txt
├── data/
│   ├── cew/             # Closed Eyes in the Wild dataset
│   ├── fyp/             # Additional eye dataset
│   ├── yawdd/           # YawDD driver drowsiness dataset
│   └── drowsy/          # Processed dataset for drowsy CNN
└── src/
		├── classic/
		│   ├── config.py
		│   ├── detector.py     # FatigueDetector class (classic)
		│   ├── ear_mar.py
		│   ├── head_pose.py
		│   └── utils.py
		├── ai/
		│   ├── infer_eye.py
		│   ├── infer_yawn.py
		│   ├── infer_drowsy.py
		│   ├── preprocess_cew.py
		│   ├── preprocess_fyp.py
		│   ├── preprocess_yawdd_yawn.py
		│   ├── preprocess_drowsy.py
		│   ├── train_eye.py
		│   ├── train_yawn.py
		│   └── train_drowsy.py
		└── models/
				└── mobilenet_fatigue.py
```

> Note: dataset folders are not included in the repository; you must place them under `data/` as described above.

## 🛠️ Technologies Used
- **Language:** Python 3.8+
- **Computer vision:** OpenCV, MediaPipe
- **Deep learning:** PyTorch, Torchvision (MobileNet-based models)
- **Numerical tools:** NumPy, pandas

---

## 📦 Installation

### 1. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> On some systems you may want to install a specific build of `torch` (CPU or GPU) following the official PyTorch instructions.

### 3. Prepare datasets (optional, for training)
- Place original datasets under `data/` (YAWDD, CEW, FYP, etc.).
- Run the preprocessing scripts in `src/ai/` to generate processed datasets:
	- `preprocess_cew.py`, `preprocess_fyp.py`, `preprocess_yawdd_yawn.py`, `preprocess_drowsy.py`.
- Train models (if you don't already have `.pt` files in `models/`):
	- `train_eye.py`, `train_yawn.py`, `train_drowsy.py`.

For simply **running the demos**, you only need the trained model files:

- `models/eye_cnn.pt`
- `models/yawn_cnn.pt`
- `models/drowsy_cnn.pt`

---

## 🚀 Usage

### 1. Classic mode (rule-based)
Runs the classic fatigue detector based on EAR, MAR and head pose.

```bash
python main.py
```

This will open a window with the webcam feed and overlay:
- EAR/MAR values
- head tilt information
- textual alerts when fatigue patterns are detected.

### 2. AI mode (CNN-based)
Runs the AI-based detector using the three CNN models (eyes, yawn, drowsy).

```bash
python ai_main.py
```

The window shows probability bars for:
- **Eyes Closed** (Eye CNN)
- **Yawning** (Yawn CNN)
- **Drowsy** (Global drowsiness CNN)

Press `q` to exit in both modes.

### 3. Classic vs AI comparison
- Run `main.py` to observe classic, rule-based decisions.
- Run `ai_main.py` to observe AI probabilities.

You can compare:
- When classic raises an alert vs when AI probabilities are high.
- Sensitivity to brief events (short eye closure, quick yawns).

---

# 🇪🇸 Versión en Español

## 📋 Resumen del Proyecto
Este proyecto implementa un **sistema de monitoreo de fatiga del conductor** con dos enfoques complementarios:

1. **Enfoque clásico (basado en reglas)** – Usa características manuales e umbrales:
	 - Eye Aspect Ratio (EAR) para cierre de ojos.
	 - Mouth Aspect Ratio (MAR) para bostezos.
	 - Inclinación de la cabeza (head tilt).
2. **Enfoque IA (deep learning)** – Usa redes neuronales convolucionales (CNN):
	 - CNN de ojos → probabilidad de *ojos cerrados*.
	 - CNN de bostezo → probabilidad de *bostezo*.
	 - CNN de somnolencia → probabilidad de *somnolencia global* usando el rostro completo.

El objetivo es **comparar el enfoque clásico vs IA**, utilizando la misma cámara y condiciones.

## 🎯 Objetivos
- Construir un sistema de detección de fatiga en tiempo real enfocado en conductores.
- Proveer una línea base clásica y una alternativa basada en IA.
- Permitir experimentos y análisis comparativos (classic vs IA).

## ✨ Características Principales
- Detección de rostro y landmarks en tiempo real con MediaPipe Face Mesh.
- Métricas clásicas:
	- Cálculo de EAR, MAR e inclinación de la cabeza.
	- Detección de fatiga mediante reglas y registro en CSV.
- Modelos de IA:
	- Clasificador de ojos, bostezo y somnolencia (MobileNet).
	- Barras de probabilidad en pantalla.
- Demo con cámara web, pensada para escenarios de conducción.

## 📦 Instalación

### 1. Crear y activar entorno virtual (recomendado)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Preparar datasets (opcional, para entrenamiento)
- Coloca los datasets originales dentro de `data/`.
- Ejecuta los scripts de preprocesado en `src/ai/`.
- Entrena los modelos si no tienes los archivos `.pt` en `models/`.

Para **solo ejecutar las demos**, basta con contar con los modelos entrenados:

- `models/eye_cnn.pt`
- `models/yawn_cnn.pt`
- `models/drowsy_cnn.pt`

## 🚀 Uso

### 1. Modo clásico (reglas)
Inicia el detector clásico basado en EAR, MAR y pose de la cabeza:

```bash
python main.py
```

Verás en pantalla:
- Valores de EAR/MAR.
- Información de inclinación de cabeza.
- Mensajes de alerta cuando se detectan patrones de fatiga.

### 2. Modo IA (redes neuronales)
Inicia el detector basado en IA con los tres modelos CNN:

```bash
python ai_main.py
```

En la ventana se mostrarán barras de probabilidad para:
- **Eyes Closed** (ojos cerrados).
- **Yawning** (bostezo).
- **Drowsy** (somnolencia).

Pulsa `q` para salir en ambos modos.

### 3. Comparación Classic vs IA
- Ejecuta `main.py` para ver el comportamiento clásico.
- Ejecuta `ai_main.py` para ver el comportamiento IA.

Puedes comparar:
- Cuándo el clásico lanza una alerta frente a cuándo las probabilidades de IA son altas.
- Sensibilidad a eventos cortos (cierres breves de ojos, bostezos rápidos).

---

## 👥 Autores
- **Juan David Reyes Cure**
- **Juan Esteban Monroy Castellanos**
- **Camilo Novoa Montenegro**

---

## ⚠️ Limitaciones y Consideraciones
- El rendimiento puede verse afectado por:
	- Condiciones de iluminación pobres.
	- Calidad de la cámara.
	- Obstrucciones parciales del rostro (gafas, manos, etc.).
- Los modelos de IA han sido entrenados con datasets específicos y pueden requerir reajuste para otros contextos.

Este proyecto está orientado a fines académicos y de investigación, y **no debe utilizarse como único sistema de seguridad en vehículos reales**.

---

*Last updated / Última actualización: November 2025*

