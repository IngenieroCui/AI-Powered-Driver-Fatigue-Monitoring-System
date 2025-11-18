# Guía rápida de ejecución

Este archivo explica **cómo ejecutar el proyecto en dos escenarios**:

1. **Desde cero**, con solo los datasets *raw* (sin datos procesados ni modelos `.pt`).
2. **Con los datos ya procesados y los modelos entrenados**.

> Todos los comandos están pensados para **Windows (PowerShell)** desde la carpeta raíz del proyecto.

---

## 0. Preparación inicial (para ambos casos)

1. Abrir una terminal en la carpeta del proyecto, por ejemplo:
   - `D:\Users\User\Downloads\vendoANovoa`

2. (Opcional pero recomendado) Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

---

## 1) Ejecutar TODO desde cero (solo con data raw)

Esta ruta es para cuando **solo tienes los datasets originales** (CEW, FYP, YAWDD, etc.) y aún **no existen**:
- las carpetas de datos procesados (`data/eyes_combined`, `data/yawdd`, `data/drowsy`, ...),
- ni los modelos de IA (`models/eye_cnn.pt`, `models/yawn_cnn.pt`, `models/drowsy_cnn.pt`).

### 1.1. Colocar los datasets raw

En la carpeta `data/` deberías tener algo similar a:

```text
data/
├── cew/          # Closed Eyes in the Wild (imágenes de ojos)
├── fyp/          # Dataset de ojos (Open_Eyes / Closed_Eyes)
├── yawdd/        # YawDD (videos / frames de conducción)
└── ...           # (Opcional) otras fuentes que quieras añadir
```

Asegúrate de que las rutas internas coincidan con lo que esperan los scripts de `src/ai/`.

### 1.2. Preprocesar datasets para IA

Desde la raíz del proyecto, ejecutar en este orden:

1. **Preprocesar CEW (ojos)**

```powershell
python src/ai/preprocess_cew.py
```

Esto generará una estructura tipo:

```text
data/
└── cew_processed/
    ├── train/
    │   ├── open/
    │   └── closed/
    └── val/
        ├── open/
        └── closed/
```

2. **Preprocesar FYP (ojos)**

```powershell
python src/ai/preprocess_fyp.py
```

Esto generará algo como:

```text
data/
└── fyp_processed/
    ├── train/
    │   ├── open/
    │   └── closed/
    └── val/
        ├── open/
        └── closed/
```

3. **Unir CEW + FYP en un solo dataset de ojos**

```powershell
python src/ai/merge_eyes_datasets.py
```

Esto debe crear:

```text
data/
└── eyes_combined/
    ├── train/
    │   ├── open/
    │   └── closed/
    └── val/
        ├── open/
        └── closed/
```

4. **Preprocesar YAWDD para bostezos (yawn)**

```powershell
python src/ai/preprocess_yawdd_yawn.py
```

Esto generará la estructura de datos de boca (bostezo vs no_bostezo), normalmente en algo como:

```text
data/
└── yawdd/
    ├── train/
    │   ├── yawn/
    │   └── no_yawn/
    └── val/
        ├── yawn/
        └── no_yawn/
```

5. **Preprocesar dataset global para somnolencia (drowsy)**

```powershell
python src/ai/preprocess_drowsy.py
```

Esto combina información de YAWDD, CEW, FYP, etc. y crea un dataset de rostro completo:

```text
data/
└── drowsy/
    ├── train/
    │   ├── alert/
    │   └── drowsy/
    └── val/
        ├── alert/
        └── drowsy/
```

### 1.3. Entrenar los modelos de IA

Con los datasets procesados listos, ahora se entrenan los tres modelos CNN.

> Nota: estos entrenamientos pueden tardar, dependiendo de tu hardware.

1. **Entrenar modelo de ojos (eye CNN)**

```powershell
python src/ai/train_eye.py
```

Esto debe guardar un archivo de pesos en:

```text
src/models/eye_cnn.pt
```

2. **Entrenar modelo de bostezo (yawn CNN)**

```powershell
python src/ai/train_yawn.py
```

Debería generar:

```text
src/models/yawn_cnn.pt
```

3. **Entrenar modelo de somnolencia global (drowsy CNN)**

```powershell
python src/ai/train_drowsy.py
```

Debería generar:

```text
src/models/drowsy_cnn.pt
```

### 1.4. Ejecutar los modos Classic y AI

Una vez entrenados los modelos, ya puedes ejecutar los dos enfoques:

1. **Modo clásico (reglas)**

```powershell
python main.py
```

2. **Modo IA (CNNs)**

```powershell
python ai_main.py
```

Pulsa `q` en la ventana de OpenCV para salir.

---

## 2) Ejecutar con data procesada y modelos ya entrenados

Este es el caso donde **ya tienes**:

- Carpetas de datos procesados (`data/eyes_combined`, `data/yawdd`, `data/drowsy`, etc.), **y/o**
- Archivos de modelos en `models/`:
  - `models/eye_cnn.pt`
  - `models/yawn_cnn.pt`
  - `models/drowsy_cnn.pt`

En este escenario solo necesitas:

1. Asegurarte de tener las dependencias instaladas (ver sección 0).
2. Verificar que los archivos `.pt` están en la carpeta `models/`.

### 2.1. Ejecutar modo clásico

No depende de los modelos entrenados, solo de la webcam y MediaPipe:

```powershell
python main.py
```

### 2.2. Ejecutar modo IA

Usa los modelos `.pt` ya entrenados:

```powershell
python ai_main.py
```

Si algún archivo `.pt` falta, los scripts de inferencia (`infer_eye.py`, `infer_yawn.py`, `infer_drowsy.py`) deberían lanzar un error indicando qué modelo falta y sugerir el script de entrenamiento correspondiente.

---

## 3) Resumen rápido

- **Primera vez (sin nada listo)**:
  1. Colocar datasets en `data/`.
  2. Ejecutar scripts de preprocesado.
  3. Entrenar `eye_cnn.pt`, `yawn_cnn.pt`, `drowsy_cnn.pt`.
  4. Ejecutar `main.py` (clásico) y `ai_main.py` (IA).

- **Con data y modelos ya listos**:
  - Solo necesitas tener el entorno configurado y ejecutar:

```powershell
python main.py
python ai_main.py
```
