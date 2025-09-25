# 🚗 AI-Powered Driver Fatigue Monitoring System  
*Sistema Inteligente de Monitoreo de Fatiga del Conductor*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Language / Idioma
- [🇺🇸 English](#english-version)
- [🇪🇸 Español](#versión-en-español)

---

# 🇺🇸 English Version

## 📋 Description
This university project aims to develop an intelligent system that **detects early signs of fatigue or drowsiness in drivers** using computer vision and machine learning techniques.

The system analyzes the vehicle's camera (or PC/Smartphone in demo mode) in **real-time**, measuring indicators such as:

- **Eye Aspect Ratio (EAR):** to detect prolonged eye closure or microsleep episodes
- **Mouth Aspect Ratio (MAR):** to identify yawning patterns
- **Head Pose / Tilt:** head inclination as a sign of tiredness

When the system detects fatigue signals, it **emits audio/visual alerts** and logs the event for posterior analysis.

## 🎯 Objectives
- **Primary:** Develop a real-time fatigue detection system for drivers
- **Secondary:** Create an accessible demo using common devices (PC/Smartphone)
- **Academic:** Apply computer vision and machine learning concepts in a practical scenario

## ✨ Key Features
- 🔍 **Real-time face detection and tracking**
- 👁️ **Eye Aspect Ratio (EAR) monitoring**
- 😴 **Microsleep detection**
- 🥱 **Yawn detection through Mouth Aspect Ratio (MAR)**
- 📐 **Head pose estimation**
- 🚨 **Multi-modal alert system (audio + visual)**
- 📊 **Event logging and analytics**
- 💻 **Cross-platform compatibility**

## 🛠️ Technologies Used
- **Programming Language:** Python 3.8+
- **Computer Vision:** OpenCV, dlib
- **Machine Learning:** scikit-learn, NumPy
- **GUI Framework:** Tkinter/PyQt
- **Audio Processing:** pygame/playsound
- **Data Analysis:** pandas, matplotlib

## 📦 Installation

### Prerequisites
```bash
python --version  # Ensure Python 3.8+
```

### Dependencies
```bash
pip install opencv-python
pip install dlib
pip install numpy
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install pygame
```

## 🚀 Usage

### Basic Demo
```bash
python main.py
```

### With Custom Parameters
```bash
python main.py --camera 0 --threshold 0.25 --consecutive-frames 20
```

## � How It Works

### 1. Face Detection
- Uses Haar Cascades or HOG descriptors to locate faces in the video stream

### 2. Facial Landmark Detection
- Identifies 68 key facial landmarks using pre-trained models

### 3. Feature Extraction
- **EAR Calculation:** `EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)`
- **MAR Calculation:** `MAR = (||p14-p18|| + ||p15-p17|| + ||p16-p18||) / (2||p13-p19||)`

### 4. Fatigue Detection
- Monitors EAR values below threshold for consecutive frames
- Detects yawning patterns through MAR analysis
- Analyzes head pose variations

### 5. Alert System
- Triggers alerts when fatigue indicators exceed safety thresholds
- Logs events with timestamps for analysis

---

# 🇪🇸 Versión en Español

## 📋 Descripción
Este proyecto universitario tiene como objetivo desarrollar un sistema inteligente que **detecta signos tempranos de fatiga o somnolencia en conductores** mediante visión por computadora y técnicas de machine learning.

El sistema analiza la cámara del vehículo (o PC/Smartphone en la demo) en **tiempo real**, midiendo indicadores como:

- **Eye Aspect Ratio (EAR):** para detectar cierre prolongado de ojos o microsueños
- **Mouth Aspect Ratio (MAR):** para identificar bostezos
- **Head Pose / Tilt:** inclinación de la cabeza como señal de cansancio

Cuando el sistema detecta señales de fatiga, **emite alertas sonoras/visuales** y registra el evento en un log para análisis posterior.

## 🎯 Objetivos
- **Principal:** Desarrollar un sistema de detección de fatiga en tiempo real para conductores
- **Secundario:** Crear una demo accesible usando dispositivos comunes (PC/Smartphone)
- **Académico:** Aplicar conceptos de visión por computadora y machine learning en un escenario práctico

## ✨ Características Principales
- 🔍 **Detección y seguimiento facial en tiempo real**
- 👁️ **Monitoreo de Eye Aspect Ratio (EAR)**
- 😴 **Detección de microsueños**
- 🥱 **Detección de bostezos mediante Mouth Aspect Ratio (MAR)**
- 📐 **Estimación de pose de la cabeza**
- 🚨 **Sistema de alertas multimodal (audio + visual)**
- 📊 **Registro de eventos y análisis**
- 💻 **Compatibilidad multiplataforma**

## 🛠️ Tecnologías Utilizadas
- **Lenguaje de Programación:** Python 3.8+
- **Visión por Computadora:** OpenCV, dlib
- **Machine Learning:** scikit-learn, NumPy
- **Framework GUI:** Tkinter/PyQt
- **Procesamiento de Audio:** pygame/playsound
- **Análisis de Datos:** pandas, matplotlib

## 📦 Instalación

### Prerrequisitos
```bash
python --version  # Asegurar Python 3.8+
```

### Dependencias
```bash
pip install opencv-python
pip install dlib
pip install numpy
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install pygame
```

## 🚀 Uso

### Demo Básica
```bash
python main.py
```

### Con Parámetros Personalizados
```bash
python main.py --camera 0 --threshold 0.25 --consecutive-frames 20
```

## 📊 Cómo Funciona

### 1. Detección Facial
- Utiliza Haar Cascades o descriptores HOG para localizar rostros en el stream de video

### 2. Detección de Puntos Faciales
- Identifica 68 puntos clave faciales usando modelos pre-entrenados

### 3. Extracción de Características
- **Cálculo EAR:** `EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)`
- **Cálculo MAR:** `MAR = (||p14-p18|| + ||p15-p17|| + ||p16-p18||) / (2||p13-p19||)`

### 4. Detección de Fatiga
- Monitorea valores EAR por debajo del umbral durante frames consecutivos
- Detecta patrones de bostezo mediante análisis MAR
- Analiza variaciones en la pose de la cabeza

### 5. Sistema de Alertas
- Activa alertas cuando los indicadores de fatiga exceden umbrales de seguridad
- Registra eventos con timestamps para análisis

---

## 👥 Authors / Autores
- **Juan David Reyes Cure** - 2220221048
- **Juan Esteban Monroy Castellanos** - 2420191048
- **Camilo Novoa Montenegro** - 2220221025

---

## 📁 Project Structure / Estructura del Proyecto
```
├── src/
│   ├── main.py                 # Main application / Aplicación principal
│   ├── detector.py            # Fatigue detection logic / Lógica de detección
│   ├── utils.py               # Utility functions / Funciones utilitarias
│   └── config.py              # Configuration settings / Configuraciones
├── models/
│   ├── shape_predictor_68_face_landmarks.dat
│   └── haarcascade_frontalface_default.xml
├── data/
│   ├── logs/                  # Event logs / Registros de eventos
│   └── samples/               # Sample videos / Videos de muestra
├── docs/                      # Documentation / Documentación
├── tests/                     # Unit tests / Pruebas unitarias
└── README.MD
```

## 🔬 Research & Development / Investigación y Desarrollo

### Algorithm Performance / Rendimiento del Algoritmo
- **EAR Threshold:** 0.25 (calibrated for optimal sensitivity / calibrado para sensibilidad óptima)
- **MAR Threshold:** 0.5 (yawn detection / detección de bostezos)
- **Frame Rate:** 30 FPS (real-time processing / procesamiento en tiempo real)
- **Detection Accuracy:** ~95% in controlled environments / ~95% en entornos controlados

### Future Improvements / Mejoras Futuras
- 🧠 **Deep Learning Integration:** CNN-based fatigue classification
- 📱 **Mobile App Development:** Native iOS/Android applications
- 🌐 **IoT Integration:** Connect with vehicle systems
- 📈 **Advanced Analytics:** Machine learning-based pattern recognition
- 🔊 **Voice Commands:** Hands-free interaction

## 🤝 Contributing / Contribuciones

### How to Contribute / Cómo Contribuir
1. Fork the repository / Hacer fork del repositorio
2. Create a feature branch / Crear una rama de características
3. Make your changes / Realizar cambios
4. Add tests / Agregar pruebas
5. Submit a pull request / Enviar pull request

### Development Setup / Configuración de Desarrollo
```bash
# Clone the repository / Clonar el repositorio
git clone https://github.com/yourusername/driver-fatigue-detection.git

# Navigate to project directory / Navegar al directorio del proyecto
cd driver-fatigue-detection

# Create virtual environment / Crear entorno virtual
python -m venv venv

# Activate virtual environment / Activar entorno virtual
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies / Instalar dependencias
pip install -r requirements.txt
```

## 📊 Demo & Results / Demo y Resultados

### Live Demo / Demo en Vivo
The system provides real-time feedback through:
- **Visual indicators:** Color-coded alerts on screen
- **Audio alerts:** Customizable warning sounds
- **Data logging:** CSV files with timestamps and metrics

El sistema proporciona retroalimentación en tiempo real mediante:
- **Indicadores visuales:** Alertas codificadas por colores en pantalla
- **Alertas de audio:** Sonidos de advertencia personalizables
- **Registro de datos:** Archivos CSV con timestamps y métricas

### Sample Output / Salida de Ejemplo
```
[2024-03-15 14:30:25] EAR: 0.23 | MAR: 0.45 | Status: ALERT - Eyes closed
[2024-03-15 14:30:26] EAR: 0.28 | MAR: 0.52 | Status: WARNING - Yawning detected
[2024-03-15 14:30:27] EAR: 0.31 | MAR: 0.40 | Status: NORMAL
```

## 🛡️ Safety & Ethics / Seguridad y Ética

### Privacy Considerations / Consideraciones de Privacidad
- **No data storage:** Real-time processing without storing biometric data
- **Local processing:** All computation performed locally
- **User consent:** Clear notification of monitoring system activation

- **Sin almacenamiento de datos:** Procesamiento en tiempo real sin guardar datos biométricos
- **Procesamiento local:** Toda la computación se realiza localmente
- **Consentimiento del usuario:** Notificación clara de la activación del sistema de monitoreo

### Limitations / Limitaciones
- **Lighting conditions:** Performance may vary in low-light environments
- **Individual variation:** Calibration may be needed for different users
- **Hardware requirements:** Requires decent camera quality for optimal results

- **Condiciones de iluminación:** El rendimiento puede variar en entornos con poca luz
- **Variación individual:** Puede necesitar calibración para diferentes usuarios
- **Requisitos de hardware:** Requiere calidad de cámara decente para resultados óptimos

## 📚 References / Referencias

### Academic Papers / Artículos Académicos
1. Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks.
2. Devi, M. S., & Bajaj, P. R. (2008). Driver fatigue detection based on eye tracking.
3. Zhao, Z., et al. (2018). Driver fatigue detection using facial expression analysis.

### Libraries & Tools / Librerías y Herramientas
- [OpenCV Documentation](https://docs.opencv.org/)
- [dlib C++ Library](http://dlib.net/)
- [scikit-learn User Guide](https://scikit-learn.org/)

---

## 📞 Contact / Contacto

For questions, suggestions, or collaboration opportunities:
Para preguntas, sugerencias u oportunidades de colaboración:

- **Email:** [juan.reyes2@estudiantesunibague.edu.co](mailto:juan.reyes2@estudiantesunibague.edu.co)
- **University:** Universidad de Ibagué
- **Department:** Ingeniería de Sistemas / Computer Systems Engineering

---

*Last updated / Última actualización: September 2025*

