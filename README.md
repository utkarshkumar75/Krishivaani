\# Krishi Vaani 🌱



AI-powered agricultural assistance prototype developed for \*\*Smart India Hackathon 2026 (PS26131)\*\*.



\## 🔗 Live Demo



\[\*\*Open Krishi Vaani →\*\*](https://prakhar4844.github.io/New-Start/)



\## Overview



Krishi Vaani is an agricultural assistance platform designed for early detection of crop diseases and pests, along with a mechanism for farmers to request assistance from agricultural officers.



The broader concept is intended to support multiple crops. For the current prototype, the AI disease-detection pipeline has been implemented and evaluated on \*\*tomato leaves\*\* to validate the core computer-vision system.



\## Current Prototype



\- Tomato leaf disease and pest classification

\- 10-class image classification

\- MobileNetV2-based deep learning model

\- Confidence-based prediction

\- Disease and severity information

\- Farmer assistance / agricultural officer request

\- Scan history

\- Report generation

\- Weather information in the prototype interface



\## AI/ML



| Component | Details |

|---|---|

| Dataset | PlantVillage-derived tomato dataset |

| Images | \~14,529 |

| Classes | 10 |

| Model | MobileNetV2 |

| Framework | TensorFlow / Keras |

| Task | Image classification |

| Input Size | 224 × 224 |



\## My Contribution



\- Dataset preparation and preprocessing

\- AI/ML model development

\- MobileNetV2 training

\- Model evaluation

\- Error analysis

\- Disease-classification integration



\## Repository Structure



```text

Krishivaani/

├── AI\_HANDOFF/

│   ├── class\_names.json

│   ├── predict.py

│   └── tomato\_mobilenetv2\_v2\_93\_43.keras

│

└── .gitignore

