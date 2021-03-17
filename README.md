# anomaly-detection-in-endoscopic-imagery
classification of anomalies in gastrointestinal tract through endoscopic imagery with deep learning pretrained models. Mainly, used famous ResNet50, VGG19, MobileNetv2 and inception model for model experiments.

# folder structure
```
anomaly-detection-in-endoscopic-imagery/
│
├── src/
│   ├─── model/
│   │   ├── tl_mobileNet.ipynb
│   │   ├── tl_ResNet50.ipynb
│   │   ├── ti_VGG19.ipynb
│   │   └── tl_inception.ipynb
│   └─── sup/
│       ├── cf_metrix.py
│       ├── evaluation.py
│       ├── support.py
│       └── test_set_eval.py
│
├── logs/
├── data/
│   ├─── kvasir-dataset/
│   ├─── kvasir-dataset-v2/
│   └─── model/
├── documents/
│
├── .gitignore
├── .dvcignore
├── README.md
└── requirements.txt
```
