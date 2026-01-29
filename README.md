## TB Detector - Chest X-ray Classification

Projekt klasyfikuje zdjęcia RTG klatki piersiowej na:
Normal / Tuberculosis (TB), z możliwością użycia GPU AMD przez DirectML.

## Struktura projektu
```
project_root/
│
├── data/
│   ├── TB_Chest_Radiography_Database/
│   │   ├── Normal/           # Zdjęcia zdrowe
│   │   └── Tuberculosis/     # Zdjęcia TB
│   └── test_images/           # Zdjęcia do predykcji
│
├── models/
│   └── model.py               # Definicja TB_Detector
│
├── models/saved_models/
│   ├── tb_detector.pt         # Finalny model
│   └── best_tb_detector.pt    # Najlepszy model według walidacji
│
├── scripts/
│   ├── train.py               # Skrypt treningowy
│   └── predict.py             # Skrypt do predykcji
│
└── requirements.txt           # Wymagane pakiety
```
## Wymagania

- Python 3.10 (zalecany)
- PyTorch 2.x
- Torchvision
- Torch-DirectML (dla AMD GPU)
- numpy
- scikit-learn
- PIL / Pillow

## Instalacja wszystkich pakietów:
```pip install -r requirements.txt```

## Trening modelu

Uruchom skrypt treningowy:

python scripts/train.py


Model zostanie zapisany w:

models/saved_models/tb_detector.pt
models/saved_models/best_tb_detector.pt

## Dane treningowe

Dane treningowe do modelu należy pobrać z Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

Następnie należy przenieść je do folderu data (tak jak wskazuje na to struktura projektu)


## Skrypt używa:

- Augmentacji obrazu (resize 256x256, flip)
- Early stopping
- Scheduler ReduceLROnPlateau
- Gradient clipping
- Podział danych: 80% train / 20% validation

## Predykcja nowych obrazów

Włóż nowe zdjęcia do:

data/test_images/

Uruchom skrypt predykcyjny:

python scripts/predict.py


## Wynik:

- xray1.jpg: Tuberculosis (0.7923)
- xray2.jpg: Normal (0.1831)


Predykcja zwraca prawdopodobieństwo TB. Możesz zmienić próg decyzyjny w predict.py (domyślnie 0.5).

## Obsługa GPU AMD (DirectML)

PyTorch + DirectML pozwala trenować na GPU AMD.

Jeśli DirectML nie jest dostępny, trening i predykcja wykonują się na CPU.

Sprawdzenie urządzenia:

import torch_directml
device = torch_directml.device()
print(device)

## Screenshoty wyników oraz uczenia

- Uczenie modelu
  
<img width="511" height="196" alt="test1234" src="https://github.com/user-attachments/assets/702fcb48-598a-4e5a-af51-1dc5ad296bfe" />

- Wyniki na podstawowym CNN

<img width="370" height="384" alt="test123" src="https://github.com/user-attachments/assets/daebf390-a7e4-44fe-8b34-d5c68b13be25" />

- Wyniki na rozszerzonym CNN ResNet18-style

<img width="360" height="379" alt="test123new" src="https://github.com/user-attachments/assets/5bddd75b-5794-4d5a-bbd8-fc203a74cda3" />

