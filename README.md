# SIRV — Sistema de Identificação e Rastreamento de Veículos

Aplicação em Python que detecta e rastreia veículos em vídeos, câmeras locais ou streams usando YOLO e ByteTrack.

## Requisitos

- Python 3.11 ou superior
- Windows para seleção de vídeo pela janela gráfica; em outros sistemas, informe a fonte pela linha de comando

## Instalação no Windows (PowerShell)

Na pasta do projeto, crie e ative o ambiente virtual e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Executar

Para abrir o seletor de vídeo:

```powershell
python .\src\vehicle_detector.py
```

Também é possível indicar um arquivo, câmera ou stream:

```powershell
python .\src\vehicle_detector.py --source .\data\videos\rodovia.mp4
python .\src\vehicle_detector.py --source 0
python .\src\vehicle_detector.py --source "rtsp://endereco-do-stream"
```

O modelo `yolo11s.pt` deve ficar na raiz do projeto. O repositório já inclui esse arquivo.

## Encerrar

Pressione `q` ou `Esc` na janela do vídeo.
