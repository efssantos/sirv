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

Por padrão, a inferência usa resolução 960 e confiança mínima 0,15 para ajudar a localizar veículos menores, como motos distantes. Isso exige mais processamento. Ajuste esses valores conforme a câmera e o computador:

```powershell
python .\src\vehicle_detector.py --source .\data\videos\rodovia.mp4 --imgsz 1280 --conf 0.10
```

Uma confiança menor ou uma resolução maior pode aumentar detecções incorretas ou reduzir o FPS. A classe de moto usa o modelo YOLO pré-treinado; o resultado ainda depende da qualidade e do enquadramento da imagem.

O programa seleciona automaticamente a primeira GPU NVIDIA compatível com CUDA. **Se a máquina não tiver uma GPU compatível, ou se o PyTorch não tiver CUDA disponível, ele roda na CPU automaticamente.** Para escolher manualmente, use `--device 0` (GPU) ou `--device cpu`. A aceleração por GPU requer uma GPU NVIDIA, driver compatível e uma instalação CUDA do PyTorch; confira a mensagem de inicialização para saber qual dispositivo está sendo usado.

### Ativar GPU NVIDIA no Windows

Se o programa informar que CUDA não está disponível, instale uma compilação CUDA do PyTorch no mesmo ambiente virtual usado para executar o SIRV:

1. Ative `.venv` conforme as instruções de instalação acima.
2. Abra o [seletor oficial de instalação do PyTorch](https://pytorch.org/get-started/locally/) e escolha **Windows**, **Pip**, **Python** e uma opção **CUDA** compatível com a placa e o driver. Execute o comando gerado no PowerShell, com `.venv` ainda ativado. Não escolha **CPU**.
3. Confirme que esse Python reconhece a GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA indisponível')"
```

O resultado de `torch.cuda.is_available()` precisa ser `True`. Depois, inicie o SIRV normalmente; ele selecionará a GPU automaticamente. O comando de instalação varia conforme as opções CUDA oferecidas pelo PyTorch, então use o comando atual gerado pelo seletor oficial.

O modelo `yolo11s.pt` deve ficar na raiz do projeto. O repositório já inclui esse arquivo.

## Encerrar

Pressione `q` ou `Esc` na janela do vídeo.
