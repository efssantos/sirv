# Conectar câmeras

O SIRV aceita vídeos, webcams conectadas ao computador e streams de rede compatíveis com OpenCV/FFmpeg.

## Webcam USB

No PowerShell, inicie com o índice da câmera (geralmente `0` para a primeira):

```powershell
.\venv\Scripts\python.exe src\vehicle_detector.py --source 0
```

Se houver várias câmeras, tente `1`, `2` e assim por diante.

## Câmera IP ou DVR/NVR

Use a URL RTSP fornecida pelo fabricante da câmera, do DVR ou do NVR:

```powershell
.\venv\Scripts\python.exe src\vehicle_detector.py --source "rtsp://USUARIO:SENHA@IP:554/CAMINHO_DO_STREAM"
```

O caminho depois do endereço e da porta varia por fabricante. Configure a câmera para disponibilizar um stream H.264/H.265 acessível pela máquina que executa o SIRV. Para HTTP/MJPEG, também é possível informar a URL correspondente.

Não publique URLs que contenham usuário e senha. Evite salvar credenciais em arquivos versionados.

## Arquivo de vídeo

É possível continuar usando o seletor de arquivos ou passar o caminho diretamente:

```powershell
.\venv\Scripts\python.exe src\vehicle_detector.py --source "data\videos\rodovia.mp4"
```

Pressione `Q` ou `Esc` para encerrar. Streams ao vivo tentam reconectar quando o sinal cai.

O detector usa as classes de veículos do modelo YOLO incluído: carro, moto, ônibus e caminhão. A precisão depende da posição, resolução, iluminação, distância e configuração da câmera. Antes do uso operacional, ajuste o enquadramento e confira os resultados com gravações da própria rodovia.
