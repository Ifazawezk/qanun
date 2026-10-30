# Builds dist-desktop\qanun.exe. Run from the repo root, with web/dist already built.
# The .ico is made from docs\images\logo.png.
$ErrorActionPreference = "Stop"
$python = ".venv\Scripts\python.exe"

& $python -m pip install --quiet pyinstaller pywebview pillow
& $python -c "from PIL import Image; Image.open('docs/images/logo.png').save('desktop/qanun.ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"

& $python -m PyInstaller desktop\launcher.py `
    --name qanun --onefile --noconsole --noconfirm --clean `
    --icon "$PWD\desktop\qanun.ico" `
    --distpath dist-desktop --workpath build-desktop --specpath build-desktop `
    --paths src `
    --add-data "$PWD\data\corpus;data\corpus" `
    --add-data "$PWD\web\dist;web\dist" `
    --add-data "$PWD\src\qanun\prompts;qanun\prompts" `
    --collect-submodules uvicorn `
    --collect-submodules qanun `
    --exclude-module torch --exclude-module sentence_transformers `
    --exclude-module transformers --exclude-module scipy --exclude-module sklearn `
    --exclude-module faiss --exclude-module tkinter --exclude-module matplotlib
