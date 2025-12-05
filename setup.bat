@echo off
SETLOCAL

:: Caminho do ambiente virtual
SET VENV_DIR=venv

echo [1/5] Criando ambiente virtual em %VENV_DIR%...
python -m venv %VENV_DIR%
IF ERRORLEVEL 1 (
    echo Erro ao criar o ambiente virtual.
    EXIT /B 1
)

echo [2/5] Ativando ambiente virtual...
CALL %VENV_DIR%\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo Erro ao ativar o ambiente virtual.
    EXIT /B 1
)

echo [3/5] Atualizando pip...
python -m pip install --upgrade pip
IF ERRORLEVEL 1 (
    echo Erro ao atualizar pip.
    EXIT /B 1
)

echo [4/5] Detectando GPU CUDA...
python -c "import torch; print('CUDA disponível' if torch.cuda.is_available() else 'CUDA não disponível')" 2>nul
IF ERRORLEVEL 1 (
    echo PyTorch não instalado ainda. Verificando GPU via nvidia-smi...
    nvidia-smi >nul 2>&1
    IF ERRORLEVEL 1 (
        echo CPU detectada. Usando requirements_cpu.txt
        SET REQUIREMENTS_FILE=requirements_cpu.txt
    ) ELSE (
        echo GPU CUDA detectada. Usando requirements_gpu.txt
        SET REQUIREMENTS_FILE=requirements_gpu.txt
    )
) ELSE (
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
    IF ERRORLEVEL 1 (
        echo CPU detectada. Usando requirements_cpu.txt
        SET REQUIREMENTS_FILE=requirements_cpu.txt
    ) ELSE (
        echo GPU CUDA detectada. Usando requirements_gpu.txt
        SET REQUIREMENTS_FILE=requirements_gpu.txt
    )
)

echo [5/5] Instalando dependências de %REQUIREMENTS_FILE%...
IF EXIST %REQUIREMENTS_FILE% (
    pip install -r %REQUIREMENTS_FILE%
    IF ERRORLEVEL 1 (
        echo Erro ao instalar dependências.
        EXIT /B 1
    )
    echo Dependências instaladas com sucesso.
) ELSE (
    echo Arquivo %REQUIREMENTS_FILE% não encontrado.
    echo Tentando instalar requirements.txt como fallback...
    IF EXIST requirements.txt (
        pip install -r requirements.txt
        IF ERRORLEVEL 1 (
            echo Erro ao instalar dependências.
            EXIT /B 1
        )
    ) ELSE (
        echo Nenhum arquivo de requisitos encontrado.
        EXIT /B 1
    )
)

echo ✅ Setup concluído com sucesso.
ENDLOCAL
