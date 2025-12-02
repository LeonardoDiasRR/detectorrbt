@echo off
SETLOCAL

:: Caminho do ambiente virtual
SET VENV_DIR=.venv

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

echo [4/5] Instalando dependências do requirements.txt...
IF EXIST requirements.txt (
    pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        echo Erro ao instalar dependências.
        EXIT /B 1
    )
    echo [5/5] Dependências instaladas com sucesso.
) ELSE (
    echo Arquivo requirements.txt não encontrado. Nenhuma dependência instalada.
)

echo ✅ Setup concluído com sucesso.
ENDLOCAL
