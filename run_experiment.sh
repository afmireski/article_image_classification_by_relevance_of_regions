#!/bin/bash

# Nome do script Python
SCRIPT="main.py"

# Nome da venv
VENV=".venv"

# Nome da sessão tmux
TMUX_SESSION="experimento"

# Pasta de logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Nome do log com timestamp
LOG_FILE="$LOG_DIR/execucao_$(date +%Y%m%d_%H%M%S).log"

# Comando a ser executado dentro do tmux
CMD="source $VENV/bin/activate && python $SCRIPT > $LOG_FILE 2>&1"

# Cria uma sessão tmux e executa o comando
tmux new-session -d -s $TMUX_SESSION "$CMD"

echo "Sessão tmux '$TMUX_SESSION' criada."
echo "Script '$SCRIPT' rodando em background dentro do tmux."
echo "Logs sendo salvos em '$LOG_FILE'."

echo "Para conectar à sessão tmux e ver a execução: tmux attach -t $TMUX_SESSION"
