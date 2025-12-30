# Image Classification by Relevance of Regions Using Machine Learning


## Setup
Para clonar e executar o projeto, siga as seguintes instruções

### Opção 1: Setup tradicional com pip

```bash
# Clone o projeto
git clone git@github.com:afmireski/tcc_image_classification_by_relevance_of_regions.git 
## or
git clone https://github.com/afmireski/tcc_image_classification_by_relevance_of_regions.git

# Configure a venv do projeto
python -m venv .venv    

# Ative a venv
source ./.venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Selecione a versão do python da venv como kernel do main.ipynb
```

### Opção 2: Setup com Python uv (Recomendado)

```bash
# Clone o projeto
git clone git@github.com:afmireski/tcc_image_classification_by_relevance_of_regions.git 
## or
git clone https://github.com/afmireski/tcc_image_classification_by_relevance_of_regions.git

# Entre no diretório do projeto
cd tcc_image_classification_by_relevance_of_regions

# Sincronize as dependências (isso criará automaticamente o ambiente virtual)
uv sync

# Ative o ambiente virtual
source .venv/bin/activate

# Selecione a versão do python da venv como kernel do main.ipynb
```

> **Nota:** O Python uv é uma ferramenta moderna e mais rápida para gerenciamento de dependências Python. Para instalá-lo, visite: https://docs.astral.sh/uv/getting-started/installation/

## Executando com TMUX

> Guia rápido para rodar experimentos longos (Machine Learning, IA, etc.) em servidores remotos com segurança, sem perder o progresso.

---

### 🚀 **Básico**

| Ação | Comando |
|------|----------|
| Criar nova sessão | `tmux new -s nome_sessao` |
| Entrar em sessão existente | `tmux attach -t nome_sessao` |
| Listar sessões | `tmux ls` |
| Sair da sessão **sem encerrá-la** | `Ctrl + B`, depois `D` *(de detach)* |
| Encerrar a sessão (dentro dela) | `exit` ou `Ctrl + D` |
| Matar sessão específica | `tmux kill-session -t nome_sessao` |
| Matar todas as sessões | `tmux kill-server` |

---

### 🪄 **Dentro do tmux**

Todos os comandos começam com **`Ctrl + B`**, depois o atalho.

| Ação | Teclas |
|------|--------|
| Mostrar todos os atalhos | `Ctrl + B`, depois `?` |
| Criar nova janela | `Ctrl + B`, depois `C` |
| Alternar entre janelas | `Ctrl + B`, depois número (ex: `Ctrl+B`, `0`) |
| Renomear janela | `Ctrl + B`, depois `,` |
| Fechar janela atual | `exit` ou `Ctrl + D` |
| Dividir painel **verticalmente** | `Ctrl + B`, depois `%` |
| Dividir painel **horizontalmente** | `Ctrl + B`, depois `"` |
| Alternar entre painéis | `Ctrl + B`, depois seta (↑ ↓ ← →)` |
| Redimensionar painel | `Ctrl + B`, depois `Alt` + setas |
| Fechar painel atual | `exit` |
| Sincronizar comandos entre painéis | `Ctrl + B`, depois `:` → `setw synchronize-panes on` *(off para desativar)* |

---

### 🧩 **Gerenciamento avançado**

| Ação | Comando |
|------|----------|
| Criar sessão e já rodar comando | `tmux new -d -s nome_sessao "comando"` |
| Reanexar à última sessão usada | `tmux attach` |
| Renomear sessão | `tmux rename-session -t antigo_nome novo_nome` |
| Ver detalhes de uma sessão | `tmux info -t nome_sessao` |
| Detach remoto (se estiver preso a outro terminal) | `tmux detach -s nome_sessao` |

---

### 🪵 **Logs e persistência**

| Ação | Comando |
|------|----------|
| Entrar no modo de rolagem | `Ctrl + B`, depois `[` (rola com setas, sai com `q`) |
| Copiar texto | `Ctrl + B`, depois `[` → selecione com `Espaço`, cole com `Ctrl + B`, `]` |
| Salvar histórico num arquivo | `tmux capture-pane -S - -p > saida.txt` |

---

### ⚙️ **Atalhos úteis (resumo rápido)**

| Ação | Teclas |
|------|--------|
| Novo painel vertical | `Ctrl + B` → `%` |
| Novo painel horizontal | `Ctrl + B` → `"` |
| Alternar painel | `Ctrl + B` → setas |
| Novo terminal (janela) | `Ctrl + B` → `C` |
| Detach (sair sem parar) | `Ctrl + B` → `D` |
| Fechar painel/janela | `exit` |
| Ver sessões | `tmux ls` |
| Entrar na sessão | `tmux attach -t nome` |

---

### 💡 **Fluxo típico para experimentos**

```bash
# 1. Criar sessão tmux e rodar experimento
tmux new -s experimento

# 2. Dentro da sessão:
source .venv/bin/activate
python main.py > logs/experimento_$(date +%Y%m%d_%H%M%S).log 2>&1

# 3. Sair sem encerrar:
Ctrl + B, depois D

# 4. Voltar depois:
tmux attach -t experimento

# 5. Encerrar quando terminar:
exit
