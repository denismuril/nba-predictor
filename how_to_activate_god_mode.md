# 🚀 Ativando o God Mode (v27.0)

Este guia descreve como ativar a automação completa do NBA Predictor usando a nova arquitetura baseada em Systemd.

## 1. Instalação Automática

Execute o script de instalação para configurar os serviços do Systemd e remover cron jobs antigos:

```bash
chmod +x scripts/install_systemd_timers.sh
./scripts/install_systemd_timers.sh
```

Isso irá:

1. Copiar os arquivos de serviço para o sistema (`/etc/systemd/system/`).
2. Desativar agendamentos legados (`nba-orchestrator`, etc.).
3. Ativar o novo timer `nba-god-mode.timer`.

## 2. Verificando o Status

Verifique se o timer está ativo e agendado:

```bash
systemctl list-timers --all | grep nba
```

Você deve ver `nba-god-mode.timer` listado com a próxima execução programada (Next) e quanto tempo falta (Left).

## 3. Execução Manual (Teste)

Para testar o pipeline imediatamente sem esperar o agendamento:

```bash
sudo systemctl start nba-god-mode.service
```

## 4. Monitoramento de Logs

Para acompanhar a execução em tempo real, use a ferramenta de monitoramento criada:

```bash
./scripts/monitor_god_mode.sh
```

Isso mostrará logs combinados do serviço (stdout/stderr) e do orquestrador Python.

## 5. Arquivos Importantes

* **Wrapper Script**: `scripts/run_god_mode.sh` (Define ambiente e chama Python)
* **Service Unit**: `scripts/systemd/nba-god-mode.service`
* **Timer Unit**: `scripts/systemd/nba-god-mode.timer`
* **Orchestrator**: `orchestrator.py` (Lógica principal atualizada)

## 6. CLV Tracking (NOVO)

O sistema agora verifica automaticamente o **Closing Line Value** como parte do fluxo diário.

* Conecta-se ao **Action Network** para pegar odds reais.
* Atualiza apostas pendentes no arquivo `logs/betting_performance.csv`.
* Feedback visual nos logs: `✅ CLV atualizado para X apostas`.

---
**Status**: God Mode Fully Operational ⚡
