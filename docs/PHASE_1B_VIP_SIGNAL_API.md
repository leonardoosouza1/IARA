# PHASE 1B: VIP Signal API (FastAPI)
## Objective: Automated Delivery & Scaling

Este documento detalha a criação da camada de serviço para expor os sinais do Saci para o ecossistema de monetização.

---

### 1. Arquitetura da API

*   **Framework:** FastAPI (Python).
*   **Banco de Dados:** PostgreSQL (via SQLAlchemy) - Já existente no projeto.
*   **Autenticação:** JWT (JSON Web Tokens) para clientes pagantes e integradores.

---

### 2. Endpoints Principais

*   `GET /api/v1/signals/latest`: Retorna o sinal "Sniper" mais recente.
*   `GET /api/v1/signals/history`: Histórico de sinais para validação de performance no Dashboard.
*   `GET /api/v1/market-health`: Status de risco do mercado (Baseado no Sentinela).
*   `POST /api/v1/webhooks/telegram`: Disparar alerta via bot para os assinantes VIP.

---

### 3. Integração de Fluxo

1.  O **TrinityEngine** grava um sinal no DB.
2.  A **FastAPI** detecta o novo registro via polling ou SQLAlchemy Event Listeners.
3.  A API envia uma requisição para o bot de Telegram com os dados do sinal + o link para o relatório de autoridade (Frente C).

---
*Status: Aguardando Implementação Técnica*
