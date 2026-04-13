# MindCore

心理健康场景下的 **AI 辅助对话与风险评估** 原型：同步 API（FastAPI）、PostgreSQL 会话与消息、Redis + Celery 异步任务占位、Qdrant 知识库示例脚本，以及可运行的 HTTP 级 E2E 测试。

> **声明**：本仓库为工程原型，输出不构成医疗诊断或治疗建议。涉及自伤/伤人风险时请接入合规流程与真人专业支持。

## 技术栈

| 组件 | 用途 |
|------|------|
| FastAPI + Uvicorn | HTTP API、`/metrics`（Prometheus 文本） |
| PostgreSQL 15 | 会话、消息、标注任务、模型版本、A/B 配置 |
| Redis 7 | Celery Broker / Result |
| Qdrant | RAG 向量集合 `mental_health_knowledge`（示例脚本写入） |
| Celery | 多模态等异步任务占位（`worker.tasks.process_multimodal`） |

## 环境要求

- Docker / Docker Compose
- [uv](https://docs.astral.sh/uv/)（管理 Python 版本、虚拟环境与依赖锁）
- Python 3.11+（由 `uv` 按 `requires-python` 自动选用，当前环境 3.13 亦可）
- 项目根目录作为工作目录执行下文命令

## 快速开始

### 1. 启动依赖容器

```bash
docker compose up -d
```

Compose 会一并启动 **Ollama**（本机 `11434`）。首次需拉模型可执行 `./scripts/ensure_ollama_models.sh`。首次启动 Postgres 时会自动执行 `scripts/init_db.sql` 建表与种子数据。

### 2. 安装 uv 与项目依赖

[安装 uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在项目根目录执行：

```bash
# 同步运行时依赖（生成/更新 .venv，并以 uv.lock 为准）
uv sync

# 若需要跑 pytest，一并安装可选开发依赖
uv sync --extra dev
```

依赖声明在 `pyproject.toml`，锁定版本在 `uv.lock`。日常命令优先用 **`uv run …`**，无需手动 `source` 虚拟环境（fish / zsh / bash 行为一致）。

若未安装 uv、仍想用传统 venv，可自行 `python -m venv .venv` 后 `pip install -e ".[dev]"`（不推荐与本仓库文档混用）。

### 3. 环境变量

```bash
cp .env.example .env
```

按需修改数据库、Redis、Qdrant 与推理相关变量（见下文「环境变量」）。

### 4. 启动 API

```bash
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

浏览器打开 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 查看 OpenAPI。

**说明**：`uvicorn` / `celery worker` 是**常驻进程**，会一直占用当前终端；若把「起服务 + 跑脚本 + 跑 pytest」写在一条命令里且不放到后台，看起来会像**卡死**。开发时建议：

- **两个终端**：一个专门跑 `uv run uvicorn`（或 `uv run celery`），另一个跑 `curl` / `uv run pytest`；
- **或后台启动 API**（项目内脚本，日志在 `logs/api.log`，PID 在 `logs/api.pid`）：

```bash
./scripts/dev_api_background.sh
# 结束：./scripts/dev_api_stop.sh
```

### 5.（可选）启动 Celery Worker

```bash
uv run celery -A worker.celery_app worker --loglevel=info
```

同样需要**单独终端**或自行 `nohup ... &` 放到后台，否则会一直阻塞在该命令。

### 6.（可选）写入 Qdrant 示例知识

需容器 `qdrant` 已运行：

```bash
uv run python scripts/build_rag_knowledge.py
```

### 7.（可选）主动学习脚本

根据消息置信度与风险筛选样本并写入 `annotation_tasks`：

```bash
uv run python scripts/active_learning.py
```

## HTTP 接口摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 进程存活；JSON 中 `status` 为 `ok` |
| GET | `/ready` | 数据库连通性 |
| GET | `/metrics` | Prometheus 指标 |
| POST | `/v1/chat` | 创建或延续会话，返回回复与风险等级等；Ollama 不可用且未开启模板回退时为 503 |

### `POST /v1/chat` 示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo","message":"最近总是睡不着，感觉做什么都没意思"}'
```

请求体字段：`user_id`（必填）、`message`（必填）、`session_id`（可选，UUID）、`audio_url`（可选）。

## 本地 LLM、LangChain 与 Qdrant RAG（ARM Mac Studio）

- **是否还要关键字**：**风险等级**仍用关键词启发式（轻量、可解释）。**自然语言回复默认始终经 Ollama**（`OLLAMA_BASE_URL` 默认为 `http://127.0.0.1:11434`），不再提供「不配 Ollama 就直接关键词模板」的路径；仅当 `USE_TEMPLATE_FALLBACK=true` 且 Ollama 失败时，才回退模板。Ollama 不可用且未开回退时，`POST /v1/chat` 返回 **503**。
- **Ollama 和 LangChain 选哪个**：不是二选一。**Ollama** 负责在本机拉模型、推理（Apple Silicon / Metal 上体验好）。**LangChain**（或 LangGraph）是编排框架，适合做复杂链路与 Agent；本仓库先用 **`httpx` 调 Ollama HTTP API**，避免默认再绑一层框架；需要时你可在独立模块引入 LangChain 拼管道。
- **模型放容器里还是宿主机**：默认 **`docker compose up -d` 会起 Compose 内 Ollama**（`11434` 映射到本机）。在 Mac Studio 上若要用 **宿主机 Ollama（Metal）**，请先停掉 Compose 里的 `ollama` 服务或改端口，避免与本机 `11434` 冲突；Metal 路径下仍设 `OLLAMA_BASE_URL=http://127.0.0.1:11434` 即可。
- **小模型**：默认对话模型为 **`llama3.2:1b`**（`ollama pull llama3.2:1b` 或 `./scripts/ensure_ollama_models.sh`）；亦可改用 `qwen2.5:3b` 等并在 `.env` 设 `OLLAMA_CHAT_MODEL`。RAG 嵌入常用 `nomic-embed-text`（768 维）。
- **和 Qdrant 怎么对齐**：写入与查询必须用**同一嵌入模型**。推荐流程：1）`docker compose up -d` 启动 Qdrant 与 Ollama，并拉好嵌入/对话模型；2）设置 `OLLAMA_BASE_URL=http://127.0.0.1:11434`，执行 `uv run python scripts/build_rag_knowledge.py`（脚本会检测该变量，**用 Ollama 嵌入建库**，与线上一致）；3）在 `.env` 中设置 `OLLAMA_BASE_URL`、`OLLAMA_CHAT_MODEL`、`OLLAMA_EMBED_MODEL`、`QDRANT_RAG_COLLECTION=mental_health_knowledge`；4）启动 API。

### RAG 与本项目行为说明（概念）

- **「让大模型读向量库」**：实际上大模型**不会**去连 Qdrant。做法是**应用层**用向量库做检索，把命中的**文本片段**写进给模型的**提示词**（这里是 `system`），模型再基于「用户原话 + 这些片段」生成回答。这就是常见 **RAG**：检索和生成分开，模型读的是**文字上下文**，不是向量 API。
- **「回复走大模型」**：默认即走 Ollama `/api/chat`（除非走 `INFERENCE_URL` 远程分支）；返回给用户的 `reply` 来自模型生成；Qdrant 只参与拼 `system`。当前 `INFERENCE_URL` 远程分支尚未接 Qdrant，仅 **Ollama** 路径带检索。
- **「用户直接用向量库」**：当前对外只有 `POST /v1/chat`，没有把 Qdrant 暴露给终端用户，用户也不能直接查向量库。

### 风险等级、置信度与 `model_version`（实现与设计说明）

`/v1/chat` 返回并在 PostgreSQL `messages` 中落库的 **`risk_level`、`confidence`** 来自 `services/inference.py` 中的 **`_baseline_risk_and_confidence`**：在调用远程推理或 Ollama **之前**即确定——统计用户 `message` 命中预置关键词列表的个数（每个词至多计一次），再限制在 1～5；若 `risk_level` 为 1 或 5 则 `confidence` 为 `0.9`，否则为 `0.6`。二者均为**与模型输出无关的规则值**，不是分类器或 LLM 的概率。

- **`model_version`**：按代码路径写入字符串标签（例如 Ollama 成功时为 `ollama:<聊天模型名>`，远程 `INFERENCE_URL` 分支为 `remote-mlx`，模板回退时为 `v1.0-fallback` 等），并非从权重文件读取的语义化版本。
- **`inference_time_ms`**：Ollama 路径一般为整次 `infer` 的墙钟耗时（含可选 RAG 检索与 HTTP）；远程分支优先使用响应 JSON 中的 `inference_time_ms`，缺失或非正时回退为本地计时。

**为何不用向量数据库做风险**：Qdrant 在本项目中的职责是 **RAG 检索**（把语义相近的知识片段拼进提示词），与风险打分是**两条链路**。用向量库做风险需要另行设计（例如与参考语料相似度或专用向量分类），当前仓库未实现；关键词基线是**低成本、行为可预期的占位实现**，便于原型联调与落库、`annotation_tasks` 等下游逻辑。

**是否应改为模型推理**：若 `risk_level` / `confidence` 用于真实分流、告警或人工介入，更合理的方向是 **专用风险/危机意念分类模型**、经评估的 **结构化 LLM 输出**（配合校验与人工复核），而非当前启发式。关键词易漏检、误报，且与「置信度」的常规定义不一致；扩展前请明确合规与产品流程。

## 环境变量

| 变量 | 说明 |
|------|------|
| `DATABASE_URL` | AsyncPG 连接串，默认指向本机 Compose 中的 Postgres |
| `REDIS_URL` | 预留，供后续缓存或任务扩展 |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | Celery 使用 |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant 地址；`build_rag_knowledge.py` 与 RAG 检索共用 |
| `USE_MOCK_INFERENCE` | `true`（默认）时使用内置规则基线；`false` 且配置 `INFERENCE_URL` 时请求外部 `/generate` |
| `INFERENCE_URL` | 外部推理服务根 URL（例如自建 MLX/llama 网关），勿带末尾路径 |
| `OLLAMA_BASE_URL` | 未走 `INFERENCE_URL` 远程分支时，**默认** `http://127.0.0.1:11434`，用 Ollama `/api/chat` 生成回复 |
| `OLLAMA_CHAT_MODEL` | 对话模型名，默认 `llama3.2:1b`（须与 `ollama list` 中已有模型一致，否则会 404） |
| `OLLAMA_EMBED_MODEL` | 嵌入模型，默认 `nomic-embed-text`（与 RAG 检索、`build_rag_knowledge` 的 Ollama 路径一致） |
| `QDRANT_RAG_COLLECTION` | 非空则启用 RAG：用上述嵌入查 Qdrant，再把命中片段写入 system 提示 |
| `QDRANT_RAG_TOP_K` | 检索条数，默认 `3` |
| `USE_TEMPLATE_FALLBACK` | `true` 时 Ollama 失败或返回空内容则回退关键词模板；默认 `false`（生产建议保持 false） |

## 测试

一键验证 **Ollama 真回复**（需 `docker compose up -d` 起库与 Ollama，且已拉好默认模型 `llama3.2:1b`，例如 `./scripts/ensure_ollama_models.sh`）：

```bash
./scripts/smoke_chat.sh
```

```bash
# 单元测试（不依赖容器；需已 uv sync --extra dev）
uv run pytest tests/test_inference.py -q

# HTTP E2E：需先 docker compose up -d，且 API 已在运行（另一终端 uv run uvicorn，或 ./scripts/dev_api_background.sh）
# 若本机未起 Ollama，请在 .env 中设 USE_TEMPLATE_FALLBACK=true，或在启动 API 的 shell 里 export 后再起服务
uv run pytest tests/test_e2e_api.py -v
```

自定义 API 地址：

```bash
MINDCORE_E2E_BASE_URL=http://127.0.0.1:8080 uv run pytest tests/test_e2e_api.py -v
```

## 仓库结构（节选）

```
api/                 # FastAPI 应用与配置
services/            # 推理、RAG（Ollama + Qdrant）、远程 INFERENCE_URL
worker/              # Celery 应用与任务
scripts/             # 初始化 SQL、RAG 构建、主动学习
tests/               # 单元测试与 E2E
pyproject.toml       # 项目与依赖声明（uv）
uv.lock              # 依赖锁定（提交到仓库）
docker-compose.yml
```

## 与需求文档的关系

更完整的产品形态（Kafka、Airflow、vLLM/MLX 独立服务、K8s 等）见仓库内 `req.txt` 中的架构说明；本仓库当前实现为可本地跑通的最小闭环，便于迭代 API 与数据模型。

## 许可证

本项目以 [MIT License](LICENSE) 发布（Copyright (c) 2026 liweinan）。
