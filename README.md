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
- Python 3.11+（当前开发环境使用 3.13 亦可）
- 项目根目录作为工作目录执行下文命令

## 快速开始

### 1. 启动依赖容器

```bash
docker compose up -d
```

首次启动 Postgres 时会自动执行 `scripts/init_db.sql` 建表与种子数据。

### 2. Python 虚拟环境与依赖

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 环境变量

```bash
cp .env.example .env
```

按需修改数据库、Redis、Qdrant 与推理相关变量（见下文「环境变量」）。

### 4. 启动 API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

浏览器打开 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 查看 OpenAPI。

**说明**：`uvicorn` / `celery worker` 是**常驻进程**，会一直占用当前终端；若把「起服务 + 跑脚本 + 跑 pytest」写在一条命令里且不放到后台，看起来会像**卡死**。开发时建议：

- **两个终端**：一个专门跑 `uvicorn`（或 Celery），另一个跑 `curl` / `pytest`；
- **或后台启动 API**（项目内脚本，日志在 `logs/api.log`，PID 在 `logs/api.pid`）：

```bash
./scripts/dev_api_background.sh
# 结束：./scripts/dev_api_stop.sh
```

### 5.（可选）启动 Celery Worker

```bash
celery -A worker.celery_app worker --loglevel=info
```

同样需要**单独终端**或自行 `nohup ... &` 放到后台，否则会一直阻塞在该命令。

### 6.（可选）写入 Qdrant 示例知识

需容器 `qdrant` 已运行：

```bash
python scripts/build_rag_knowledge.py
```

### 7.（可选）主动学习脚本

根据消息置信度与风险筛选样本并写入 `annotation_tasks`：

```bash
python scripts/active_learning.py
```

## HTTP 接口摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 进程存活 |
| GET | `/ready` | 数据库连通性 |
| GET | `/metrics` | Prometheus 指标 |
| POST | `/v1/chat` | 创建或延续会话，返回回复与风险等级等 |

### `POST /v1/chat` 示例

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo","message":"最近总是睡不着，感觉做什么都没意思"}'
```

请求体字段：`user_id`（必填）、`message`（必填）、`session_id`（可选，UUID）、`audio_url`（可选）。

## 环境变量

| 变量 | 说明 |
|------|------|
| `DATABASE_URL` | AsyncPG 连接串，默认指向本机 Compose 中的 Postgres |
| `REDIS_URL` | 预留，供后续缓存或任务扩展 |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | Celery 使用 |
| `QDRANT_HOST` / `QDRANT_PORT` | `build_rag_knowledge.py` 连接 Qdrant |
| `USE_MOCK_INFERENCE` | `true`（默认）时使用内置规则基线；`false` 且配置 `INFERENCE_URL` 时请求外部 `/generate` |
| `INFERENCE_URL` | 外部推理服务根 URL（例如自建 MLX/llama 网关），勿带末尾路径 |

## 测试

```bash
# 单元测试（不依赖容器）
pytest tests/test_inference.py -q

# HTTP E2E：需先 docker compose up -d，且 API 已在运行（另一终端 uvicorn，或 ./scripts/dev_api_background.sh）
pytest tests/test_e2e_api.py -v
```

自定义 API 地址：

```bash
MINDCORE_E2E_BASE_URL=http://127.0.0.1:8080 pytest tests/test_e2e_api.py -v
```

## 仓库结构（节选）

```
api/                 # FastAPI 应用与配置
services/            # 推理与领域逻辑（默认可切换远程生成服务）
worker/              # Celery 应用与任务
scripts/             # 初始化 SQL、RAG 构建、主动学习
tests/               # 单元测试与 E2E
docker-compose.yml
requirements.txt
```

## 与需求文档的关系

更完整的产品形态（Kafka、Airflow、vLLM/MLX 独立服务、K8s 等）见仓库内 `req.txt` 中的架构说明；本仓库当前实现为可本地跑通的最小闭环，便于迭代 API 与数据模型。

## 许可证

本项目以 [MIT License](LICENSE) 发布（Copyright (c) 2026 liweinan）。
