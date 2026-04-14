# MindCore 后续改进方向

面向本仓库当前实现（FastAPI、`services/inference.py` 直连 Ollama、PostgreSQL 会话与消息、`worker/` 中 Celery 占位任务、Qdrant RAG）的演进说明；**第 6 节**补充多模态接入思路。不涉及医疗合规承诺；上线前须单独完成产品与法务评审。

---

## 1. Celery + Redis 任务管理

### 现状

- **Broker / Backend**：`api.config` 中的 `CELERY_BROKER_URL`、`CELERY_RESULT_BACKEND` 指向 Redis（与 `docker-compose.yml` 中 `redis` 服务一致）。
- **应用入口**：`worker/celery_app.py` 创建 `Celery("mindcore", …)`，序列化为 JSON。
- **示例任务**：`worker/tasks.process_multimodal` 为占位（模拟耗时、可重试），用于演示「多模态等重型逻辑」与 API 解耦。

### 建议用法

| 能力 | 说明 |
|------|------|
| **从 API 投递** | 在 `POST /v1/chat` 或专用管理接口中 `process_multimodal.delay(...)`，立即返回 `task_id`；客户端轮询 Redis/backend 或另增 `GET /v1/tasks/{id}` 查状态。 |
| **队列拆分** | 为不同 SLA 建队列（如 `celery -Q audio,default`），重任务进 `audio`，避免阻塞轻量任务。 |
| **幂等与去重** | 以 `message_id` 或业务 id 作任务参数，在任务内先查库是否已处理，避免重复转写/重复扣费。 |
| **Beat 定时任务** | 增加 `celery beat` 进程：例如周期性跑 `scripts/active_learning.py` 同类逻辑、清理过期结果、对账。 |
| **监控** | 可选 Flower 或 Prometheus exporter，与现有 `/metrics` 并存。 |

### 与同步推理的边界

- **在线对话**：仍宜在请求路径内完成低延迟推理（当前 `infer()`）；Celery 适合 **异步后处理**（音频转写、摘要、批量导出、离线评测、告警通知）。
- **Redis 其它用途**：除 Celery 外，可对热点配置或会话级限流使用同一 Redis（注意 key 前缀与 DB 分片，避免与 broker 键冲突）。

---

## 2. Kubernetes 算力管理与调用

### 目标

在集群中分别伸缩 **无状态 API**、**Celery Worker**、**有状态依赖**（Postgres、Redis、Qdrant 可用托管或 Operator），并对 **GPU 推理** 做节点与资源隔离。

### 典型拆分

| 工作负载 | 建议 |
|----------|------|
| **API Deployment** | `Deployment` + `Service`，水平扩缩（`replicas` 或 HPA，指标可用 CPU/自定义 QPS）。 |
| **Celery Worker** | 独立 `Deployment`，`replicas` 按队列积压调整；GPU 任务与 CPU 任务分不同 Deployment，避免 GPU 节点被纯 IO 任务占满。 |
| **Ollama / 自研推理** | 若容器内跑 Ollama：使用带 GPU 的 `DaemonSet` 或专用 `Deployment`，`resources.limits` 声明 `nvidia.com/gpu`；或通过 **集群外推理网关**（专用推理集群），API 仅配置 `INFERENCE_URL` / `OLLAMA_BASE_URL` 指向 Service 或 Ingress。 |
| **配置与密钥** | `DATABASE_URL`、`CELERY_*`、`QDRANT_*` 放入 `Secret`；非敏感默认用 `ConfigMap`。 |

### 调度与弹性

- **HPA**：API 与无 GPU Worker 可按 CPU/内存或自定义指标扩缩。
- **GPU 节点**：使用节点池 + **taints/tolerations**，仅推理 Pod 调度到 GPU 节点。
- **探针**：API 沿用 `/health`、`/ready`；Worker 进程可用 Celery 健康检查脚本或 sidecar。

### 与本项目代码的衔接

- 镜像沿用根目录 `Dockerfile`；Compose 中的环境变量迁移为 K8s 清单中的 env。
- 若推理在集群内：`OLLAMA_BASE_URL` 指向集群内 `Service` DNS（如 `http://ollama.default.svc.cluster.local:11434`）。

---

## 3. 引入 LangChain

### 动机

当前 `services/inference.py` 使用 **httpx** 调用 Ollama `/api/chat`，`services/rag.py` 直接调 Qdrant Client。逻辑清晰、依赖少。当需要 **多步链**（查询改写、多轮检索、工具调用、路由）时，LangChain（或 **LangGraph** 做有状态流）可减少样板代码。

### 建议集成方式

- **独立模块**：新增例如 `services/chains/`（或 `services/langchain_pipeline.py`），不一次性替换全部 `infer()`，先封装「RAG + chat」为 `Runnable`，便于 A/B。
- **与 Ollama 对齐**：使用官方 `ChatOllama` / `OllamaEmbeddings`，保证与现有 `OLLAMA_CHAT_MODEL`、`OLLAMA_EMBED_MODEL` 一致；向量维与 Qdrant collection 与 `scripts/build_rag_knowledge.py` 一致。
- **检索**：`langchain_community.vectorstores.Qdrant` + 与现网相同的 `collection`、host/port；或保留 `retrieve_rag_context`，仅在链中作为一步调用。
- **依赖**：在 `pyproject.toml` 增加 `langchain`、`langchain-community`（及按需的 `langgraph`），版本与 Python 3.11+ 锁定。

### 注意

- LangChain 升级较快，宜 **锁版本** 并单测覆盖主路径。
- 风险打分仍建议独立模块（见第 4 节），避免把「危机等级」完全绑在单一 LLM 链上而不加校验。

---

## 4. 完善风险评估模块

### 现状摘要

- `services/inference.py` 中 `RISK_KEYWORDS` + `_baseline_risk_and_confidence`：子串计数 → `risk_level` 1～5，固定规则 `confidence`；在调用模型 **之前** 计算，与 RAG/模型输出无关。

### 演进路径（可组合）

| 层级 | 内容 |
|------|------|
| **配置化** | 将关键词、权重、生效区间写入 PostgreSQL（或配置中心），启动或定时加载到内存；支持按租户/场景切换。 |
| **语义辅助** | 单独 Qdrant collection 存「危机表述参考向量」，对用户句 embedding 后检索 Top-K，相似度超阈则抬高风险或触发复核；与关键词 **融合**（加权或 max），并记录命中 id 便于审计。 |
| **专用分类器** | 训练轻量二分类/多分类模型（危机意念等），服务化后 HTTP 调用；输出概率与校准后的 `confidence`。 |
| **结构化 LLM** | 使用 JSON schema / tool 调用约束模型输出 `risk_level` + `rationale`（内部用），**不得**单独作为唯一依据；与规则/分类器结果做一致性检查或投票。 |
| **闭环** | 利用已有 `annotation_tasks.ground_truth_risk` 与 `scripts/active_learning.py` 思路，持续评估关键词与模型表现，迭代词表与阈值。 |

### 数据与合规

- 落库字段已包含 `messages.risk_level`、`confidence`；扩展时增加 `risk_signals`（JSON，记录规则 id、向量命中、模型版本）便于追溯。
- 产品若涉及真实危机干预，须定义 **人工复核 SLA** 与 **误判** 处理流程，超出本仓库范围。

### 与架构文档的对应关系

- **Kubernetes 目标态下**各组件所在地盘（规则库、危机向量集合、可选分类服务、融合逻辑）见 **`docs/technical-analysis-data-rag.md` 第 6.5 节**。

---

## 5. 大模型生成参数与调参

与「给模型调 temperature」相关的改动在本仓库**有实际落点**；当前多数生成参数仍写在 `services/inference.py` 内，若要可运维、可实验，宜迁入 `api.config.Settings` / 环境变量（与 `qdrant_rag_top_k`、`ollama_chat_model` 同级管理）。

### 与第 4 节的关系

- **`risk_level` / `confidence`（关键词基线）**：在调用大模型 **之前** 即算好，**不受**生成侧 `temperature`、`max_tokens` 等影响。
- 若未来在风险评估中引入 **结构化 LLM 输出或专用分类服务**，那些调用才会成为 **新的调参对象**，与主对话生成仍宜 **分路配置**。

### 各环节可调项

| 环节 | 现状（代码侧） | 可调方向 |
|------|----------------|----------|
| **Ollama `/api/chat`（主对话）** | `options` 中仅 `temperature: 0.6` | `temperature`、`top_p`、`num_ctx`、`repeat_penalty` 等，影响风格、稳定性与长度。 |
| **远程 `INFERENCE_URL` `/generate`** | 写死 `max_tokens: 256`、`temperature: 0.7` | 与远程服务约定对齐，按需暴露为配置。 |
| **RAG 检索** | `qdrant_rag_top_k` 已在 `Settings` 中可配 | 取回片段条数；可再扩展相似度阈值、重排序等（当前未实现）。 |
| **嵌入** | `ollama_embed_model` 可配 | 换模型通常需 **重建 Qdrant 索引**（与 `scripts/build_rag_knowledge.py` 一致），属选型而非单纯旋钮。 |

### 实验与指标

- **`ab_tests` 表**（`scripts/init_db.sql`）可用于 **A/B**：不同参数组合映射到不同配置，对比人工标注、`annotation_tasks` 或离线测试集。
- 调参应有明确目标（如更短回复、更少越权表述、固定场景下的共情评分），避免无基准改数字。

---

## 6. 多模态能力建设

**含义**：在应用层将语音/视频等 **先变为可与现有链路对齐的文本（或结构化摘要）**，再进入 `infer()`、RAG 与风险规则；不必首版就换成端到端多模态大模型。

### 现状（本仓库）

- `POST /v1/chat` 已支持可选 `audio_url`，`messages` 表含 `has_audio`、`audio_url`、`has_video`、`video_url`；但 **`infer()` 仅使用 `message` 文本**，媒体 URL 不参与推理。
- `worker/tasks.process_multimodal` 为占位，可与第 1 节 Celery 队列（如 `audio`）结合。

### 产品形态

| 形态 | 说明 | 适用 |
|------|------|------|
| **同步** | 短语音：拉流/下载 → ASR → 拼文本 → 同请求内调用 `infer()` | 低延迟、实现相对简单 |
| **异步** | 长音频/视频：`202` + `task_id`，Celery 内 ASR/抽帧，结果写库或回调 | 与第 1 节「从 API 投递任务」一致 |

可组合：**短语音走同步扩展，重任务走异步**。

### 推荐管线

1. **获取媒体**：按 URL 拉取（鉴权、超时、大小上限）。
2. **语音**：ASR（本地 Whisper / 云 API 等）→ 转写文本 `transcript`。
3. **视频（可选）**：抽音轨 ASR；若需画面信息再抽帧 + 视觉 caption 或小 VLM，与 transcript 合成 **一段融合文本** 或 JSON。
4. **与用户输入融合**：例如 `transcript` 与用户 `message` 拼接为单一字符串再传入 `infer()`；纯语音场景可令 `message` 为占位或全文使用转写。
5. **下游**：融合串仍走现有 **文本嵌入 + Qdrant 检索 + `/api/chat`**；关键词风险可对 **拼接后的全文** 计算（见第 4 节扩展）。

### 与 RAG / 嵌入

- 检索查询句应来自 **融合后的用户意图文本**；继续沿用 **`OLLAMA_EMBED_MODEL`**，与建库脚本一致（见 `docs/rag-embedding-and-qdrant.md`）。
- 若未来引入「音频向量」等，需单独集合与编码器，与当前文本 RAG **分路** 设计。

### 工程落点

| 环节 | 建议 |
|------|------|
| **新模块** | 如 `services/multimodal.py`：`audio_url` → `transcript`，错误可映射为 HTTP 4xx/5xx。 |
| **API** | 在调用 `infer()` **之前**完成转写与拼接；必要时扩展分片上传、`video_url` 等。 |
| **Celery** | 实现 `process_multimodal`：按 `message_id` 写回 `transcript` 或更新 `content` 补充字段；幂等与去重见第 1 节。 |
| **数据层** | 可新增 `messages.transcript` 或 JSONB `media_metadata`；与 `scripts/init_db.sql` 迁移配套。 |
| **配置** | 在 `Settings` 中增加 ASR 端点、模型名、时长上限、同步/异步策略等。 |

### 评测与安全

- ASR：WER、领域词表（心理相关术语）；端到端用例含带音频请求。
- 风险与隐私：转写内容分级存储、留存策略、日志脱敏；危机流程仍须人工复核设计（见第 4 节）。

### 实施顺序建议

1. 语音 + ASR → 拼入 `message` 再 `infer()`（同步或短异步）。  
2. 打通 Celery 写回与任务查询。  
3. 再考虑视频画面、专用多模态模型（成本与评测复杂度更高）。

---

## 文档与代码索引

| 主题 | 位置 |
|------|------|
| Celery 应用 | `worker/celery_app.py` |
| 示例任务 | `worker/tasks.py` |
| 推理、生成参数与风险基线 | `services/inference.py` |
| RAG 检索 | `services/rag.py` |
| 可配置项（含 `qdrant_rag_top_k` 等） | `api/config.py` |
| 数据库表（含 `ab_tests`） | `scripts/init_db.sql` |
| 环境变量说明 | `README.md` |
| 风险评估目标架构（K8s 地盘与数据流） | `docs/technical-analysis-data-rag.md` §6.5 |
| RAG 与嵌入模型一致性 | `docs/rag-embedding-and-qdrant.md` |
