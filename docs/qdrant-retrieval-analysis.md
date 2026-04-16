# Qdrant 检索链路分析：从 Query Vector 到 RAG 注入

本文独立说明以下问题：

1. `vector = await ollama_embed(...)` 之后如何从 Qdrant 取回相关内容  
2. 以“晚安”为例，系统如何判定“相关”  
3. 当前 Qdrant 里实际有哪些数据  
4. 后续如何补充数据并避免检索质量下降

**建库侧（向量如何写入 Qdrant）**的完整约定（环境变量、仅 Ollama 真实嵌入、与在线查询对齐）见 **`docs/rag-embedding-and-qdrant.md` 第 2 节**；本节只保留与检索分析直接相关的摘要。

### 建库向量生成（摘要）

- `scripts/build_rag_knowledge.py` **必须**设置 **`OLLAMA_BASE_URL`**，对每条文档 `content` 调用 **`POST .../api/embeddings`**（模型名来自 **`OLLAMA_EMBED_MODEL`**，默认 `nomic-embed-text`），得到 `vectors` 后：若集合已存在则 **`delete_collection`**，再 **`create_collection`**（向量维度取首条 embedding 长度）与 **`upsert`** 写入 Qdrant；`payload.embedding_model` 与当前嵌入模型名一致。  
- 未配置 `OLLAMA_BASE_URL` 时脚本**直接退出**，不会写入任何点（避免与在线 Ollama 嵌入空间不一致的占位数据）。

---

## 1. 从 Query Vector 到检索结果的调用链

在线请求（`POST /v1/chat`）进入 `infer()` 后，会调用 `retrieve_rag_context()` 执行 RAG 检索。

```74:82:services/inference.py
        rag_block = await retrieve_rag_context(
            query=message,
            ollama_base_url=ollama_base,
            embed_model=embed_model,
            qdrant_host=cfg.qdrant_host,
            qdrant_port=cfg.qdrant_port,
            collection=collection,
            top_k=top_k,
        )
```

在 `retrieve_rag_context()` 内部，先调用 Ollama embedding 接口把用户输入编码为向量：

```46:50:services/rag.py
    async with httpx.AsyncClient(
        timeout=_build_http_timeout(cfg.ollama_embed_timeout_sec),
        trust_env=False,
    ) as http_client:
        vector = await ollama_embed(http_client, ollama_base_url, embed_model, query)
```

然后把这个 `vector` 作为 `query_vector` 发给 Qdrant 做近邻搜索：

```62:67:services/rag.py
    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
```

最后从命中点中提取 `payload.content`，拼成 `rag_block`，再注入 chat 模型的 `system` 提示。

```84:93:services/rag.py
    chunks: list[str] = []
    for hit in hits:
        payload = hit.payload or {}
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content.strip())
    if not chunks:
        return ""
```

---

## 2. “晚安”这类输入如何判定相关

当前实现的相关性判定是**向量相似度近邻**，不是关键词精确匹配：

- 用户文本（如“晚安”）先变成 embedding 向量  
- Qdrant 在集合内按距离度量返回最接近的 top-k  
- 代码直接使用命中结果中的 `payload.content`

当前建库脚本把集合配置为 `Cosine` 距离：

```74:77:scripts/build_rag_knowledge.py
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
```

### 关键行为说明

- **有 top-k，但无 score 阈值过滤**：当前代码会取回 top-k，不会因为分数低而丢弃。  
- 因此“晚安”也会命中若干条，但是否“业务上足够相关”取决于语料覆盖和分数分布。  
- 若知识库语料过少，系统会在有限候选里选“相对最近”的条目。

---

## 3. 当前 Qdrant 数据现状

当前集合由 `scripts/build_rag_knowledge.py` 的 `documents` 写入，默认是 3 条数据：

```48:69:scripts/build_rag_knowledge.py
    documents: list[dict] = [
        {
            "content": "抑郁症的核心症状包括：持续的情绪低落、兴趣丧失、精力减退，持续时间超过2周。",
            "tags": ["抑郁", "诊断标准"],
            "chunk_type": "knowledge",
            "source": "dsm5",
        },
        {
            "content": (
                "CBT认知重构：帮助来访者识别并挑战自动负性思维，例如'我一无是处'可以重构为'我有缺点，但也有优点'。"
            ),
            "tags": ["CBT", "认知重构"],
            "chunk_type": "intervention",
            "source": "cbt_manual",
        },
        {
            "content": "当用户表达自杀意念时，立即询问计划、评估风险，必要时转介紧急干预，不要承诺保密。",
            "tags": ["安全", "危机干预"],
            "chunk_type": "knowledge",
            "source": "expert",
        },
    ]
```

运行态上可见的关键信息：

- 集合名：`mental_health_knowledge`
- 向量维度：`768`
- 距离度量：`Cosine`
- 点数量：`3`

### 3.1 「维度」与「点数」分别指什么

| 术语 | 含义 | 在本仓库中的典型值 |
|------|------|---------------------|
| **维度**（`vectors.size` / 向量长度） | 每条 **向量** 有多少个浮点分量；建库时由首条 `embedding` 的长度决定，且全集合必须一致 | `nomic-embed-text` 为 **768**（与 `GET /collections/...` 里 `vectors.size` 一致） |
| **点数**（`points_count`） | 集合里有多少个 **Point**（一条 Point = 一个 `id` + 一份 `vector` + 可选 `payload`） | 当前示例脚本写入 **3** 条，故 `points_count=3` |

维度决定的是「向量空间有几维」；点数决定的是「库里存了几条可检索记录」。二者无关：可以 768 维但只有 3 个点，也可以上百万点仍是 768 维。

### 3.2 `points/scroll` 返回如何读（结合你的 curl 输出）

请求体里 **`with_vector:false`** 时，响应里每个点通常只有 **`id`** 与 **`payload`**，**不包含** `vector` 数组（向量仍在库里，只是本次不返回）。

你这次返回的 `result.points` 长度为 **3**，与 `points_count=3` 一致；三条 `payload` 分别对应建库脚本里的三条 `content`（顺序不一定与脚本 `documents` 数组顺序相同，以 `id` 为准）：

- `source=cbt_manual`：CBT 认知重构片段  
- `source=expert`：危机干预安全片段  
- `source=dsm5`：抑郁核心症状片段  

`payload.embedding_model` 均为 **`nomic-embed-text`**，表示这些点是用该嵌入模型生成的向量写入的，与在线 RAG 查询侧应使用同一模型名。

若要同一次响应里带上向量，把请求改为 **`"with_vector": true`** 即可（响应体积会明显变大）。

---

## 4. 后续补充数据的方式与注意事项

### 4.1 当前脚本行为（全量重建）

脚本在写入前若检测到集合已存在会先 **`delete_collection`**，再 **`create_collection`** 并 **`upsert`**，等价于对默认集合做一次「清空后重写」：

- 适合“全量重刷”
- 不适合“在线小批量追加”

### 4.2 补充策略建议

1. **扩充语料覆盖面**：增加问候、睡眠困扰、情绪低落、危机干预、求助路径等常见表达。  
2. **保持嵌入模型一致**：建库与在线查询都使用同一 embedding 模型（当前为 `nomic-embed-text`）。  
3. **优先结构化 payload**：保留 `chunk_type/source/tags/content/embedding_model`，便于后续筛选与审计。  
4. **按需求选择写入模式**：  
   - 全量替换：保留 `recreate_collection`  
   - 增量追加：改为仅 `upsert` 新点（不重建）

### 4.3 质量风险

- 若集合语料太少（当前仅 3 条），对泛化 query（如“晚安”）的命中解释力弱。  
- 若建库模型和查询模型不一致，会导致相似度排序失真。  
- 若长期不做阈值/重排策略，可能出现“有命中但不够贴题”的注入内容。  
- 若集合仍由旧版「未配置 Ollama 即写入伪向量」的脚本生成，与当前线上嵌入不在同一语义空间，需用当前 `build_rag_knowledge.py` 在设置好 **`OLLAMA_BASE_URL`** 的前提下重建集合。

---

## 5. 一句话结论

当前实现是标准的“Query Embedding -> Qdrant Top-K -> payload.content 注入 system prompt”链路；  
“相关”由向量近邻定义，当前库仅 3 条示例数据，后续应优先扩充语料并保持模型一致性。
