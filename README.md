# NF-Atlas：纳滤文献知识智能体

NF-Atlas 是面向纳滤研究的可持久化垂直智能体。上下文问答通过硅基流动调用 `deepseek-ai/DeepSeek-V4-Pro`，默认启用、并允许用户关闭深度思考；文献扩词、结构化抽取和后台处理继续使用 `Pro/deepseek-ai/DeepSeek-V3.2`。模型不做微调，也不会把论文写入模型参数。这里的“学习文献”特指文献已持久化解析并进入知识库。

系统不只复述论文：它会比较多篇文献中的“材料—工艺—结构—实验条件—性能”链条，发现条件依赖规律、跨论文矛盾和可验证的新机理假设，并写入独立的 AI 知识发现图层。该图层与论文原始事实严格分开；未经人工审阅或实验验证的内容始终标记为 `AI综合` 或 `AI假设`，不能冒充已证实知识。

项目已实现一条端到端工程链：

```text
Streamlit Cloud 公开前端（无知识库数据）
          │ 受限 SSH 转发或 HTTPS/Cloudflare Tunnel
       服务器 FastAPI ── PostgreSQL（对话、任务、文献、证据、训练轨迹）
          │
   LangGraph 问题路由              Redis/RQ 后台任务
      ├─ Neo4j/SQL facts              ├─ DeepSeek 词表扩展
      ├─ Chroma + bge-m3              ├─ OpenAlex/Crossref/S2/Elsevier 检索
      ├─ SQL keyword                  ├─ 四级查重 + 公开网页/合法全文获取
      └─ AIInsight memory             ├─ 题名摘要文献自动补全文升级
          │                           ├─ MinerU/GROBID/PyMuPDF 解析
     DeepSeek 有证据回答              ├─ DeepSeek 结构化抽取
          │                           └─ Neo4j + Chroma 入库
   跨文献规律/矛盾/假设
   + 边界条件 + 证伪方案 ── AIInsight 图层（可人工审核）
PDF/XML/解析 JSON ── 本地目录或 MinIO
```

## 已落实的关键约束

- 输入“界面聚合”“盐湖提锂”“抗污染膜”等主题后，先合并领域内置词表与 DeepSeek 中英文扩词，再向 OpenAlex、Crossref、Semantic Scholar 和已配置的 ScienceDirect API 发起真实检索。
- 对话 Agent 会读取近期消息和滚动摘要，把“它呢”“继续找相关的”等追问改写为独立问题；图谱/向量检索后再判断知识缺口，并在用户启用时主动生成文献候选。候选仍需用户确认，不会擅自批量导入。
- 文献模块支持手动采集 50～200 篇和 Redis/RQ 持续自动采集。自动任务逐轮扩词、检索、查重、入库并定时安排下一轮，可从页面安全停止或重启。
- Worker 启动时会把 PostgreSQL 中未完成、但 Redis 已丢失执行者的导入/自动采集任务重新入队；断点恢复会跳过 `indexed` 文献，不重复下载、解析、抽取或调用模型计费。
- “后台文献处理中心”每 5 秒自动刷新，集中展示运行中、排队中、等待选择和最近完成/失败任务；每个运行任务会显示当前文献题名、DOI、批次位置、题名/摘要或全文 PDF/XML/JATS/HTML 模式、全文来源、文件保存状态、切片数、结构化事实数，以及单篇和整批两级进度。用户上传、公开网址/DOI 导入和单篇全文补充进入高优先级队列，批量采集进入普通文献队列。
- 从 OpenAlex 种子文献继续发现参考文献、被引文献和相关作者研究；跨来源候选自动合并并按题名、摘要、领域锚点与全文线索排序。
- DOI、OpenAlex ID、题名—第一作者指纹、全文 SHA-256 四级查重；前三层在用户选择前执行，SHA-256 在全文到达时执行。数据库唯一约束是并发导入的最后防线。
- 检索结果明确区分“知识库已有”和“去重后可新增”；已有文献仍可显示为相关推荐，但不能再次选择、下载、解析或计入新增数。
- 用户在当前对话中选择 0～200 篇。0 篇会直接完成任务并继续使用现有知识库。
- 系统会优先按 DOI 查询 Europe PMC 正式开放全文（JATS XML），再读取候选中的公开文章页、PDF、XML 或结构完整的 HTML 正文，并遵守站点 `robots.txt`、域名限速、文件大小和安全跳转限制；Unpaywall 与 ScienceDirect TDM API 是补充渠道。出版社页面禁止自动读取时，只要 Europe PMC 有合法开放副本，仍可在原文献记录上升级全文。
- “公开网址/DOI导入”允许直接粘贴公开文章页、PDF 地址或 DOI；系统自动识别正文、题名和 DOI。Unpaywall 只使用合法开放位置；ScienceDirect 全文只走授权 TDM API；订阅 PDF/XML 只能由用户上传并勾选合法权限确认。没有学校账号、密码或自动登录代码，也不会绕过验证码或付费墙。
- 合法全文不可用时不会丢掉候选：系统把题名和摘要作为 `metadata-only` 切片进入 Chroma/图谱，并在阅读页和回答中明确标注“仅摘要证据、无页码”。
- 阅读页可以对单篇或最多 200 篇 `metadata-only` 文献重新检查公开全文；成功后在原文献记录上替换摘要切片、事实和图谱节点，避免重复计数。
- Crossref、Semantic Scholar 等渠道按各自速率串行/并发控制，对 HTTP 429 读取 `Retry-After` 或指数退避重试；Elsevier 401/403 会立即熔断本轮该渠道，其他来源继续返回结果。
- MinerU 可用命令模板接入；未配置或失败时依次回退 GROBID、PyMuPDF。正文、表格、图注和解析结果按文献持久化。
- DeepSeek 分批抽取膜材料、膜批次、工艺、条件、溶质、性能、结构和机理；性能事实绑定原句、页码、表格、DOI、条件、单位及置信度。常见压力、温度、通量/渗透率单位会规范化。
- 结构化事实写入 Neo4j，同时保留 PostgreSQL 事实表作为审核记录与图谱不可用时的检索回退；全文切片使用 `BAAI/bge-m3` 写入 Chroma。
- 每次回答先扫描整个知识库文献清单和全部已有切片，再进行纳滤中英文词表扩展。LangGraph 路由材料/性能/实验关系问题到图谱，机理/综述问题到向量检索，复杂问题使用混合检索；关键词检索始终作为补充。只有最终送入 LLM 的引用证据按相关性和独立文献覆盖度压缩，因此上下文条数限制不会变成知识库扫描范围限制。回答提示强制使用 DOI、页码和短原文证据。
- 开启“跨文献知识发现”后，DeepSeek 会在本次命中的多篇证据之间寻找 `pattern`、`contradiction` 和 `hypothesis`。每条结果都保存支持/反对证据、来源模式、置信度、新颖性、假设、适用边界及验证/证伪方案；只靠题名摘要时置信度上限为 0.45，混合摘要证据上限为 0.65，全文证据上限为 0.82。
- AI 发现保存为 Neo4j `AIInsight` 节点，并通过 `SUPPORTED_BY` / `CONTRADICTED_BY` 连接文献。状态分为 AI 综合、AI 假设、人工已审阅、实验/外部证据已验证和驳回；只有人工明确填写验证依据后才能标记为“已验证”。历史 AI 假设只作为推理记忆，不会被当作论文原始证据引用。
- 对话、滚动摘要字段、当前任务、引用证据、工具轨迹、新增记录、索引版本和人工反馈都在数据库持久化。关闭页面或重启服务不会清空外部卷。
- 高质量轨迹可经人工评分、修订和授权后导出为 `instruction`、`input`、`output` JSONL，供后续合规 LoRA/蒸馏使用。

## 目录

```text
app/
  api.py                 FastAPI 路由
  models.py              PostgreSQL/SQLite 数据模型与唯一约束
  services/
    literature.py        多源检索与引文网络扩展
    discovery_service.py 候选持久化、相关性与查重标记
    fulltext.py           公开 PDF/XML/HTML 直读 + Unpaywall/TDM 补充
    parser.py             MinerU/GROBID/PyMuPDF/XML/公开 HTML 解析
    extractor.py          DeepSeek 结构化抽取、实体/单位规范化
    graph_store.py        Neo4j 图谱写入与检索
    vector_store.py       bge-m3/Chroma 持久向量索引
    rag.py                LangGraph 混合 RAG 与证据回答
    knowledge_discovery.py 跨文献规律、矛盾、假设与验证计划
    pipeline.py           文献后台处理状态机
    automation.py         Redis/RQ 持续发现、停止和下一轮调度
    import_service.py     用户选择和自动任务共用的并发安全导入
ui/streamlit_app.py       对话 + 文献采集/阅读 + 知识发现图谱三大模块
data/nanofiltration_vocab.json
docker-compose.yml        PostgreSQL/Redis/Neo4j/MinIO/Chroma/GROBID/API/Worker/UI
```

## 方式一：Docker Compose（完整部署）

你已经在 `Ubuntu-24.04` WSL 中安装 Docker Engine 时，不需要再安装 Docker Desktop。可直接在普通 PowerShell 的项目目录执行；脚本会自动转入该 WSL 发行版运行 Compose：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\configure_env.ps1
powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1
docker compose logs -f api worker
```

`.env` 至少填写：

```dotenv
SILICONFLOW_API_KEY=你的硅基流动密钥
OPENALEX_API_KEY=你的免费OpenAlex密钥
OPENALEX_EMAIL=你的联系邮箱
UNPAYWALL_EMAIL=你的联系邮箱
```

按权限选填 `SEMANTIC_SCHOLAR_API_KEY`、`ELSEVIER_API_KEY` 和 `ELSEVIER_INSTTOKEN`。OpenAlex 从 2026-02-13 起要求 API key。ScienceDirect TDM 是否能返回全文取决于 API Key 对应的授权范围、机构 IP 或 insttoken；系统不会绕过订阅权限。

默认 Docker 后端不启动 Streamlit，便于在 Anaconda Prompt 中运行：

```bat
conda activate nf-agent
cd /d "D:\桌面\agent for NF\gpt"
streamlit run ui\streamlit_app.py
```

也可以运行 `scripts\start_streamlit_anaconda.cmd`。完整的 Windows、WSL 2 和 Anaconda 步骤见 [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md)。如希望 Streamlit 也由 Docker 运行，使用：

```powershell
docker compose --profile docker-ui up --build -d
```

启动后访问：

- Streamlit：<http://localhost:8501>
- FastAPI 文档：<http://localhost:8000/docs>
- Neo4j Browser：<http://localhost:7474>
- MinIO Console：<http://localhost:9001>

生产环境务必在 `.env` 修改 `POSTGRES_PASSWORD`、`NEO4J_PASSWORD`、`MINIO_ACCESS_KEY` 和 `MINIO_SECRET_KEY`。不要提交 `.env`。

## 方式二：本机轻量开发

此模式默认使用 SQLite、本地对象目录、内嵌 Chroma 和 FastAPI 后台任务，不要求 Redis/PostgreSQL/Neo4j/MinIO。没有 Neo4j 时会查询 PostgreSQL/SQLite 的结构化事实表；没有 MinerU/GROBID 时使用 PyMuPDF。适合先验证界面和流程。

建议 Python 3.11 或 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
python -m app.cli init-db
```

分别打开两个终端：

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run ui/streamlit_app.py --server.port 8501
```

本机首次使用 bge-m3 需要在 `.env` 设置 `ALLOW_EMBEDDING_DOWNLOAD=true`，模型会由 `sentence-transformers` 下载并缓存。为保证未联网时流程仍可验证，模型不可用时会明确退化为确定性哈希向量；生产部署应启用并确认健康页所对应的 bge-m3 模型文件已就绪。

## Streamlit Community Cloud 部署

云端入口为 `streamlit_cloud.py`，它现在是**纯前端**：不会在 Streamlit 实例内启动 FastAPI、SQLite、Chroma 或后台 Worker，也不会保存 PDF、向量或对话。正式主库和全部任务只运行在长期在线服务器上。Streamlit Cloud 休眠、重启或重建后会重新连接同一服务器，因此不会清空文献、知识图谱、向量、任务或对话；服务器 Worker 在网页关闭期间仍继续运行。

推荐使用专用受限 SSH 密钥，把云前端转发到服务器回环地址 `127.0.0.1:8000`。该密钥在服务器 `authorized_keys` 中只能转发这个端口，不能打开 Shell、PTY、代理转发或其他端口。将 [.streamlit/server-secrets.example.toml](.streamlit/server-secrets.example.toml) 的内容复制到 Community Cloud 的 **App settings → Secrets**，替换主机、指纹和私钥。真实私钥只能放 Secrets，绝不能提交到 GitHub。

Streamlit Secrets 只需要服务器连接信息，不再填写 SiliconFlow、OpenAlex、Elsevier、PostgreSQL、Neo4j 或 MinIO 密钥；这些凭据全部留在服务器 `.env`。仓库根目录的 `requirements.txt` 也只安装云前端所需的 Streamlit、HTTPX 和 Paramiko，不再在云端下载 PyTorch、Chroma 或解析器依赖。

在 Community Cloud 中选择：

```text
Repository: effortV/agent-for-nf
Branch: main
Main file path: streamlit_cloud.py
```

如果后续已有可信 HTTPS 反向代理或 Cloudflare Tunnel，也可以只设置 `NF_BACKEND_URL=https://...`；可选的 Bearer/Cloudflare Access 服务令牌字段已列在 Secrets 示例中。程序默认拒绝纯 HTTP 公网地址。

云前端只是服务器主库的操作界面。所有访问者目前共享同一个知识库和对话列表，因此只应把应用链接发给可信协作者；需要面向不受信任公众开放时，还应增加用户登录、权限分级、租户隔离和配额。

## MinerU 接入

在运行 worker 的环境安装你采用的 MinerU 发行版，然后设置命令模板，必须保留 `{input}`、`{output}` 占位符。例如：

```dotenv
MINERU_COMMAND=mineru -p {input} -o {output}
```

命令通过参数数组执行，不启用 shell。不同 MinerU 版本的 CLI 与 JSON 结构可能不同；当前适配器会寻找输出目录中的 JSON 或 Markdown，并在不兼容时记录原因后自动回退 GROBID。可在 `app/services/parser.py` 针对固定版本扩展字段映射。

## 使用流程

1. 打开 Streamlit，新建或从侧栏恢复历史对话。
2. 展开“扩充知识库”，输入主题并检索。
3. 查看扩展词、已有文献推荐和去重后新候选；选择 0～200 篇并确认。
4. 在任务区查看公开网页/全文、题名摘要兜底、解析、抽取和图谱/向量入库进度。没有合法全文的候选会明确以题名/摘要模式入库，不会伪造正文或页码。
5. 已成功入库的单篇文献立即可以参与问答；任务结束后知识库索引版本递增。
6. 免费全文没有被题录 API 识别时，在“公开网址/DOI导入”粘贴文章页或 PDF 地址；已有摘要记录会原位升级。也可在“文献阅读”批量重新检查摘要文献。
7. 对订阅文献使用“合法全文上传”，确认权限后进入同一解析链。
8. 继续在原对话提问，可打开“跨文献知识发现”。回答中的论文事实和 AI 推断分区显示；到“知识发现图谱”查看证据链、边界条件、验证计划并人工审核。

## AI 知识发现的边界

“AI 发掘人类难以手工发现的东西”在这里指机器可以快速比较大量异构证据、定位条件组合与异常关系，并提出高信息增益的实验方向，不代表模型天然拥有超出证据的真理。系统采用以下分层：

1. `Fact`：从单篇文献原文抽取，绑定 DOI、页码、表格、实验条件和原句。
2. `AIInsight(pattern/contradiction)`：由至少两篇独立文献归纳或比较得到，属于可追溯的 AI 综合。
3. `AIInsight(hypothesis)`：尚未验证的新解释或预测，必须附带变量、预期结果和证伪判据。
4. `validated`：研究人员已提供实验或外部证据并人工确认，才进入已验证状态。

因此，系统可以帮助发现潜在新知识，但最终科学结论仍需实验、统计检验和同行评议。

## 训练数据导出

用户反馈接口会把评分、人工修订和“允许用于训练”标记写入 `training_traces`。默认只导出批准的数据：

```powershell
python -m app.cli export-training --output data/runtime/exports/training.jsonl
```

如仅做内部审核、希望同时导出未批准项：

```powershell
python -m app.cli export-training --include-unapproved
```

导出的每行包含 `instruction`、`input`、`output` 以及证据和工具轨迹元数据。后续训练前仍须单独执行版权、隐私、数据许可和质量审查。

## 运维与数据安全

- `docker compose down` 只停止服务，命名卷仍保留。不要使用 `docker compose down -v`，除非明确要删除全部知识库数据。
- 本机模式的数据位于 `data/runtime/`；此目录已被 Git 忽略。
- 文献对象键按 `kb/{knowledge_base_id}/documents/{document_id}/` 隔离。
- 每个回答保存索引版本和证据；即使索引日后扩充，旧回答的证据快照仍在 PostgreSQL。
- FastAPI 继续只绑定服务器 `127.0.0.1:8000`；Streamlit Cloud 使用受限 SSH 端口转发访问，不需要开放 API 端口。若改为公网 HTTPS 反向代理，应增加身份认证、TLS、速率限制、上传扫描、租户隔离和审计策略。

## 测试

```powershell
python -m compileall app ui tests
pytest
ruff check app ui tests
```

测试不访问网络，也不需要真实 API Key。
