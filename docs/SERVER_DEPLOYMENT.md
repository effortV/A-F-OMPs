# NF-Atlas 服务器主库部署

## 角色约束

- 服务器是唯一正式主库，运行 PostgreSQL、Redis/RQ、Neo4j、Chroma、MinIO、GROBID、FastAPI、Worker 和 Streamlit。
- D 盘原始 Docker 卷在服务器验收后停止服务并保留，作为迁移时点的离线备份，不再持续采集。
- Streamlit 只是客户端。关闭浏览器或 SSH 隧道不会停止服务器 Worker。
- 所有容器使用 `restart: unless-stopped`；Docker 随系统启动后自动恢复。
- Worker 启动时根据 PostgreSQL 恢复未完成任务。已入库文献由 DOI、OpenAlex ID、题名作者指纹和文件 SHA-256 去重。

## 服务器启动

```bash
cd "/home/root2/data/zzh/agent for nf"
scripts/server_compose.sh \
  -f docker-compose.yml \
  -f docker-compose.server.yml \
  --profile docker-ui up -d
```

服务端口默认只绑定 `127.0.0.1`，避免 PostgreSQL周边服务、Neo4j、MinIO 和 Streamlit 暴露到公网。

从 Windows 建立 Streamlit 隧道：

```powershell
ssh -i "$HOME\.ssh\nf_atlas_server_ed25519" -p 9012 `
  -L 8501:127.0.0.1:8501 root2@112.15.87.51
```

保持该窗口打开，在浏览器访问 <http://localhost:8501>。关闭窗口只会关闭访问隧道，不会停止服务器任务。

## 状态检查

```bash
cd "/home/root2/data/zzh/agent for nf"
scripts/server_status.sh
```

## 每日备份

`scripts/server_backup.sh` 会：

1. 暂停 API/Worker，阻止备份期间产生新写入；
2. 用 `pg_dump -Fc` 创建一致的 PostgreSQL 逻辑备份；
3. 停止 Neo4j、Chroma 和 MinIO 后归档图数据、向量和全文对象；
4. 记录表计数、部署配置和 SHA-256；
5. 自动恢复整套服务；
6. 默认保留最近 30 天备份。

安装 `root2` 用户的每日 03:20 定时任务：

```bash
scripts/install_server_backup_cron.sh
```

备份目录为：

```text
/home/root2/data/zzh/agent for nf/backups/YYYY-MM-DDTHHMMSSZ/
```

`.env` 的备份权限为 `0600`，其中包含私密 API 凭据，不应公开或上传 GitHub。

## 恢复

恢复会清空服务器目标数据卷，只能在明确选择了正确快照后执行：

```bash
cd "/home/root2/data/zzh/agent for nf"
NF_ATLAS_ALLOW_RESTORE=YES scripts/server_restore.sh \
  "/home/root2/data/zzh/agent for nf/backups/YYYY-MM-DDTHHMMSSZ"
```

Redis 队列不直接还原，避免陈旧 RQ 调用重复执行。Worker 会从 PostgreSQL 的任务记录安全恢复，数据库唯一约束和四级查重会跳过已经入库的文献。
