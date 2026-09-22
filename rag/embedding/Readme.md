# embedding: bge-m3 常驻嵌入服务 —— 可独立部署、对外开放的独立项目

> 本项目把它生产化: **模型常驻的服务 + 客户端**, 更进一步补上了

**独立服务**该有的东西 —— 服务端自动攒批、可选鉴权、监控、部署配套,
而不是"只在自己机器上演示的 demo"。

```
                调用方(任意服务 / curl / 向量库写入端)
                    │  HTTP (可跨机器, 可选 X-API-Key)
                    ▼
        ┌─────────────────────────────┐
        │  server.py (uvicorn 单进程)  │   不要 --workers>1: 每进程复制一份模型
        │  ┌──────────┐  ┌──────────┐ │   横向扩容 = 多副本 + 负载均衡
        │  │ Batcher  │→ │ encode   │ │
        │  │ 攒批合并  │  │ executor │ │   单模型并发前向无收益(并发=1):
        │  │ (服务端)  │  │  (串行)  │ │   小请求在服务端合并成一次前向
        │  └──────────┘  └──────────┘ │
        │  bge-m3 模型 (常驻 ~2.3GB)   │
        └─────────────────────────────┘
```



## 目录结构

```
embedding/   (独立项目, 与 rag 平级)
├── server.py        服务: /health /info /embed /metrics + 服务端攒批器(Batcher)
├── client.py        客户端 SDK: EmbeddingClient(分块/重试/等待就绪/API key)
├── batcher_test.py  攒批器离线单测(不加载模型, 秒级跑完)
├── smoke_test.py    冒烟测试: 自动起服务 → 真实调用 → 断言 → 收尾
├── deploy/          embedding.service(systemd) + Dockerfile
├── requirement.txt
├── .env.example     全部可配置项
└── emb1.py          原来的实验脚本(对照用, 可删)
```



## 快速开始

以下命令在本项目目录内执行, 虚拟环境是项目自己的 `.venv`。

```bash
# 0. 创建独立虚拟环境并装依赖(首次装 torch/flagembedding, 需要几分钟)
python3 -m venv .venv
.venv/bin/pip install -r requirement.txt

# 1. 起服务(首次会自动下载 bge-m3, 已缓存则秒起)
#    模型已缓存(~/.cache/huggingface)或离线环境, 加 HF_HUB_OFFLINE=1 完全离线加载:
#    —— 否则即使有缓存, 启动时仍会去 huggingface.co 校验, 国内直连不通会报
#       httpx.ConnectTimeout 而启动失败。需联网下载时 server.py 默认已走 hf-mirror.com 镜像。
HF_HUB_OFFLINE=1 .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001
#    或: HF_HUB_OFFLINE=1 .venv/bin/python server.py

# 2. 验证 + 演示
curl -s localhost:8001/health
.venv/bin/python client.py demo
.venv/bin/python client.py "什么是BGE M3？" "BGE M3支持密集、稀疏和多向量检索" --sparse

# 3. 攒批器离线单测(不加载模型) / 全链路冒烟(加载真实模型)
.venv/bin/python batcher_test.py
.venv/bin/python smoke_test.py
```



## 为什么它能"独立开放", 而不只是 demo

三个层次, 从现状到完整:


| 层次  | demo 阶段            | 现在                                                           |
| --- | ------------------ | ------------------------------------------------------------ |
| 网络  | 本机手动跑              | 监听 `0.0.0.0`, 任何机器可 curl; client 是普通 HTTP SDK, 可被任意服务 import |
| 吞吐  | 并发=1 → 要求"调用方自己攒批" | **服务端自动攒批**: 并发小请求合并成一次前向, 调用方无需感知(见下)                       |
| 运营  | 无                  | 可选 API Key、`/metrics`、systemd/Docker 托管、多副本 + LB 扩容          |




### 核心: 服务端自动攒批 (并发=1 不是瓶颈的正确解法)

单模型**同时**跑多个前向只会互相拖慢、还可能爆显存, 所以模型前向放在单线程
executor 里串行(这也是为什么不能靠"多进程/多线程"解决)。但串行不意味着
低吞吐 —— 吞吐 = 每次前向处理的文本数 ÷ 耗时, 所以关键是**让每次前向尽量满**:

- 请求入队, 攒批器把同选项(dense/sparse/colbert/max_length)的并发小请求合并成
一批, 一次 `model.encode()` 后按各自文本数把结果切回 —— 相当于服务端替
调用方实现了"攒 batch"。10 个并发小请求 = 1 次满批前向, 而不是 10 次空转。
- 单个小请求只多等 `EMBED_BATCH_GRACE_MS`(默认 4ms)找"同路人", 没有就单独发车;
单请求自带大量文本(批量索引)时立即发车, 零额外延迟。
- 选项不同的请求分桶攒批, 不会出现"为了一个 colbert 请求给所有人算 colbert"。

行为契约在 `batcher_test.py`(桩 encode, 不加载模型, 秒级验证)。

### 怎么用: 调用方视角

服务端已开放给局域网/内网, 从**另一台机器**直接调:

```bash
# 不开鉴权时(仅限内网), 或已配好代理网关:
curl -s -X POST http://<服务机IP>:8001/embed -H 'Content-Type: application/json' \
  -d '{"texts":["什么是BGE M3？","BGE M3支持混合检索"],"return_sparse":true}'

# 开了鉴权(推荐), 带 X-API-Key:
curl -s -X POST http://<服务机IP>:8001/embed \
  -H 'Content-Type: application/json' -H 'X-API-Key: <你的key>' \
  -d '{"texts":["什么是BGE M3？"]}'
```

在自己服务里当 SDK 用 —— 批量索引与在线检索都无需自己拼 batch:

```python
from client import EmbeddingClient
c = EmbeddingClient("http://<服务机IP>:8001", api_key=os.environ["EMBED_API_KEY"])

# 批量索引: 一次塞一批文本(自动分块), 服务端一次前向
vecs = c.embed([d.text for d in fetch_docs()])["dense"]

# 在线检索: 单个 query 随手发即可, 并发查询由服务端自动攒批
q_vec = c.embed(query_text)["dense"][0]

c.wait_until_ready()   # 编排/探活: 阻塞直到模型就绪(超时抛异常)
```



## API

鉴权: 设置 `EMBED_API_KEY` 后, `/embed` `/info` 需请求头 `X-API-Key`;
`/health` 不鉴权(负载均衡探活用), `/metrics` 建议放在内网或网关后面。


| 端点             | 说明                                               |
| -------------- | ------------------------------------------------ |
| `GET /health`  | 存活/就绪: `status/device/dim/pending(排队数)/uptime_s` |
| `GET /info`    | 模型信息 + 攒批器当前配置                                   |
| `POST /embed`  | 嵌入 (dense/sparse/colbert), 见下                    |
| `GET /metrics` | Prometheus 文本格式                                  |




### /embed 请求/响应

请求:


| 字段                    | 默认    | 说明                                                                                                  |
| --------------------- | ----- | --------------------------------------------------------------------------------------------------- |
| `texts`               | 必填    | 文本列表, 1~128 条(服务端会跨请求合并攒批, 无需自行分块)                                                                  |
| `return_dense`        | true  | 1024 维向量                                                                                            |
| `return_sparse`       | false | `sparse_ids`(token_id→权重, 喂 OpenSearch/ES 的 rank_features 用) + `sparse_tokens`(token 文本→权重, 人看/调试用) |
| `return_colbert_vecs` | false | 多向量, 开销大, 研究用                                                                                       |


响应 `meta`: `num_texts / elapsed_ms / dim / device / model`。

## /metrics (接 Prometheus/Grafana)

`bge_embed_requests_total{status}` 请求量与失败 / `bge_embed_latency_seconds`
平均端到端延迟 / `bge_batcher_{batches,requests,texts}_total` 攒批效率
(看 `batches` vs `requests`: 若两者几乎 1:1, 说明没攒上, 调大 grace 或确认调用方批量)
/ `bge_batcher_pending` 排队数(服务忙不忙, 配告警)。

## 部署: 让它开机常驻、可扩展



### 方案 A: systemd(单机最省事)

```bash
# 复制 unit 并按机器改路径; 环境变量放到 ~/.config/embedding.env(模板: .env.example)
cp deploy/embedding.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now embedding
journalctl --user -u embedding -f      # 看日志
loginctl enable-linger $USER           # 想开机自启(不登录也跑)时开
```



### 方案 B: Docker

```bash
docker build -t bge-embed .   # 见 deploy/Dockerfile(GPU 部署的注意事项在里面)
docker run -d --name bge-embed -p 8001:8001 \
  -e HF_HUB_OFFLINE=1 -e EMBED_API_KEY=change-me \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface bge-embed
```



### 方案 C: 多副本 + 负载均衡(吞吐不够时)

**原则: 一个容器/进程一份全量模型(~2.3GB), 绝不** `--workers>1`**。**
吞吐不够就多起几个副本(不同端口/机器), 前端 nginx 轮询:

```nginx
upstream embed {
    server 10.0.0.5:8001;
    server 10.0.0.6:8001;   # 显存大就多放几个, 每副本一份模型
}
server { listen 80; location / { proxy_pass http://embed; } }
```

健康检查就用 `/health`(不鉴权)。副本内 `pending` 升高是正常排队现象,
攒批器会把它并成大批次; 副本整体长时间 5xx/超时再摘除。

## 设计要点(为什么这么做)

1. **模型单例、启动时加载+预热**: 一次 2.3GB, 不能每个请求/每次运行都加载;
  预热把 kernel 编译和显存分配挪到启动期, OOM/下载失败提前暴露。
2. **服务端攒批, 而不是要求客户端攒批**: 见上文"核心"。原"并发=1 → 客户端攒批"
  的设计只适用于单一受控调用方; 对外服务必须自己消化小请求。
   模型前向统一在单线程 executor 执行 —— 串行 + 不占事件循环(原来用
   BoundedSemaphore+线程池兜底, 现在由攒批器调度, 更可控)。
3. **同步校验、异步排队**: 空串/超长/无输出 flag 等校验不占模型资源, 进队列前做掉;
  真正的前向只有攒批器触发的那么几次。
4. **稀疏权重给两种形态**: FlagEmbedding 原生返回 token_id→权重(喂倒排索引最省),
  `convert_id_to_token` 解码成词形式, 便于理解调试——混合检索两个通道各取所需。
5. **分桶攒批**: 不同 `return_`* 选项的请求独立成批, 互不污染、互不拖累。



## 生产化清单

- [x] 模型常驻、与业务解耦、HTTP 接口
- [x] 服务端自动攒批(小请求高吞吐, 不再依赖调用方攒批)
- [x] 可选 API Key 鉴权 + `/metrics` + 客户端重试/等待就绪
- [x] 部署配套: systemd / Docker / 多副本 + nginx LB(勿 `--workers>1`)
- [ ] 版本治理: 模型版本 tag 进索引元数据, 换模型 = 全量向量重建
- [ ] 监控告警落地: Prometheus scrape `/metrics` + Grafana + 队列深度告警
- [ ] 降级: 服务挂时熔断到 BM25, 别让检索整体不可用
- [ ] 大规模高并发时再考虑 vLLM(TEI 不支持 bge-m3)



## 常见问题

- **内存不足/OOM**: 调低 `EMBED_INFER_BATCH`(M1 16G 用 8); fp16 已默认开。
- **首次启动在下载权重**: bge-m3 ~2.1GB → `~/.cache/huggingface`; 国内可先
`export HF_ENDPOINT=https://hf-mirror.com` 再启动。
- **想要多进程高吞吐**: 起多个实例(副本) + nginx 轮询, 而不是加 worker。
- **单个小请求的额外延迟**: 攒批等"同路人"最多 `EMBED_BATCH_GRACE_MS`(默认 4ms);
延迟敏感可调到 1~2, 批量吞吐场景可调大(如 20)换取更大合并窗口。
- **并发高但** `/metrics` **里 batches≈requests(没攒上)**: 调用方请求到达间隔大于
grace, 属正常 —— 要么调大 grace, 要么让调用方一次多塞几条文本。
- **MPS 上 fp16 有算子报错**: `EMBED_FP16=0` 或 `EMBED_DEVICE=cpu`(慢一些)。
- **不想对外开放 / 纯本机**: `EMBED_HOST=127.0.0.1` + 不开 `EMBED_API_KEY` 即可。

