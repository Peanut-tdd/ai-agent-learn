# pdfimgpipeline

文档图像 **OCR / VLM 智能路由与执行服务**：对单张图片自动判断该走本地 OCR、视觉大模型（VLM）
还是两者融合，带真实超时、重试、降级、并发上限与两级缓存，对外提供 HTTP 接口。

- 核心逻辑与 HTTP 服务分层：核心是纯 Python 包，可当库用；服务层是薄薄一层 FastAPI。
- 缓存可插拔：Redis 连不上自动降级进程内内存缓存，缓存故障不会拖垮服务。

---

## 一、功能一览


| 功能     | 说明                                                                             |
| ------ | ------------------------------------------------------------------------------ |
| 图像特征提取 | OpenCV 提取 6 维特征：文字占比、图像熵、边缘密度、颜色方差、连通域数量、宽高比                                   |
| 规则打分   | 按阈值加权求和，得到路由分数与可解释命中原因                                                         |
| OCR 探测 | 仅在模糊区间触发一次轻量 RapidOCR，用「字符数 + 置信度」二次判断                                         |
| 三路决策   | `ocr` / `vlm` / `hybrid`，附完整 `reason`                                          |
| 两级缓存   | 决策 `pdfimg:route:<hash>` + 结果 `pdfimg:result:<hash>`，键为图片内容 SHA-256，TTL 默认 7 天 |
| 真实超时   | 编排层 `asyncio.wait_for` + SDK 层客户端超时，两层保证真正中断                                   |
| 重试     | 仅对超时/连接类瞬时故障指数退避重试，参数类错误立即抛出                                                   |
| 降级     | VLM 失败自动降级 OCR；hybrid 两条腿并发，谁成功用谁                                              |
| 并发背压   | OCR / VLM 各自 `asyncio.Semaphore` 限流（VLM 很贵）                                    |
| 全异步    | Redis 用 `redis.asyncio`，cv2/文件 hash/同步 OCR 全部 `to_thread`                      |
| 批量处理   | `/route:batch`、`/extract:batch` 一次多张，内部并发 + 单项失败隔离                             |
| 可观测    | 结构化日志 + Prometheus 文本指标 `/metrics`                                             |
| 鉴权     | 设置 `PDFIMG_API_KEY` 后受保护端点需带 `X-API-Key`                                       |


---



## 二、目录结构

```
pdfimgpipeline/
├── pyproject.toml                 # 打包与依赖
├── requirement.txt                # 等价的运行期依赖（pip 用户）
├── .env.example                   # 配置示例
├── readme.md
├── src/pdfimgpipeline/
│   ├── models.py                  # RouteDecision / ImageFeatures / RouteResult
│   ├── config.py                  # pydantic-settings 集中配置
│   ├── features.py                # PDFImageRouter：特征 + 规则打分
│   ├── cache.py                   # 可插拔缓存后端 + RouteCache / ResultCache
│   ├── router.py                  # HybridRouter：缓存 -> 特征 -> 探测 -> 决策
│   ├── executor.py                # ExtractorExecutor：超时/重试/降级/并发融合
│   ├── pipeline.py                # PDFImagePipeline：顶层入口
│   ├── wiring.py                  # Settings -> pipeline 依赖装配
│   ├── adapters/                  # RapidOCR / OpenAI 兼容 VLM 适配器
│   └── service/                   # FastAPI app / schemas / metrics
├── tests/
└── deploy/                        # Dockerfile / docker-compose / systemd
```

---



## 三、快速开始

```bash
cd pdfimgpipeline
# ⚠️ 用 Python 3.11 / 3.12（3.13 无 RapidOCR wheel，OCR 通道不可用）
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # 或 pip install -r requirement.txt

cp .env.example .env           # 按需填写 VLM_API_KEY / VLM_BASE_URL
# 本地起 Redis（可选；不启动会自动降级内存缓存）
docker run -d -p 6379:6379 redis:7-alpine

pdfimgpipeline                 # 或: uvicorn pdfimgpipeline.service.app:app --port 8000
```

调用：

```bash
curl localhost:8000/health

# 只做路由决策（便宜）
curl -X POST localhost:8000/route -F "file=@sample.png"

# 跑完整流水线
curl -X POST localhost:8000/extract -F "file=@sample.png"

# 配置了 PDFIMG_API_KEY 时
curl -X POST localhost:8000/extract -H "X-API-Key: <key>" -F "file=@sample.png"
```

---



## 四、HTTP 接口


| 方法   | 路径               | 鉴权  | 作用                                             |
| ---- | ---------------- | --- | ---------------------------------------------- |
| GET  | `/health`        | 否   | 存活/就绪（LB 探活）                                   |
| GET  | `/info`          | 是   | 配置摘要（不泄露密钥）                                    |
| POST | `/route`         | 是   | 只做路由决策，返回 `decision/reason/features/ocr_probe` |
| POST | `/extract`       | 是   | 跑完整流水线，返回最终 JSON                               |
| POST | `/route:batch`   | 是   | 批量路由决策                                         |
| POST | `/extract:batch` | 是   | 批量完整处理                                         |
| GET  | `/metrics`       | 否   | Prometheus 文本格式                                |


`/route`、`/extract` 均接收 `multipart/form-data` 的 `file` 字段（图片），
大小超过 `MAX_UPLOAD_MB` 返回 `413`。

批量端点接收多个 `files` 字段，张数超过 `MAX_BATCH_SIZE` 返回 `413`，内部按
`BATCH_MAX_CONCURRENCY` 并发，单张失败只影响该项（`ok=false` + `error`），顺序与输入一致。

---



## 五、执行逻辑

```
输入图片
  │
  ├─▶ 结果缓存 pdfimg:result:<hash> ──命中──▶ 直接返回
  │
  ├─▶ 决策缓存 pdfimg:route:<hash>  ──命中──▶ 复用决策
  │
  ├─▶ cv2 提 6 维特征 → 规则打分 score
  │
  ├─ score ≥ 3 ───────────────▶ vlm
  ├─ score ≤ -1 ──────────────▶ ocr
  └─ -1 < score < 3 ──────────▶ 先 OCR 探测，再判：
        probe_chars == 0                     → vlm
        probe_chars > 50 且 probe_conf > .85 → ocr
        probe_chars < 10                     → vlm
        其余                                  → hybrid
  │
  └─▶ 执行（超时 + 重试 + 降级；hybrid 时 OCR/VLM 并发）
        vlm 失败 → 降级 ocr
  │
  └─▶ 组装 JSON，写结果缓存
```



### 规则打分明细（分数越高越像「需要 VLM 的复杂图」）


| 条件                               | 分值  | reason              |
| -------------------------------- | --- | ------------------- |
| `text_ratio < 0.15`              | +2  | `low_text_ratio`    |
| `text_ratio > 0.4`               | −2  | `high_text_ratio`   |
| `image_entropy > 6.5`            | +2  | `high_entropy`      |
| `image_entropy < 4.5`            | −1  | `low_entropy`       |
| `edge_density > 0.12`            | +1  | `high_edge_density` |
| `color_variance > 800`           | +2  | `rich_color`        |
| `color_variance < 100`           | −1  | `gray_image`        |
| `num_connected_components > 500` | −1  | `many_components`   |




### 超时策略

- **编排层**：`asyncio.wait_for(fn(...), timeout)`，到期 cancel 内层任务并触发降级。
- **SDK 层**：`AsyncOpenAI(timeout=28)`，默认略小于编排层 30s，保证 HTTP 连接先断开。
- 重试预算上界 ≈ `(max_retries + 1) * timeout + 退避`。
- 注意：`to_thread` 里的同步调用（RapidOCR）无法被强杀，超时后线程仍会跑完、结果被丢弃；
对幂等 OCR 可接受，如需硬隔离请改用进程池。

---



## 六、输出结果

`POST /extract` 返回：

```json
{
  "image_hash": "3f2a...c9",
  "decision": "hybrid",
  "reason": "low_text_ratio|high_entropy|ocr_probe(chars=23,conf=0.71)|probe_medium->hybrid",
  "features": {
    "text_ratio": 0.08, "image_entropy": 7.12, "edge_density": 0.15,
    "color_variance": 921.4, "num_connected_components": 312, "aspect_ratio": 0.707
  },
  "ocr_probe": { "chars": 23, "conf": 0.71 },
  "result": {
    "engine": "hybrid",
    "data": {
      "ocr_text": "……",
      "vlm_description": "……",
      "merged": "……\n[VLM]:……"
    }
  },
  "from_cache": false
}
```

`result.engine` 取值与 `data` 形态：


| engine                            | data                                        |
| --------------------------------- | ------------------------------------------- |
| `ocr`                             | `{"text": "..."}`                           |
| `vlm`                             | `{"text": "..."}`                           |
| `hybrid`                          | `{"ocr_text", "vlm_description", "merged"}` |
| `ocr(fallback)` / `vlm(fallback)` | 由存活通道产出；附加 `fallback_from` / `vlm_error`    |
| `none`                            | `null`，附 `error: "both failed"`             |


> 降级到 OCR 且 OCR 也失败时，接口仍返回 200 + 结构化 `error`，便于调用方统一处理；
> 只有非预期异常（如文件损坏、内部错误）才返回 500。



### 批量返回

`POST /extract:batch` / `POST /route:batch` 返回：

```json
{
    "count": 2,
    "ok_count": 2,
    "failed_count": 0,
    "results": [
        {
            "index": 0,
            "filename": "cgjBBoMvToo_5ZdjBm87isLu27Dj6vXbAz5HdeKPJyu-UhjvYTJFerXjRcdhy0xd.jpg",
            "ok": true,
            "data": {
                "image_hash": "93f8663725606297708d2b9529aa8e76f27490da399026b62f818ed37de494b6",
                "decision": "vlm",
                "reason": "high_entropy|rich_color|many_components",
                "features": {
                    "text_ratio": 0.21912990746707303,
                    "image_entropy": 7.844100475311279,
                    "edge_density": 0.06825602535341438,
                    "color_variance": 7318.02001953125,
                    "num_connected_components": 29762,
                    "aspect_ratio": 0.6708165507140241
                },
                "ocr_probe": {
                    "chars": 0,
                    "conf": 0.0
                },
                "result": {
                    "engine": "vlm",
                    "data": {
                        "text": "图类型为食物照片，非文档类图像。图中无坐标轴与单位，无分组与曲线含义，无可读的关键数值与结论。画面主体为盛在白色盘中的红烧肉块与一颗卤蛋，表面点缀绿色葱花，底部有深色酱汁。"
                    }
                },
                "from_cache": true
            },
            "error": null
        },
        {
            "index": 1,
            "filename": "WechatIMG245.jpg",
            "ok": true,
            "data": {
                "image_hash": "354fbaa391d3edd74e31845447829d5abd98f88b74e0b1ed3a31ee2861c68049",
                "decision": "ocr",
                "reason": "low_entropy|rich_color|many_components|ocr_probe(chars=341,conf=0.99)|probe_high_quality_text->ocr",
                "features": {
                    "text_ratio": 0.3359170143530333,
                    "image_entropy": 3.932445526123047,
                    "edge_density": 0.03696716220412903,
                    "color_variance": 13978.2822265625,
                    "num_connected_components": 1276,
                    "aspect_ratio": 0.46208530805687204
                },
                "ocr_probe": {
                    "chars": 341,
                    "conf": 0.9899744337752457
                },
                "result": {
                    "engine": "ocr",
                    "data": {
                        "text": "00:39\n74\n关注\n朋友\n推荐\nNotes\n我的学习顺序，慢慢清晰了\n我的学习顺序也逐渐清晰了：\n先搞懂大模型原生APl、消息格式、Token和\nFunction Calling ;\n再学习Prompt与上下文管理；\n然后完整跑通ToolUse和AgentLoop;\n接着补RAG的切分、检索、重排与评测；\n最后再学LangGraph等框架和工程化部署。\n真正让我完成转型的，不是刷了多少门课程，\n而是做出了一个完整项目：前端能交互，\n后端能调用工具，RAG能检索企业资料，\n关键操作有人为确认，系统还有日志、\n监控和失败重试。\n先把底层跑通，再学框架；\n先把项目做出来，再把项目做稳。\n第3页/共4页\n后端开发拜拜，哥转型Agent成功了\n#Al#Agent#Java#Al学习#AI应用开发\n黑马老高\n+关注\n聊AI\n推荐\n评论\n3"
                    }
                },
                "from_cache": true
            },
            "error": null
        }
    ]
}
```

批量端点对每张图独立走缓存，重复图片会直接命中；批量指标见 `/metrics`
（`pdfimg_extract_batch_ok_total` / `..._error_total` / `..._latency_seconds`）。

---



## 七、配置

全部可用环境变量 / `.env` 覆盖，完整列表见 `.env.example`。常用：


| 变量                                             | 默认                   | 说明                             |
| ---------------------------------------------- | -------------------- | ------------------------------ |
| `VLM_MODEL`                                    | `qwen-vl-max`        | 视觉模型名                          |
| `VLM_API_KEY` / `OPENAI_API_KEY`               | —                    | 视觉接口密钥                         |
| `VLM_BASE_URL` / `OPENAI_BASE_URL`             | —                    | 接口地址（DashScope 兼容模式等）          |
| `VLM_MAX_CONCURRENCY`                          | `4`                  | VLM 在途并发上限                     |
| `OCR_MAX_WORKERS`                              | `4`                  | OCR 并发上限                       |
| `OCR_SERIALIZE`                                | `true`               | RapidOCR 引擎调用是否加锁串行            |
| `CACHE_BACKEND`                                | `auto`               | `auto`/`redis`/`memory`/`none` |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | `localhost`/`6379`/空 | Redis                          |
| `CACHE_TTL`                                    | `604800`             | 缓存秒数                           |
| `PDFIMG_API_KEY`                               | 空                    | 留空=不鉴权                         |
| `PDFIMG_PORT`                                  | `8000`               | 监听端口                           |
| `MAX_UPLOAD_MB`                                | `16`                 | 单文件上传上限                        |
| `MAX_BATCH_SIZE`                               | `32`                 | 单次批量请求最多图片数                    |
| `BATCH_MAX_CONCURRENCY`                        | `8`                  | 批量内部并发上限                       |


---



## 八、测试

```bash
pip install -e ".[dev]"
pytest -q
```

覆盖：规则打分、路由决策表、缓存命中、执行器超时/重试/非重试错误/降级/hybrid 融合、
服务健康检查/端点/鉴权/上传限制/指标。测试用假 OCR/VLM，不触网、不依赖 Redis。

---



## 九、部署

```bash
# Docker + Redis 一键（在 deploy/ 下）
docker compose -f deploy/docker-compose.yml up -d --build

# 或用户级 systemd（改好 deploy/pdfimgpipeline.service 里的路径）
cp deploy/pdfimgpipeline.service ~/.config/systemd/user/
systemctl --user enable --now pdfimgpipeline
```

生产注意：

- **不要开多个 uvicorn worker**：RapidOCR 引擎是进程内单例，多 worker 会各自复制模型；
扩容请起多个副本 + 负载均衡。
- 缓存用 Redis 时是**多副本共享**的；用内存缓存则各副本独立。
- VLM 昂贵：建议调用方先 `/route` 预筛，再对需要的图 `/extract`。

---



## 十、作为库使用

```python
import asyncio
from pdfimgpipeline import Settings, build_pipeline

async def main():
    settings = Settings()
    pipeline, cache = await build_pipeline(settings)
    try:
        # 单张
        out = await pipeline.process("sample.png")
        # 批量：并发 8，单项失败隔离
        outcomes = await pipeline.process_many(
            ["a.png", "b.png", "c.png"], max_concurrency=8)
        for o in outcomes:
            print(o.ok, o.data, o.error)
    finally:
        await cache.aclose()

asyncio.run(main())
```

也可自行注入 OCR/VLM 实现：

```python
from pdfimgpipeline import ExtractorExecutor, PDFImagePipeline

executor = ExtractorExecutor(my_async_ocr, my_async_vlm, vlm_timeout=30)
```

---



## 开发文档

```
http://127.0.0.1:8000/docs
```



## 十一、已知限制

- **Python 版本**：`rapidocr-onnxruntime` 目前要求 Python < 3.13，因此在 3.13 上安装会跳过它，
OCR 通道不可用（探测返回 0，路由偏 VLM；执行降级为结构化错误）。**请用 Python 3.11 / 3.12**。
- **onnxruntime 版本**：1.22+ 的 macOS wheel 要求 macOS 13.4+。本机 macOS 13.0 已钉在
`onnxruntime>=1.17,<1.22`（实测 1.20.1 可用）。新机器若报 `Symbol not found` 或
`_ARRAY_API not found`，按此调整。
- RapidOCR 引擎调用默认串行化（`OCR_SERIALIZE=true`）以保证线程安全；确认可并发后可关闭。
- `to_thread` 派发的同步调用超时后无法强杀。
- hybrid 模式 OCR 与 VLM 双跑，成本最高。

