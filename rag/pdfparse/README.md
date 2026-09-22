# 02-rag — 药品说明书版面解析

针对 CDE 药品说明书（规整单栏版式）的规则式版面解析，把 PDF 解析为
**章节 / 段落 / 表格 / 图片** 结构，并切成 RAG 可用的 chunk。解析结果全部落盘，可逐项检查。

> **v2：PDF 预检 + 路由分流**。新增 `pdfagent/` 分层（参考 `../pdfimgpipeline`）。
> 解析前先做一次毫秒级预检，按类型选引擎，并把路由决策与结果写入带版本号的缓存：
>
> ```
> PDF ──▶ 预检(逐页文本层 vs 图片覆盖)
>           ├─ text   （文本型）  ──▶ PyMuPDF 规则式解析（快、零额外依赖）
>           ├─ hybrid （混合型）  ──▶ MinerU 版式解析
>           └─ scanned（扫描型）  ──▶ MinerU 版式解析
>                     │
>                     └─ 图片 ──curl──▶ pdfimgpipeline /extract:batch（OCR + VLM）
> ```
>
> - **预检**（`pdfagent/precheck.py`）：逐页统计文本层字符数与图片覆盖面积，聚合判定
>   `text / hybrid / scanned`，带可解释 `reason`；`--show-precheck` 可看每页明细。
> - **路由 + 版本化缓存**（`pdfagent/router.py` + `pdfagent/cache.py`）：路由决策与最终
>   结果都进缓存，键为 **PDF 内容 SHA-256**，命名空间带**版本号**
>   `<ns>:<version>:<kind>:<hash>`。改 `pdfagent.__version__` / `SCHEMA_VERSION` 或设置
>   `PDF_CACHE_VERSION` 即让旧缓存自然失效。Redis 不可用时自动降级进程内内存缓存。
> - **图片语义化**（`describe_images.py`）：用 **curl** 调 `pdfimgpipeline` 的
>   `/extract:batch`，**整篇图一次批量请求**（服务端并发 + 缓存，单张失败隔离），
>   批量失败自动回退逐张。需先启动 pdfimgpipeline 服务。

## 用法

```bash
# 文本型 PDF：自动预检 -> pymupdf
.venv/bin/python main.py "../rag/_samples/达格列净片 （JXHS2200079-80）说明书.pdf"

# 扫描 / 混合型 PDF：自动预检 -> mineru（需先安装 MinerU）
.venv/bin/python main.py "<扫描版.pdf>"

# 图片语义化：curl 调 pdfimgpipeline（整篇图批量）
.venv/bin/python main.py "<pdf>" --describe

# 常用开关：--route auto|text|mineru  --fallback-text  --no-cache  --show-precheck
.venv/bin/python main.py <pdf路径> [-o out] [--chunk-limit 600] [--describe] [--show-precheck]

# 依赖：pymupdf（必装）；redis（可选，未装/未启动自动降级内存缓存）
pip install -r requirement.txt
```

配置见 `.env.example`（`PDFIMG_BASE_URL` / `REDIS_HOST` / MinerU 参数等）。

## pdfagent 目录结构（预检 / 路由层）

```
pdfagent/
├── config.py         # 集中配置（env / .env，纯标准库，不强制 pydantic）
├── models.py         # PdfKind / PageFeature / PrecheckResult / RouteResult / BatchOutcome
├── precheck.py       # 逐页文本覆盖率 -> text / hybrid / scanned
├── cache.py          # 可插拔缓存（redis/memory/none）+ 版本号命名空间
├── router.py         # 预检 -> pymupdf / mineru 决策，带缓存
├── image_client.py   # curl 调 pdfimgpipeline（单张 + /extract:batch 批量）
├── pipeline.py       # 顶层入口：路由 -> 解析 -> 缓存
├── wiring.py         # Settings -> pipeline 依赖装配
└── adapters/
    ├── pymupdf_layout.py  # 包装根目录 layout_parser
    └── mineru.py          # MinerU CLI / HTTP，输出归一化
```

## 输出物（out/）

| 文件 | 内容 |
| --- | --- |
| `layout.json` | 完整结构化结果：文档元数据、章节树、表格、图片、chunk 列表、元素框坐标 |
| `chunks.md` | 全部 chunk 的可读清单（id / 类型 / 章节路径 / 页码 / 字数 / 正文） |
| `tables/table-XX.md` | 每个表格的 Markdown（含表题） |
| `figures/figure-XX.*` | 提取的图片原图 |
| `pages/page-XXX.png` | 逐页标注图：按元素类型画彩色框（heading/subhead/caption/para/table/figure） |
| `report.html` | 汇总报告：概览、章节结构、表格、图片、chunk、标注页面，浏览器打开即可 |

检查解析质量的最快路径：打开 `report.html`，或直接翻 `pages/` 标注图对照原文。

## 下一步：chunk → 向量化（embed_demo.py）

**需要 embedding 的是 `out/layout.json` 里 `chunks` 数组（`chunks.md` 是它的可读渲染）**——
版面解析器只负责切 chunk（已完成），真正要做向量的是每条 chunk 的正文 `content`。
本仓库不带向量库，配套演示脚本 `embed_demo.py` 把 chunk 编码成 1024 维向量并落盘检索：

```bash
python3 embed_demo.py                     # 建库(vec_store/) + 4 个示例问题检索
python3 embed_demo.py query "推荐起始剂量是多少？" -k 5
python3 embed_demo.py index --backend service   # 连 ../embedding 的常驻 bge-m3 服务
```

向量化规则：表格/图 chunk 直接编码 `content`（已前置表题/图题）；文本 chunk 默认拼上
「章节/小节」前缀；只有图片路径、没有任何正文的图 chunk（无图题且未做 OCR/VLM）
没有可嵌入语义，建库时跳过。
模型默认 `BAAI/bge-m3`（与 `../embedding` 同款，首次需下载 ~2.3GB）。

### 图片语义化入口（OCR + VLM）—— 给图也补上“文字”

纯图片本身不进 embedding（没有能看图的模型）。本项目**不本地实现** OCR/VLM，而是把图
交给独立的 `pdfimgpipeline` HTTP 服务，`describe_images.py` 用 curl 调用：

1. **整篇图批量**：解析器先把每张图落盘，再一次性调 `/extract:batch`；
2. **归一化**：服务端返回的 `ocr` / `vlm` / `hybrid` 结果转换成
   `ocr_text` + `vlm_description`，拼进图 chunk 的 `content`（layout.json /
   chunks.md / report.html 立即可见），embed_demo 建库时自动把这类图一并向量化。

```bash
# 先启动 pdfimgpipeline（另开终端）
cd ../pdfimgpipeline && pdfimgpipeline
# 再解析并开启图语义化
.venv/bin/python main.py "<pdf>" --describe
```

服务不可用时不会中断解析：每张图记 `description_error`，正文只保留图题/路径。

## 解析规则

1. **章节切分**：`【...】` 独占一行的行 → 一级章节标题。
   说明书全文统一 12pt 宋体，标题无字体特征，只能靠正则；
   PyMuPDF 有时把标题行并进相邻文本块，因此在**行级别**识别标题。
2. **小节标题 / 表图题**：加粗短行 → 小节标题（如「低血糖症」），记入 chunk 的
   `sub_section`；加粗且以「表N/图N」开头 → 表图题，挂到相邻表格/图片
   （表题在表上方，图题在图下方）。
3. **表格**：`find_tables(vertical_strategy="lines_strict", horizontal_strategy="lines_strict")`
   只按明确框线检测——默认策略会把无边框的对齐段落误判成表格（本文件 p28 曾误检 5 个）。
   表格区域从正文流剔除，避免表格文字被打散进段落。
4. **断表合并**：相邻且无其他元素间隔的同列数表格合并（跨页续表去重复表头，
   匹配前 3 行表头）；带表题的跨页表头碎片（列数与表体不一致）单独放行合并，
   如表16 表头在 p28、表体在 p29。
5. **chunk 策略**：短章节整章一个 chunk；长章节按小节标题/段落累积到
   `--chunk-limit`（默认 600 字）；表格、图片始终是原子 chunk，
   内容前置「表题/图题」保证脱离上下文可检索。
6. **元数据**：从文档头解析核准/修改日期，从【药品名称】解析通用名/商品名/英文名，
   从文件名解析受理号，注入每个 chunk 的 `meta` 字段。

## 已知取舍

- 多行合并表头（如「达格列净5mg / N=1145」两行叠加）在 Markdown 里展开为多行，
  第一行作表头——语义保留但列归属需人工或 LLM 后处理。
- 无框线表格（纯三线表且无竖线）会被 `lines_strict` 漏检；本文件所有表格均有框线，未遇到。
- 图片内容（像素）不进 embedding：默认只提取+挂图题；需要语义化时用 `--describe` 接入
  pdfimgpipeline（OCR + VLM，见上节）。
- 文本型走 PyMuPDF；扫描/混合型自动分流到 MinerU（需本机安装 `mineru`/`magic-pdf`
  或配置 `MINERU_HTTP_URL`）。MinerU 不可用时默认直接报错，`--fallback-text` 可降级 pymupdf。
- 结果缓存包含 `out_dir`，换输出目录不会命中（避免图片路径指向旧目录）；`--no-cache` 可跳过。

## 文件

- `pdfagent/` — **预检 / 路由 / 缓存 / 图片客户端** 分层（见上）
- `layout_parser.py` — PyMuPDF 解析核心（`parse_document(..., image_describer=...)` 图语义化钩子，支持批量）
- `visualize.py` — 标注页面 PNG、chunks.md、report.html
- `main.py` — CLI 入口（`--describe` 图片语义化、`--route` 强制路由、`--show-precheck`）
- `describe_images.py` — 图片语义化：curl 调 pdfimgpipeline `/extract:batch`（`PdfImgPipelineDescriber`）
- `tests/test_pdfagent.py` — 预检/路由/缓存/curl 客户端/Pipeline 离线测试
  （`.venv/bin/python -m unittest discover -s tests -v`）
- `.env.example` — 预检阈值 / 缓存 / MinerU / pdfimgpipeline 配置示例
- `embed_demo.py` — 向量化 demo：chunk → embedding(bge-m3) → 向量库 → top-k 检索（不在 main.py 流程内，独立运行）
