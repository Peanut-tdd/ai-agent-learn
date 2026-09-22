# pdfimgpipeline —— 独立服务镜像
#
# 构建（在项目根目录）:
#   docker build -f deploy/Dockerfile -t pdfimgpipeline .
# 运行:
#   docker run -d --name pdfimg -p 8000:8000 \
#     -e REDIS_HOST=host.docker.internal \
#     -e VLM_API_KEY=sk-xxx -e VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
#     pdfimgpipeline
# 探活:
#   curl localhost:8000/health
#
# 说明: 未加 --workers，RapidOCR 引擎是进程内单例；扩容请起多个副本做负载均衡。

FROM python:3.11-slim

# opencv / onnxruntime 运行期系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY pyproject.toml readme.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV PDFIMG_HOST=0.0.0.0 \
    PDFIMG_PORT=8000 \
    CACHE_BACKEND=auto

EXPOSE 8000

CMD ["uvicorn", "pdfimgpipeline.service.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
