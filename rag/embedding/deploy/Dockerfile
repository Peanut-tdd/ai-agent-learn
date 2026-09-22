# bge-m3 embedding service —— 镜像示例
#
# 构建:  docker build -t bge-embed .
# 运行(CPU, 模型挂载自宿主机缓存, 不重新下载 2.1GB):
#   docker run -d --name bge-embed -p 8001:8001 \
#     -e HF_HUB_OFFLINE=1 -e EMBED_API_KEY=change-me \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     bge-embed
# 探活(容器编排/健康检查, 注意嵌入口令用于 /embed):
#   curl localhost:8001/health
#
# ⚠️ GPU: 默认装的是 CPU 版 torch, 推理慢。
#    GPU 部署时把基础镜像换成 nvidia/cuda 并装带 cu 的 torch:
#     FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
#     然后 pip install torch --index-url https://download.pytorch.org/whl/cu124
#     运行加 --gpus all; 显存不足调小 EMBED_INFER_BATCH。
# ⚠️ 多副本: 不要在一个容器里 --workers>1(每进程复制一份 ~2.3GB 模型);
#    横向扩多个容器, 前面用 nginx / 服务网格做负载均衡。

FROM python:3.11-slim

WORKDIR /srv

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY server.py client.py ./

ENV EMBED_HOST=0.0.0.0 \
    EMBED_PORT=8001 \
    HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8001

# 模型启动时加载(首次自动下载到容器内 ~/.cache/huggingface, 建议挂载持久化)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--no-access-log"]
