import os
import json
import numpy as np

# 彻底离线: 只用本地缓存 (~/.cache/huggingface), 不做任何远程校验请求
os.environ["HF_HUB_OFFLINE"] = "1"
# 若日后要联网下载新模型, 取消下一行注释并去掉 offline(国内直连 huggingface.co 会超时)
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences = ["什么是BGE M3？", "BGE M3是一个支持密集、稀疏和多向量检索的嵌入模型"]

Output = model.encode(
    sentences,
    return_dense=True,        # 出 dense_vecs:     整句语义向量 [N, 1024]
    return_sparse=True,       # 出 lexical_weights: 词/词片级权重 {token_id: w}
    return_colbert_vecs=False,  # 精排用, 学习阶段不开
    batch_size=32,
)

# ------------------------------------------------------------------
# 把 numpy 结果整理成"能落库"的形态(对照 server.py 的返回字段)
# ------------------------------------------------------------------
dense = Output["dense_vecs"].tolist()                       # → 向量库

# sparse 两形态:
sparse_ids = [{k: float(w) for k, w in d.items()}
              for d in Output["lexical_weights"]]           # {token_id: w}  喂 ES/OpenSearch rank_features
sparse_tokens = [{k: float(v) for k, v in d.items()}
                 for d in model.convert_id_to_token(
                     Output["lexical_weights"])]            # {词: w}        人看/调试

payload = {
    "dense": dense,
    "sparse_ids": sparse_ids,
    "sparse_tokens": sparse_tokens,
}

# 全部已是 Python 原生类型(str/float/list), 直接 json.dumps 即可
print(json.dumps(payload, ensure_ascii=False, indent=2))
