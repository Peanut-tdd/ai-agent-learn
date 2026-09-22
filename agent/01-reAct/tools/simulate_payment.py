"""演示工具：非幂等扣款 + 幂等下单 + 查询账本。

用于验证写操作的两种重试语义：
- charge_payment：非幂等写（at-most-once）——每次调用都会在模拟账本中落账，
  然后抛超时异常，模拟"请求已到达服务端、响应丢失"。结果未知，绝不允许重复调用。
- create_order：幂等写——order_id 是幂等键，第一次调用落账后超时，
  代码层用同一 order_id 自动重试时幂等命中，不会重复创建。
- query_payment：查询类，用于确认扣款是否真的发生。
"""

from pydantic import BaseModel, Field

# ── 模拟服务端状态（进程内）──────────────────────────────────────────────────
_LEDGER: dict[str, float] = {}  # 非幂等扣款账本
_ORDERS: set[str] = set()       # 幂等订单集合


class ChargeArgs(BaseModel):
    user: str = Field(..., min_length=1, description="用户标识")
    amount: float = Field(..., gt=0, description="扣款金额")


class CreateOrderArgs(BaseModel):
    order_id: str = Field(..., min_length=1, description="幂等键：调用方生成的唯一订单号（重试时必须保持不变）")
    item: str = Field(..., min_length=1, description="商品名")
    qty: int = Field(..., gt=0, description="数量")


class QueryArgs(BaseModel):
    user: str = Field(..., min_length=1, description="用户标识")


def charge_payment(user: str, amount: float) -> str:
    """非幂等扣款：服务端先落账再抛超时（模拟响应丢失）。绝不允许盲目重试！"""
    _LEDGER[user] = _LEDGER.get(user, 0.0) + amount              # 钱已扣
    raise TimeoutError(f"扣款 {amount} 已提交但响应超时，结果未知")  # 响应丢失


def create_order(order_id: str, item: str, qty: int) -> str:
    """幂等下单：order_id 是幂等键。首次调用落账后超时，同 key 重试幂等命中。"""
    if order_id in _ORDERS:                                      # 幂等命中
        return f"订单已存在（幂等命中）: {order_id}"
    _ORDERS.add(order_id)                                        # 服务端已创建
    raise TimeoutError(f"订单 {order_id} 已提交但响应超时")         # 响应丢失


def query_payment(user: str) -> str:
    """查询账本：确认扣款是否真的发生。"""
    return f"用户 {user} 累计扣款: {_LEDGER.get(user, 0.0)}"
