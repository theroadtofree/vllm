"""Lwd 控制面通知:仅依赖 msgspec,不携带 tensor,只承载调度决策预告。
上游输出类型经本模块 re-export,其余内核文件不得直接 import vllm.v1.engine。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import msgspec

if TYPE_CHECKING:
    # 上游输出类型 re-export:step 载体的返回注解共用,避免各文件直连上游
    from vllm.v1.engine import EngineCoreOutputs  # noqa: F401


class LwdRangeItem(msgspec.Struct, gc=False):
    """批量范围预告条目(mixed 组批);offset = 本 chunk 在 prompt 中的
    起点(chunk 支持已启用,lwd_chunked_prefill_design.md),云侧按
    offset == num_computed 对账。chunk 为 min(剩余 prompt, chunk 单元)
    的自限块;cap 截断块独占一批。

    has_mrope(pdmix 多模态):逐请求标记本 chunk 是否携带 mrope
    positions 行(UP aux 帧,[n,3] int64 与主帧同 seqno)。混批可含
    文本+图像混合,批级标记不够;纯文本恒 False,零 aux 流量。"""

    request_id: str
    offset: int
    num_tokens: int
    has_mrope: bool = False


class LwdRangeNotify(msgspec.Struct, gc=False, tag=True):
    """边->云调度范围预告(PRE_OUT);offset/num_tokens 取自原生调度决策,
    重复预告按 (request_id, offset) 幂等登记(边侧队满重试天然产生重复)。

    items(mixed 组批):本批全部条目,一个批一个 seqno;空 = 旧版单请求
    语义(顶层 request_id/offset/num_tokens)。新版边侧恒填 items 且顶层
    字段 = 首条目;新云侧优先读 items。"""

    request_id: str
    offset: int
    num_tokens: int
    seqno: int
    items: list[LwdRangeItem] = []


class LwdRequestNotify(msgspec.Struct, gc=False, tag=True):
    """边->云请求预告(PRE_OUT),线上只带调度决策所需字段;block_hashes 为
    边侧算好的 prompt 满块链,供云侧前缀缓存命中,缺省空回退本地哈希。

    采样参数透传(additive,缺省 = SamplingParams 原生缺省):只带影响
    云侧 token 选择的字段(采样核/惩罚/EOS 策略/min_tokens);stop 字符串
    等 detokenizer 层参数留在边侧前端原生处理,不上 wire。"""

    request_id: str
    num_prompt_tokens: int
    max_tokens: int = 16
    block_hashes: list[bytes] = []
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    stop_token_ids: list[int] = []
    min_tokens: int = 0
    eos_token_id: Optional[int] = None
    """结束符 id:边侧前端自 tokenizer 解析,云侧占位 prompt 无从
    得知,必须随预告透传;缺省 None = 云侧 EOS 判定落空,仅
    max_tokens 兜底(旧版边侧,additive 兼容)。"""


class LwdAbortNotify(msgspec.Struct, gc=False, tag=True):
    """边->云 abort 预告(PRE_OUT);云侧清理请求登记。"""

    request_id: str


class LwdHelloNotify(msgspec.Struct, gc=False, tag=True):
    """云->边发现通告(POST_OUT,周期发送:心跳即发现);pre_out_* 是边侧
    连接云端点的唯一事实源,pre_out_host 须为边可路由真实 IP
    (0.0.0.0 不可作通告值)。

    周期语义:边(重)启动后等下一拍即重新发现;边侧超过阈值未收到即判
    云失联(见 edge engine HELLO 静默计时)。

    调度三元组(max_num_batched_tokens/max_num_seqs/num_speculative_
    tokens)供边侧互校:三者是 cap/preflight 公式两侧的同源输入,
    不一致不会让链路报错,只会让 preflight 恒 hold、prefill 被 decode
    静默串行(真机实录:边 factor=1 云 factor=4 时 chunk 8191 +
    decode 4 > 预算 8192)。0 = 旧版未携带,边侧仅告警不拦截。"""

    pre_out_host: str
    pre_out_port: int
    max_num_batched_tokens: int = 0
    max_num_seqs: int = 0
    num_speculative_tokens: int = 0
    data_plane_host: str = ""
    """数据面(AF_SMC)端点通告:云侧 bind 的 host。边侧与本地装配配置
    比对,不一致仅告警——建连以本地配置为准(静态轨建连、HELLO 动态
    轨校验的双轨设计,同 pre_out 的既有模式)。缺省 "" = 旧版云侧
    未携带(msgspec 带默认字段,线上 additive 兼容)。"""
    data_plane_port_base: int = 0
    """数据面端口基址:UP=base, DOWN=base+1;0 = 未携带。"""
    cloud_boot_id: int = 0
    """云端点每次进程拉起的随机 id(周期通告捎带):值变化即"云服务重启
    过",边侧日志留痕(与数据面 ucx_wire 的 peer_session 同思路)。
    缺省 0 = 旧版云侧未携带(msgspec 带默认字段,additive 兼容)。"""


# 完成码哨兵:finish_reasons 中的"本步未终结"值
LWD_NOT_FINISHED = -1


class LwdC2eNotify(msgspec.Struct, gc=False, tag=True):
    """云->边步元数据通告(POST_OUT),先于隐藏张量到达,边侧据此预挂精确
    尺寸 recv;hidden_num_elements=0 为纯终结通告,边侧本地终结不派发 unembed。"""

    hidden_num_elements: int
    top_id_ths: list[list[int]]
    num_accepted_tokens: list[int]
    req_ids: list[str]
    finish_reasons: list[int] = []
    """逐请求完成码,与 req_ids 按位对齐:LWD_NOT_FINISHED(-1) = 本步未
    终结;否则为 FinishReason IntEnum 值(STOP=0/LENGTH=1/ABORT=2/ERROR=3/
    REPETITION=4),边侧原样透传 finish_reason(LENGTH 不再伪装成 STOP)。
    空列表 = 云侧未携带,错配即 IndexError fail-fast。"""
    down_seqno: int = -1
    """本步 DOWN 隐藏张量的通道序号,自 LwdC2eMeta.down_seqno 原样透传:
    云 worker 发送时分配(通道级单调),边侧按此值预挂配对 irecv;
    缺省 -1 = 旧版云侧未携带(msgspec 带默认字段,线上 additive 兼容)。"""
    token_ids: list[list[int]] = []
    """诊断旁路载荷:云侧逐请求 accepted token ids(仅
    VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE=1 时填充),与 req_ids 按位对齐。
    空列表 = 未携带,边侧走原生 rank-replay(lm_head + top_id_ths)。"""


# typing.Union 而非 PEP 604 `|`:msgspec 解码器的全版本支持路径
LwdNotify = Union[LwdRangeNotify, LwdRequestNotify, LwdAbortNotify]  # noqa: UP007
# 云->边方向(POST_OUT):HELLO 发现 + 步元数据(唯一载荷,兼结果回传
# 驱动);重同步消息在此 union 上 additive 扩展
LwdCloudNotify = Union[LwdHelloNotify, LwdC2eNotify]  # noqa: UP007
# 数据面批型定义归 vllm/v1/core/sched/output.py(LwdBatch/LwdBatchType/
# LwdEmbedBatch/LwdUnembedBatch,f8182fd5 定稿),协议层不重复声明

_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdNotify)
_CLOUD_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdCloudNotify)


def lwd_encode_notify(message: LwdNotify) -> bytes:
    """通知编码为传输字节(发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_notify(data: bytes) -> LwdNotify:
    """传输字节解码为通知(订阅端唯一入口);坏包抛异常,由接收线程捕获丢弃。"""
    return _NOTIFY_DECODER.decode(data)


def lwd_encode_cloud_notify(message: LwdCloudNotify) -> bytes:
    """云->边通知编码(POST_OUT 发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_cloud_notify(data: bytes) -> LwdCloudNotify:
    """云->边通知解码(POST_OUT 订阅端唯一入口);坏包抛异常由接收方丢弃。"""
    return _CLOUD_NOTIFY_DECODER.decode(data)
