"""传输层方向原语:OUTBOUND 发布端(side-agnostic);自有线程把有界队列的
消息编码发送,对谁发布一无所知;retarget 经队列命令由本类线程执行(亲和)。"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

import zmq

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    lwd_encode_notify,
)

logger = init_logger(__name__)

# 关停令:发布线程收到即退出;None 不会与通知混淆(它们是 msgspec 结构)
_LWD_PUBLISH_SHUTDOWN = None


class _LwdRetarget:
    """retarget 队列命令:换连接目标,由发布线程自己执行(线程亲和)。"""

    __slots__ = ("endpoint",)

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint


class _LwdPeriodic:
    """周期拍哨兵(队列空闲超时的本地替身):发送 periodic_msg。"""

    __slots__ = ()


LWD_PUBLISH_QUEUE_MAX = 1000


class LwdControlPublisher:
    """控制面发布端;publish 队满返回 False,调用方视为未派发、下一步重试。
    对端消费慢 -> send 阻塞 -> 有界队列涨满 -> publish 返回 False,天然背压。"""

    def __init__(
        self,
        endpoint: str | None,
        *,
        bind: bool,
        queue_max: int = LWD_PUBLISH_QUEUE_MAX,
        encoder: Callable[[Any], bytes] = lwd_encode_notify,
        periodic_msg: Any | None = None,
        periodic_interval_s: float = 0.0,
    ) -> None:
        """periodic_msg/periodic_interval_s: 周期自动消息(心跳即发现)。
        发送线程在队列空闲超时或持续满载超龄时自补一条——新拉起的对端
        只认发现通告(如 HELLO),真实流量不能替代它。间隔<=0 关闭。"""
        self._closed = False
        self._encoder = encoder
        self._queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self._periodic_msg = periodic_msg if periodic_interval_s > 0 else None
        self._periodic_interval_s = periodic_interval_s
        self._last_periodic = 0.0
        self._communicator = LwdControlCommunicator(endpoint, zmq.PUSH, bind=bind)
        self._thread = threading.Thread(
            target=self._send_thread, name="lwd-publisher", daemon=True
        )
        # 线程最后启动:全部自有状态就绪后才开始消费
        self._thread.start()

    @property
    def closed(self) -> bool:
        """已关停为 True(与 LwdControlSubscriber.closed 同语义)。"""
        return self._closed

    def publish(self, msg: Any) -> bool:
        """元数据入队;False = 队满未发(可预期失败,不抛异常不丢已发消息)。"""
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            return False
        return True

    def retarget(self, endpoint: str) -> bool:
        """换连接目标(队列命令,发布线程执行保亲和);队满返回 False,
        调用方保持旧目标,等下一条 HELLO 重试。"""
        if self._closed:
            return False
        try:
            self._queue.put_nowait(_LwdRetarget(endpoint))
        except queue.Full:
            logger.error("[Lwd] retarget dropped: publish queue full")
            return False
        return True

    def shutdown(self) -> None:
        """关停(幂等):关停令入队 -> join(2s) -> 残留线程以 term 收尾。"""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(queue.Full):
            self._queue.put(_LWD_PUBLISH_SHUTDOWN, timeout=1.0)
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            # 线程卡在阻塞 send:term 使其以 ETERM 退出,随后走 close(0) 收尾
            self._communicator.terminate()
            self._thread.join(timeout=1.0)

    def _send_thread(self) -> None:
        import time as _time
        self._last_periodic = _time.monotonic()
        while True:
            if self._periodic_msg is not None:
                # 周期模式:队列空闲一个间隔即自补心跳/发现通告
                try:
                    msg = self._queue.get(timeout=self._periodic_interval_s)
                except queue.Empty:
                    msg = _LwdPeriodic()
            else:
                msg = self._queue.get()
            if msg is _LWD_PUBLISH_SHUTDOWN:
                break
            if isinstance(msg, _LwdRetarget):
                self._communicator.retarget(msg.endpoint)
                continue
            if isinstance(msg, _LwdPeriodic):
                self._communicator.send(self._encoder(self._periodic_msg))
                self._last_periodic = _time.monotonic()
                continue
            # 无对端时 send 阻塞,背压由有界队列传导给 publish 返回值
            self._communicator.send(self._encoder(msg))
            # 持续满载防护:timeout 拍永不触发时发现通告会饿死(新拉起
            # 的对端只认 HELLO),超龄即补——真实流量不能替代发现语义
            if self._periodic_msg is not None:
                now = _time.monotonic()
                if now - self._last_periodic >= self._periodic_interval_s:
                    self._communicator.send(
                        self._encoder(self._periodic_msg))
                    self._last_periodic = now
        self._communicator.close()
