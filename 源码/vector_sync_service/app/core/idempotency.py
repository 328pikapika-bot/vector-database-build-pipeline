"""
幂等判定逻辑

两层幂等保障：
1. 批次内去重：同批次相同 docId 只保留 recordUpdateTime 最大的
2. 跨批次幂等：按 MongoDB 状态中的 recordUpdateTime 新旧判定
"""
from typing import List, Tuple, Optional
from app.models.kafka_event import KafkaEvent


def dedup_within_batch(events: List[KafkaEvent]) -> Tuple[List[KafkaEvent], int]:
    """
    批次内去重：同一批次中相同 docId 只保留 recordUpdateTime 最晚的那条

    Returns:
        (deduped_events, skip_count): 去重后的事件列表和被跳过的条数
    """
    if not events:
        return [], 0

    best: dict[str, KafkaEvent] = {}
    for event in events:
        existing = best.get(event.docId)
        if existing is None or event.recordUpdateTime > existing.recordUpdateTime:
            best[event.docId] = event

    deduped = list(best.values())
    skip_count = len(events) - len(deduped)
    return deduped, skip_count


def should_skip(
    existing_recordUpdateTime: Optional[str],
    event: KafkaEvent,
) -> bool:
    """
    跨批次幂等判定：若已有处理记录且其 recordUpdateTime >= 当前事件的 recordUpdateTime，
    则跳过处理（旧消息或重复消息）

    Args:
        existing_recordUpdateTime: MongoDB 中已记录的 recordUpdateTime (ISO 格式字符串或 None)
        event: 当前 Kafka 事件

    Returns:
        True 表示应跳过处理
    """
    if existing_recordUpdateTime is None:
        return False

    from datetime import datetime

    try:
        existing_time = datetime.fromisoformat(existing_recordUpdateTime)
        return existing_time >= event.recordUpdateTime
    except (ValueError, TypeError):
        # 无法解析已有时间，保守起见不跳过
        return False
