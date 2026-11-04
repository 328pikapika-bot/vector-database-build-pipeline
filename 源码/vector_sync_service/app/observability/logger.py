"""
NSTL 向量库同步服务 - 结构化日志
基于 structlog，JSON 格式输出，支持上下文变量
"""
import structlog
import logging
import re


def _mask_api_key(_, __, event_dict):
    """脱敏：移除日志中的 API Key"""
    if "api_key" in event_dict:
        event_dict["api_key"] = "***"
    # 脱敏 error 消息中可能出现的 key
    for key in ("error", "msg", "errorMsg"):
        if key in event_dict and isinstance(event_dict[key], str):
            event_dict[key] = re.sub(
                r"(api_key|apikey|authorization|token)[=:\\s]*[^\\s,;]+",
                r"\1=***",
                event_dict[key],
                flags=re.IGNORECASE,
            )
    return event_dict


def setup_logging():
    """初始化结构化日志"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _mask_api_key,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 配置标准 logging 输出 JSON 格式
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer()
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """获取结构化日志器"""
    return structlog.get_logger(name)


# 模块级日志器
logger = structlog.get_logger(__name__)
