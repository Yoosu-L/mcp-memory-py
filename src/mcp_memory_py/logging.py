import logging
import json
import time
import os
import functools
from pathlib import Path
from datetime import datetime
from typing import Callable, TypeVar, ParamSpec

logger = logging.getLogger("mcp_memory")

# Only set up file logging if DEBUG_LOGGING is enabled
if os.getenv("DEBUG_LOGGING", "").lower() == "true":
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"mcp_server_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
else:
    # Disable logging when DEBUG_LOGGING is not enabled
    logger.addHandler(logging.NullHandler())

# Type variables for decorator
P = ParamSpec("P")
R = TypeVar("R")


def log_tool_call(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to log tool calls with timing and results."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.time()
        tool_name = func.__name__

        # Log call
        try:
            call_args = {
                "args": [str(arg) for arg in args[1:]],  # Skip self
                "kwargs": {k: str(v) for k, v in kwargs.items()},
            }
            logger.info(f"Tool call: {tool_name} - Args: {json.dumps(call_args)}")

            # Execute tool
            result = await func(*args, **kwargs)

            # Log success
            execution_time = time.time() - start_time
            logger.info(
                f"Tool success: {tool_name} - "
                f"Time: {execution_time:.2f}s - "
                f"Result: {str(result)[:200]}..."  # Truncate long results
            )
            return result

        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            logger.error(
                f"Tool error: {tool_name} - "
                f"Time: {execution_time:.2f}s - "
                f"Error: {str(e)}"
            )
            raise

    return wrapper
