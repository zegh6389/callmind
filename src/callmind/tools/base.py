from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    success: bool
    summary: str
    data: dict = field(default_factory=dict)
    error: str = ""


class ToolError(Exception):
    """Raised by a tool to abort execution with a structured error."""


class Tool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    async def run(
        self,
        params: dict,
        *,
        call_id: str,
        business_id: str,
    ) -> ToolResult:
        """Execute the tool. Must not raise; return ToolResult on failure."""