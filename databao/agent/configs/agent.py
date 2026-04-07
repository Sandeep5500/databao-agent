from pydantic import BaseModel, ConfigDict


class AgentConfig(BaseModel):
    """Configuration for databao agent."""

    recursion_limit: int = 50
    """Maximum recursion depth for LLM agent execution."""

    min_retrievals: int = 0
    """Minimum number of search_context calls the agent must make before submitting."""

    parallel_tool_calls: bool = True
    """Whether agent is allowed to call several tools in one response.
    Supported by OpenAI models only."""

    model_config = ConfigDict(frozen=True, extra="forbid")


DEFAULT_AGENT_CONFIG = AgentConfig()
