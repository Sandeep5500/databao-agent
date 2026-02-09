from typing import Any

from databao_context_engine import ContextSearchResult
from langchain_core.tools import BaseTool, tool

from databao.core import Context


def make_search_context_tool(context: Context) -> BaseTool | None:
    if not context.is_static:
        return None

    @tool(parse_docstring=True)
    def search_context(retrieve_text: str) -> list[dict[str, Any]]:
        """Search the context for relevant information matching the given query text.

        Args:
            retrieve_text: Natural language query to search the context for relevant results.
        """
        search_result_list = context.search_context(retrieve_text)
        return list(map(_search_result_to_dict, search_result_list))

    def _search_result_to_dict(search_result: ContextSearchResult) -> dict[str, Any]:
        result = {
            "distance": search_result.distance,
            "context_result": search_result.context_result,
        }
        ds_name = _get_ds_name(search_result)
        if ds_name is not None:
            result["data_source_name"] = ds_name
        return result

    def _get_ds_name(search_result: ContextSearchResult) -> str | None:
        ds_id = search_result.datasource_id
        ds = context.sources.configured.get(ds_id)
        return ds.name if ds is not None else None

    return search_context
