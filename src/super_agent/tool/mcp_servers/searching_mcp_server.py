import os
import traceback
import requests
import datetime
import calendar
import json
import urllib.parse
from fastmcp import FastMCP
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters  # (already imported in config.py)
import wikipedia
import asyncio
import httpx 
from .utils.smart_request import smart_request, api_request_json
from src.super_agent.tool.logger import bootstrap_logger
from typing import List, Optional, Dict, Any
from .utils.perplexity import PerplexitySearch
from .utils.query_enhancer import enhance_query_for_perplexity
from .utils.search_content_judge import search_content_judge
from dataclasses import dataclass, field
import time

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
# JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
# Initialize FastMCP server
mcp = FastMCP("searching-mcp-server")

logger = bootstrap_logger()
# change the default engine timeout based on how long JINA can usually take to return a result 
DEFAULT_ENGINE_TIMEOUT = 600.0

@dataclass
class SearchResult:
    """search results from a specific engine"""
    engine: str
    query: str
    content: str
    success: bool
    error: Optional[str] = None



# async def jina_deep_search(query: str, timeout: float = DEFAULT_ENGINE_TIMEOUT):
#     url = "https://deepsearch.jina.ai/v1/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {JINA_API_KEY}"
#     }

#     data = {
#         "model": "jina-deepsearch-v1",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": query
#             }
#         ],
#         "stream": False,
#         "reasoning_effort": "low",
#         "team_size": 4,
#         "no_direct_answer": True,

#     }
#     try:
#         async with httpx.AsyncClient(timeout=timeout) as client:
#             response = await client.post(url, headers=headers, json=data)
#         response.raise_for_status()
#         return SearchResult("JINA", query, response.json()['choices'][0]['message']['content'], True)

#     # httpx raises TimeoutException; catching httpx.Timeout (a config class) triggers
#     # "catching classes that do not inherit from BaseException" in some environments.
#     except httpx.TimeoutException:
#         return SearchResult("JINA", query, "Request timed out", False)
#     except httpx.RequestError as e:
#         return SearchResult("JINA", query, f"Request failed: {e}", False)
#     except asyncio.CancelledError:
#         logger.warning("JINA deep search cancelled")
#         raise
#     except Exception as e:
#         return SearchResult("JINA", query, f"Unexpected error: {e}", False)


async def perplexity_search(
    query: str,
    search_recency_filter: Optional[str] = None,
    search_context_size: str = "high",
    max_tokens: int = 4096,    
    call_timeout: float = DEFAULT_ENGINE_TIMEOUT,

) -> str:
    """Perform a comprehensive web search using Perplexity AI with real-time information and citations.
    
    Perplexity leverages sophisticated AI to interpret your question, ensuring it knows exactly what you're asking. It searches the internet, gathering information from authoritative sources like articles, websites, and journals. Perplexity compiles the most relevant insights into a coherent, easy-to-understand answer.
    
    Args:
        query: a natural language question about the web
        search_recency_filter: Time filter for results ('day', 'week', 'month', 'year')
        search_context_size: Amount of context to retrieve ('low', 'medium', 'high')
        max_tokens: Maximum tokens in response (default: 4096)
        
    Returns:
        A coherent, easy-to-understand answer. with numbered citations and source links
    """
    logger.info(f"🌐 Perplexity generalweb search tool called - Query: {query[:100]}..., Recency: {search_recency_filter}")
    start_time = time.time()
    
    if not PERPLEXITY_API_KEY:
        logger.error("❌ PERPLEXITY_API_KEY not configured")
        return "[ERROR]: PERPLEXITY_API_KEY is not set, perplexity_web_search tool is not available."
    
    try:
        # Enhance and classify the query
        logger.info("🧠 Enhancing query with AI classification...")
        enhancement_result = enhance_query_for_perplexity(query)
        
        # Log enhancement results
        logger.info(f"📝 Query Enhancement Results:")
        logger.info(f"   - Original: {query[:100]}...")
        logger.info(f"   - Enhanced: {enhancement_result['enhanced_query'][:100]}...")
        logger.info(f"   - Type: {enhancement_result['query_type']} ({enhancement_result['perplexity_model']})")
        logger.info(f"   - Topic: {enhancement_result['query_topic']} ({enhancement_result['search_type']} search)")
        
        # Initialize Perplexity with the recommended model
        model = enhancement_result['perplexity_model']
        logger.info(f"🔍 Initializing Perplexity search with {model} model...")
        searcher = PerplexitySearch(model=model)
        
        # Perform search based on classification
        logger.info(f"📡 Performing {enhancement_result['search_type']} search...")
        async def _run_search() -> str:
            if enhancement_result['search_type'] == 'academic':
                return await asyncio.to_thread(
                    searcher.academic_search,
                    enhancement_result['enhanced_query'],
                    max_tokens=max_tokens,
                )
            return await asyncio.to_thread(
                searcher.web_search,
                enhancement_result['enhanced_query'],
                search_recency_filter=search_recency_filter,
                search_context_size=search_context_size,
                max_tokens=max_tokens,
            )

        result = await asyncio.wait_for(_run_search(), timeout=call_timeout)

        elapsed_time = time.time() - start_time
        logger.info(f"✅ Perplexity web search completed in {elapsed_time:.2f}s - Length: {len(result)} chars")
        return SearchResult("Perplexity", query, result, True)  # Success, exit retry loop
    except ValueError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Perplexity web search validation error in {elapsed_time:.2f}s: {e}")
        # return f"[ERROR]: Perplexity web search failed: {str(e)}"
        return SearchResult("Perplexity", query, f"[ERROR]: Perplexity web search failed: {str(e)}", False, str(e))
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Perplexity web search timed out in {elapsed_time:.2f}s")
        return SearchResult("Perplexity", query, "Request timed out", False, "timeout")
    except asyncio.CancelledError:
        logger.warning("Perplexity search cancelled")
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Perplexity web search unexpected error in {elapsed_time:.2f}s: {e}")
        # return f"[ERROR]: Unexpected error in perplexity_web_search: {str(e)}"
        return SearchResult("Perplexity", query, f"[ERROR]: Unexpected error in perplexity_search: {str(e)}", False, str(e))

async def google_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str = None,
    num: int = 10,
    tbs: str = None,
    page: int = 1,
) -> str:
    """Perform google searches via Serper API and retrieve rich results.
    It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.

    Args:
        q: Search query string.
        location: Location for search results (e.g., 'SoHo, New York, United States', 'California, United States').
        num: The number of results to return (default: 10).
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year).
        page: The page number of results to return (default: 1).

    Returns:
        The search results.
    """
    if not SERPER_API_KEY:
        return "SERPER_API_KEY is not set, google_search tool is not available."

    if not q or not q.strip():
        return "[ERROR]: Search query 'q' cannot be empty."

    serper_tool_name = "google_search"
    serper_search_arguments = {
        "q": q,
        "gl": gl,
        "hl": hl,
        "num": num,
        "page": page,
        "autocorrect": False,
    }
    if location:
        serper_search_arguments["location"] = location
    if tbs:
        serper_search_arguments["tbs"] = tbs
    serper_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "serper-search-scrape-mcp-server"],
        env={"SERPER_API_KEY": SERPER_API_KEY},
    )
    max_retries = 5
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            async with stdio_client(serper_server_params) as (read, write):
                async with ClientSession(
                    read, write, sampling_callback=None
                ) as session:
                    await session.initialize()
                    serper_tool_result = await session.call_tool(
                        serper_tool_name, arguments=serper_search_arguments
                    )
                    search_result_content = (
                        serper_tool_result.content[-1].text if serper_tool_result.content else ""
                    )
                    if not search_result_content or not search_result_content.strip():
                        raise ValueError(
                            "Empty result from google_search tool, please try again."
                        )
                    return search_result_content
        except Exception as error:
            last_error = error
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(2**attempt, 60))

    error_detail = f": {last_error}" if last_error else ""
    return f"[ERROR]: Tool execution failed after {max_retries} attempts{error_detail}"


async def search_all(query: str, engines: List[str] = None, **kwargs) -> List[SearchResult]:
    """search all engines in parallel and return results"""
    
    # 并行执行搜索
    tasks = [
        engine(query, **kwargs)
        for engine in engines
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(SearchResult(
                engine=engines[i],
                query=query,
                content="",
                success=False,
                error=str(result)
            ))
        else:
            processed_results.append(result)
    
    return processed_results


def format_search_results(results: List[SearchResult], query: str, include_metadata: bool = True) -> str:
    """format search results"""
    if not results:
        return f"No search results found for query: '{query}'"
    
    formatted_output = []
    successful_results = [r for r in results if r.success]
    # failed_results = [r for r in results if not r.success]
    
    if successful_results:
        for result in successful_results:
            formatted_output.append(f"### {result.engine.title()} Search Results:")
            formatted_output.append(result.content)
            formatted_output.append("\n" + "="*50 + "\n")
    
    # if failed_results:
    #     formatted_output.append("## Failed Searches\n")
    #     for result in failed_results:
    #         formatted_output.append(f"- **{result.engine.title()}**: {result.error}")
    #     formatted_output.append("")
    return "\n".join(formatted_output)


@mcp.tool()
async def general_search(
    query: str,
    parallel_search: bool = False,
    with_trust: bool = True,
    engine_timeout: float = DEFAULT_ENGINE_TIMEOUT,

) -> str:
    """Perform a deep search across multiple search engines (Perplexity AI, Google, Bing, DuckDuckGo) and return unified results.
    
    This tool searches across multiple engines and provides a unified, comprehensive response with results from all available sources.
    
    Args:
        query: a natural language search query
        
    Returns:
        Unified search results from all specified engines with performance metrics and citations
    """
    query = (query or "").strip()
    if not query:
        return "[ERROR]: Search query cannot be empty."
    
    # engines = [perplexity_search, jina_deep_search]
    engines = [perplexity_search]
    logger.info(f"🔍 General search called - Query: {query[:100]}..., Engines: {engines}")
    start_time = time.time()
    
    try:    
        if parallel_search:
            logger.info(f"🚀 Starting parallel search across {len(engines)} engines...")
            wrapped = [asyncio.wait_for(engine(query), timeout=engine_timeout) for engine in engines]
            raw_results = await asyncio.gather(*wrapped, return_exceptions=True)
            results = []
            for i, res in enumerate(raw_results):
                if isinstance(res, Exception):
                    results.append(
                        SearchResult(
                            engine=getattr(engines[i], "__name__", "unknown"),
                            query=query,
                            content="",
                            success=False,
                            error=str(res),
                        )
                    )
                else:
                    results.append(res)

        else:
            logger.info(f"⏭️ Starting sequential search across {len(engines)} engines...")
            results = []
            for engine in engines:
                logger.info(f"⌛️ {engine.__name__} started")
                try:
                    result = await asyncio.wait_for(engine(query), timeout=engine_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ {engine.__name__} timed out after {engine_timeout}s")
                    continue
                except asyncio.CancelledError:
                    logger.warning(f"🛑 {engine.__name__} cancelled")
                    raise
                except Exception as e:
                    # Log full traceback for easier debugging of engine-specific failures
                    logger.error(
                        "❌ %s failed with error: %s\n%s",
                        engine.__name__,
                        e,
                        traceback.format_exc(),
                    )
                    continue
                
                logger.info(f"✅ {result.engine.title()} completed")
                logger.info(f"🔍 search results: {result.content}")
                gpt_judge = await search_content_judge(query, result.content)
                if gpt_judge:
                    logger.info(f"✅ gpt-5 judge: {gpt_judge}")
                    results.append(result)
                    break
                else:
                    logger.info(f"❌ gpt-5 judge: {gpt_judge}, skip this result")
            if len(results) == 0:
                logger.info("❌ No valid search results found from any engine. Falling back to google search")
                google_result_str = await google_search(query)
                # google_search returns a string, not a SearchResult, so wrap it
                google_result = SearchResult(
                    engine="Google",
                    query=query,
                    content=google_result_str,
                    success=not google_result_str.startswith("[ERROR]"),
                    error=google_result_str if google_result_str.startswith("[ERROR]") else None
                )
                if google_result.success and len(google_result.content) > 0:
                    logger.info(f"✅ Google search ran successfully")
                else:
                    logger.info("❌ Google search has problem!")
                results.append(google_result)

        # 格式化并返回结果
        formatted_results = format_search_results(results, query)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ General search completed in {elapsed_time:.2f}s")
        # logger.info("general_search inner formatted_results:")
        # logger.info(formatted_results)
        return formatted_results
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ General search error in {elapsed_time:.2f}s: {e}")
        return f"[ERROR]: General search failed: {str(e)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Searching MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport method: 'stdio' or 'sse' (default: sse)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to use when running with SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8936,
        help="Port to use when running with SSE transport (default: 8936)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
