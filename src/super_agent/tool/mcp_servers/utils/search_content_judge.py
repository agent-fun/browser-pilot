import os
import json
from typing import Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
from src.super_agent.tool.logger import bootstrap_logger

load_dotenv()

logger = bootstrap_logger()


async def search_content_judge(query: str, results: str) -> bool:
    """Judge if the content is suitable for search.
    """
    # logger.info("-"*30)
    # logger.info('query:' + query)
    # logger.info('results:' + results)
    # logger.info("-"*30)
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL")
    client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    model = "gpt-5" 
    system_prompt = """You are given a user query and the current search results. Your task is to decide whether the search results actually contain relevant content that addresses the query.

Answer with one of the following labels:

YES if the search results clearly contain information that answers or directly relates to the query.

NO if the search results do not contain relevant information or only mention unrelated content.

Be strict: partial matches, vague mentions, or irrelevant details should be considered NO.

Return your response in the following JSON format:{
    "judge": "YES" or "NO"
}

Examples:

Input:
### Query: where is the eiffel tower?
### Search Results: I don't have any information about the Eiffel Tower.

Output: {
    "judge": "NO"
}

Always return valid JSON."""

    try:
        response = await client.chat.completions.create(
            model= model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Please judge the search results about the query:\n ### Query: {query}\n ### Search Results: {results}"
                }
            ],
            reasoning_effort="medium",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "query_results_judge",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "judge": {
                                "type": "string",
                                "description": "whether the search results actually contain relevant content that addresses the query, YES or NO"
                            }
                        },
                        "required": ["judge"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        content = response.choices[0].message.content
        logger.info("search_content_judge content:")
        logger.info(content)
        # Parse JSON response
        parsed_result = json.loads(content)
        return parsed_result.get("judge", "NO").strip().upper() == "YES"
        
    except json.JSONDecodeError as e:
            return False
    except Exception as e:
            return False