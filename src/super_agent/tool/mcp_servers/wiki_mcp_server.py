import requests
import datetime
import calendar
import json
import urllib.parse
from fastmcp import FastMCP
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters  # (already imported in config.py)
import wikipedia
from .utils.smart_request import smart_request, api_request_json
from src.super_agent.tool.logger import bootstrap_logger
from typing import Optional, Dict, Any

# Initialize FastMCP server
mcp = FastMCP("wiki-mcp-server")

logger = bootstrap_logger()

@mcp.tool()
async def wiki_get_page_content(wikipedia_entity: str, first_sentences_number: int = 10) -> str:
    """Get specific Wikipedia page content for the specific entity (people, places, concepts, events) and return structured information.

    This tool searches Wikipedia for the given entity and returns either the first few sentences
    (which typically contain the summary/introduction) or full page content based on parameters.
    It handles disambiguation pages and provides clean, structured output.

    Args:
        entity: The entity to search for in Wikipedia.
        first_sentences: Number of first sentences to return from the page. Set to 0 to return full content. Defaults to 10.

    Returns:
        str: Formatted search results containing title, first sentences/full content, and URL.
             Returns error message if page not found or other issues occur.
    """
    wikipedia_entity = (wikipedia_entity or "").strip()
    if not wikipedia_entity:
        return "[ERROR]: Wikipedia entity cannot be empty."

    if first_sentences_number < 0:
        first_sentences_number = 0

    logger.info(
        "📚 wiki_get_page_content called",
        extra={"entity": wikipedia_entity, "sentences": first_sentences_number},
    )

    try:
        # Try to get the Wikipedia page directly
        wiki_page = wikipedia.page(title=wikipedia_entity, auto_suggest=False)
        logger.info("✅ Wikipedia page resolved", extra={"title": wiki_page.title})

        # Prepare the result
        formatted_result_parts = [f"Page Title: {wiki_page.title}"]

        if first_sentences_number > 0:
            wiki_summary = ""
            summary_fetch_error: Optional[Exception] = None
            try:
                wiki_summary = wikipedia.summary(
                    wikipedia_entity, sentences=first_sentences_number, auto_suggest=False
                )
            except Exception as error:
                summary_fetch_error = error

            if not wiki_summary:
                page_content_sentences = wiki_page.content.split(". ")[:first_sentences_number]
                wiki_summary = (
                    ". ".join(page_content_sentences) + "."
                    if page_content_sentences
                    else wiki_page.content[:5000] + "..."
                )

            formatted_result_parts.append(
                f"First {first_sentences_number} sentences (introduction): {wiki_summary}"
            )
        else:
            # Return full content if first_sentences is 0
            # TODO: Context Engineering Needed
            formatted_result_parts.append(f"Content: {wiki_page.content}")

        formatted_result_parts.append(f"URL: {wiki_page.url}")

        return "\n\n".join(formatted_result_parts)

    except wikipedia.exceptions.DisambiguationError as error:
        disambiguation_options_list = "\n".join([f"- {option}" for option in error.options[:10]])
        logger.warning(
            "Wikipedia disambiguation triggered",
            extra={"entity": wikipedia_entity, "options": error.options[:10]},
        )
        output = (
            f"Disambiguation Error: Multiple pages found for '{wikipedia_entity}'.\n\n"
            f"Available options:\n{disambiguation_options_list}\n\n"
            f"Please be more specific in your search query."
        )
        try:
            wiki_search_results = wikipedia.search(wikipedia_entity, results=5)
            if wiki_search_results:
                output += f"Try to search {wikipedia_entity} in Wikipedia: {wiki_search_results}"
            return output
        except Exception:
            pass

        return output

    except wikipedia.exceptions.PageError:
        logger.warning("❌ Wikipedia page not found", extra={"entity": wikipedia_entity})
        try:
            wiki_search_results = wikipedia.search(wikipedia_entity, results=5)
            if wiki_search_results:
                suggestion_list = "\n".join(
                    [f"- {result}" for result in wiki_search_results[:5]]
                )
                return (
                    f"Page Not Found: No Wikipedia page found for '{wikipedia_entity}'.\n\n"
                    f"Similar pages found:\n{suggestion_list}\n\n"
                    f"Try searching for one of these suggestions instead."
                )
            else:
                return (
                    f"Page Not Found: No Wikipedia page found for '{wikipedia_entity}' "
                    f"and no similar pages were found. Please try a different search term."
                )
        except Exception as search_error:
            logger.error(
                "⚠️ Follow-up Wikipedia search failed",
                extra={"entity": wikipedia_entity, "error": str(search_error)},
            )
            return (
                f"Page Not Found: No Wikipedia page found for '{wikipedia_entity}'. "
                f"Search for alternatives also failed: {str(search_error)}"
            )

    except wikipedia.exceptions.RedirectError:
        logger.error("🔁 Wikipedia redirect failed", extra={"entity": wikipedia_entity})
        return (
            f"Redirect Error: Unable to follow redirects for '{wikipedia_entity}'. "
            f"Please try a more specific term."
        )

    except requests.exceptions.RequestException as error:
        logger.error(
            "Network error when contacting Wikipedia",
            extra={"entity": wikipedia_entity, "error": str(error)},
        )
        return f"Network Error: Failed to connect to Wikipedia: {str(error)}"

    except wikipedia.exceptions.WikipediaException as error:
        logger.error(
            "Wikipedia API error",
            extra={"entity": wikipedia_entity, "error": str(error)},
        )
        return f"Wikipedia Error: An error occurred while searching Wikipedia: {str(error)}"

    except Exception as error:
        logger.exception(
            "Unexpected error while fetching Wikipedia page",
            extra={"entity": wikipedia_entity},
        )
        return f"Unexpected Error: An unexpected error occurred: {str(error)}"


@mcp.tool()
async def search_wiki_revision(
    wikipedia_entity: str, year: int, month: int, max_revisions: int = 50
) -> str:
    """Search for an entity in Wikipedia and return the revision history for a specific month.

    Args:
        entity: The entity to search for in Wikipedia.
        year: The year of the revision (e.g. 2024).
        month: The month of the revision (1-12).
        max_revisions: Maximum number of revisions to return. Defaults to 50.

    Returns:
        str: Formatted revision history with timestamps, revision IDs, and URLs.
             Returns error message if page not found or other issues occur.
    """
    wikipedia_entity = (wikipedia_entity or "").strip()
    if not wikipedia_entity:
        return "[ERROR]: Wikipedia entity cannot be empty."

    if max_revisions <= 0:
        max_revisions = 1
    elif max_revisions > 500:
        max_revisions = 500

    logger.info(
        "search_wiki_revision called",
        extra={
            "entity": wikipedia_entity,
            "year": year,
            "month": month,
            "max_revisions": max_revisions,
        },
    )

    # Auto-adjust date values and track changes
    date_adjustments = []
    original_year, original_month = year, month
    current_year = datetime.datetime.now().year

    # Adjust year to valid range
    if year < 2000:
        year = 2000
        date_adjustments.append(
            f"Year adjusted from {original_year} to 2000 (minimum supported)"
        )
    elif year > current_year:
        year = current_year
        date_adjustments.append(
            f"Year adjusted from {original_year} to {current_year} (current year)"
        )

    # Adjust month to valid range
    if month < 1:
        month = 1
        date_adjustments.append(f"Month adjusted from {original_month} to 1")
    elif month > 12:
        month = 12
        date_adjustments.append(f"Month adjusted from {original_month} to 12")

    # Prepare adjustment message if any changes were made
    if date_adjustments:
        date_adjustment_message = (
            "Date auto-adjusted: "
            + "; ".join(date_adjustments)
            + f". Using {year}-{month:02d} instead.\n\n"
        )
        logger.info(
            "Date adjusted for wiki revision search",
            extra={
                "entity": wikipedia_entity,
                "adjustment": date_adjustment_message.strip(),
            },
        )
    else:
        date_adjustment_message = ""

    wikipedia_api_base_url = "https://en.wikipedia.org/w/api.php"

    try:
        # Construct the time range
        revision_start_date = datetime.datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        revision_end_date = datetime.datetime(year, month, last_day, 23, 59, 59)

        # Convert to ISO format (UTC time)
        start_timestamp_iso = revision_start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_timestamp_iso = revision_end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # API parameters configuration
        wikipedia_api_params = {
            "action": "query",
            "format": "json",
            "titles": wikipedia_entity,
            "prop": "revisions",
            "rvlimit": min(max_revisions, 500),  # Wikipedia API limit
            "rvstart": start_timestamp_iso,
            "rvend": end_timestamp_iso,
            "rvdir": "newer",
            "rvprop": "timestamp|ids",
        }

        logger.debug(
            "Fetching Wikipedia revisions",
            extra={"entity": wikipedia_entity, "params": wikipedia_api_params},
        )

        try:
            api_response_data = await api_request_json(url=wikipedia_api_base_url, params=wikipedia_api_params, timeout=30)
        except requests.exceptions.RequestException as e:
            logger.error(
                "Network error fetching Wikipedia revisions",
                extra={"entity": wikipedia_entity, "error": str(e)},
            )
            return f"[ERROR]: Network Error: Failed to connect to Wikipedia API: {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON response from Wikipedia API",
                extra={"entity": wikipedia_entity, "error": str(e)},
            )
            return f"[ERROR]: API Error: Wikipedia API returned invalid JSON response: {str(e)}"

        # Check for API errors
        if "error" in api_response_data:
            error_info = api_response_data["error"].get("info", "Unknown error")
            logger.error(
                "Wikipedia API returned an error",
                extra={"entity": wikipedia_entity, "error": error_info},
            )
            return f"[ERROR]: Wikipedia API Error: {error_info}"

        # Process the response
        wikipedia_pages = api_response_data.get("query", {}).get("pages", {})

        if not wikipedia_pages:
            logger.warning(
                "No Wikipedia pages found in revision query",
                extra={"entity": wikipedia_entity},
            )
            return f"[ERROR]: No results found for entity '{wikipedia_entity}'"

        # Check if page exists
        wikipedia_page_id = list(wikipedia_pages.keys())[0]
        if wikipedia_page_id == "-1":
            logger.warning(
                "Wikipedia page not found for revision query",
                extra={"entity": wikipedia_entity},
            )
            return f"[ERROR]: Page Not Found: No Wikipedia page found for '{wikipedia_entity}'"

        wikipedia_page_info = wikipedia_pages[wikipedia_page_id]
        wikipedia_page_title = wikipedia_page_info.get("title", wikipedia_entity)

        if "revisions" not in wikipedia_page_info or not wikipedia_page_info["revisions"]:
            logger.info(
                "No revisions in requested window",
                extra={"entity": wikipedia_entity, "year": year, "month": month},
            )
            return (
                date_adjustment_message + f"Page Title: {wikipedia_page_title}\n\n"
                f"No revisions found for '{wikipedia_entity}' in {year}-{month:02d}.\n\n"
                f"The page may not have been edited during this time period."
            )

        # Format the results
        formatted_result_parts = [
            f"Page Title: {wikipedia_page_title}",
            f"Revision Period: {year}-{month:02d}",
            f"Total Revisions Found: {len(wikipedia_page_info['revisions'])}",
        ]

        # Add revision details
        revision_details_list = []
        for i, revision in enumerate(wikipedia_page_info["revisions"], 1):
            revision_id = revision.get("revid")
            revision_timestamp = revision.get("timestamp", "Unknown timestamp")

            # Format timestamp for better readability
            try:
                revision_datetime = datetime.datetime.fromisoformat(revision_timestamp.replace("Z", "+00:00"))
                formatted_timestamp = revision_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_timestamp = revision_timestamp

            # Construct revision URL
            revision_url = f"https://en.wikipedia.org/w/index.php?title={wikipedia_entity}&oldid={revision_id}"

            revision_details_list.append(
                f"{i}. Revision ID: {revision_id}\n"
                f"   Timestamp: {formatted_timestamp}\n"
                f"   URL: {revision_url}"
            )

        if revision_details_list:
            formatted_result_parts.append("Revisions:\n" + "\n\n".join(revision_details_list))

        logger.info(
            "Wikipedia revisions retrieved",
            extra={
                "entity": wikipedia_entity,
                "revisions": len(revision_details_list),
                "period": f"{year}-{month:02d}",
            },
        )

        return (
            date_adjustment_message
            + "\n\n".join(formatted_result_parts)
            + "\n\nHint: You can use the `scrape_website` tool to get the webpage content of a URL."
        )

    except ValueError as e:
        # Only catch actual date validation errors, not JSON parsing errors
        error_str = str(e)
        # Check if this is actually a date error (from datetime construction)
        if "date" in error_str.lower() or "month" in error_str.lower() or "year" in error_str.lower():
            logger.error(
                "Invalid date provided for Wikipedia revisions",
                extra={"entity": wikipedia_entity, "error": str(e)},
            )
            return f"[ERROR]: Date Error: Invalid date values - {str(e)}"
        else:
            # This might be a JSON parsing error that was caught as ValueError
            logger.error(
                "Data parsing error for Wikipedia revisions",
                extra={"entity": wikipedia_entity, "error": str(e)},
            )
            return f"[ERROR]: Data Error: Failed to parse response from Wikipedia API: {str(e)}"

    except Exception as e:
        logger.exception(
            "Unexpected error while retrieving Wikipedia revisions",
            extra={"entity": wikipedia_entity},
        )
        return f"[ERROR]: Unexpected Error: An unexpected error occurred: {str(e)}"


@mcp.tool()
async def search_archived_webpage(url: str, year: int, month: int, day: int) -> str:
    """Search the Wayback Machine (archive.org) for archived versions of a webpage, optionally for a specific date.

    Args:
        url: The URL to search for in the Wayback Machine.
        year: The target year (e.g., 2023).
        month: The target month (1-12).
        day: The target day (1-31).

    Returns:
        str: Formatted archive information including archived URL, timestamp, and status.
             Returns error message if URL not found or other issues occur.
    """
    # Handle empty URL
    url = (url or "").strip()
    if not url:
        return f"[ERROR]: Invalid URL: URL cannot be empty."

    # Validate URL format before processing
    original_url = url
    protocol_addition_hint = ""
    if not url.startswith(("http://", "https://")):
        # Try to add https://, but validate the result
        url = f"https://{url}"
        protocol_addition_hint = f"[NOTE]: Automatically added 'https://' to URL '{original_url}' -> '{url}'\n\n"
    
    # Basic URL validation - check if it looks like a valid URL
    try:
        parsed_url = urllib.parse.urlparse(url)
        # A valid URL should have a netloc (domain) and scheme
        if not parsed_url.netloc or not parsed_url.scheme:
            return f"[ERROR]: Invalid URL: '{original_url}' is not a valid URL format."
        # Check if netloc looks like a domain (has at least one dot or is localhost)
        if not ("." in parsed_url.netloc or parsed_url.netloc == "localhost"):
            # If it doesn't look like a domain, it's probably invalid
            if not url.startswith(("http://localhost", "https://localhost")):
                return f"[ERROR]: Invalid URL: '{original_url}' does not appear to be a valid URL."
    except Exception:
        return f"[ERROR]: Invalid URL: '{original_url}' is not a valid URL format."

    wikipedia_hint_message = ""
    if ".wikipedia.org" in url:
        wikipedia_hint_message = "Note: You are trying to search a Wikipedia page. You can also use the `search_wiki_revision` tool to get the revision content of a Wikipedia page.\n\n"

    # Check if specific date is requested
    target_date_string = ""
    date_adjustment_message = ""
    if year > 0 and month > 0:
        # For very old years (before 1995), return error instead of auto-adjusting
        # This matches test expectations
        if year < 1995:
            return f"[ERROR]: Invalid date: Year {year} is before 1995, which is the minimum supported year for Wayback Machine archives."
        
        # Auto-adjust date values and track changes
        date_adjustments = []
        original_year, original_month, original_day = year, month, day
        current_year = datetime.datetime.now().year

        # Adjust year to valid range (but we already checked for < 1995 above)
        if year > current_year:
            year = current_year
            date_adjustments.append(
                f"Year adjusted from {original_year} to {current_year} (current year)"
            )

        # Adjust month to valid range
        if month < 1:
            month = 1
            date_adjustments.append(f"Month adjusted from {original_month} to 1")
        elif month > 12:
            month = 12
            date_adjustments.append(f"Month adjusted from {original_month} to 12")

        # Adjust day to valid range for the given month/year
        max_day_for_month = calendar.monthrange(year, month)[1]
        if day < 1:
            day = 1
            date_adjustments.append(f"Day adjusted from {original_day} to 1")
        elif day > max_day_for_month:
            day = max_day_for_month
            date_adjustments.append(
                f"Day adjusted from {original_day} to {max_day_for_month} (max for {year}-{month:02d})"
            )

        # Update the date string with adjusted values
        target_date_string = f"{year:04d}{month:02d}{day:02d}"

        try:
            # Validate the final adjusted date
            datetime.datetime(year, month, day)
        except ValueError as e:
            return f"[ERROR]: Invalid date: {year}-{month:02d}-{day:02d}. {str(e)}"

        # Prepare adjustment message if any changes were made
        if date_adjustments:
            date_adjustment_message = (
                "Date auto-adjusted: "
                + "; ".join(date_adjustments)
                + f". Using {target_date_string} instead.\n\n"
            )

    try:
        wayback_api_base_url = "https://archive.org/wayback/available"
        # Search with specific date if provided
        async def _fetch_wayback_snapshot(wayback_api_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Fetch a snapshot from Wayback Machine API."""
            try:
                wayback_api_response = await api_request_json(url=wayback_api_base_url, params=wayback_api_params, timeout=30, max_retries=5)
                archived_snapshots = wayback_api_response.get("archived_snapshots", {})
                archived_snapshot = archived_snapshots.get("closest") if isinstance(archived_snapshots, dict) else None
                if archived_snapshot:
                    return archived_snapshot
                return None
            except requests.exceptions.RequestException:
                # Network/HTTP errors - return None to allow fallback to most recent
                return None
            except json.JSONDecodeError:
                # JSON parsing errors - return None to allow fallback
                return None

        def _format_archived_snapshot(archived_snapshot: Dict[str, Any], date_label: str) -> str:
            archived_url = archived_snapshot["url"]
            archived_timestamp = archived_snapshot["timestamp"]
            snapshot_available = archived_snapshot.get("available", True)

            # Format date info based on whether a specific date was requested
            if year > 0 and month > 0:
                requested_date_info = f"Requested Date: {year:04d}-{month:02d}-{day:02d}\n"
            else:
                requested_date_info = ""

            if not snapshot_available:
                return (
                    wikipedia_hint_message
                    + date_adjustment_message
                    + (
                        f"Archive Status: Snapshot exists but is not available\n\n"
                        f"Original URL: {url}\n"
                        + requested_date_info
                        + f"Closest Snapshot: {archived_timestamp}\n\n"
                        + f"Try a different date"
                    )
                )

            try:
                snapshot_datetime = datetime.datetime.strptime(archived_timestamp, "%Y%m%d%H%M%S")
                formatted_timestamp = snapshot_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_timestamp = archived_timestamp

            return (
                protocol_addition_hint
                + wikipedia_hint_message
                + date_adjustment_message
                + (
                    f"Archive Found: Archived version located\n\n"
                    f"Original URL: {url}\n"
                    + requested_date_info
                    + f"Closest Snapshot: {archived_timestamp}\n\n"
                    + f"Archived URL: {archived_url}\n"
                )
                + "\n\nHint: You can also use the `scrape_website` tool to get the webpage content of a URL."
            )

        if target_date_string:
            archived_snapshot = await _fetch_wayback_snapshot({"url": url, "timestamp": target_date_string})
            if archived_snapshot:
                return _format_archived_snapshot(
                    archived_snapshot,
                    f"Requested Date: {year:04d}-{month:02d}-{day:02d}\nClosest Snapshot",
                )

        # If no specific date requested (year=0, month=0, day=0) or date search failed, try most recent
        archived_snapshot = await _fetch_wayback_snapshot({"url": url})
        if archived_snapshot:
            return _format_archived_snapshot(archived_snapshot, "Most Recent Snapshot")

        return (
            protocol_addition_hint
            + wikipedia_hint_message
            + (
                f"Archive Not Found: No archived versions available\n\n"
                f"Original URL: {url}\n\n"
                f"The URL '{url}' has not been archived by the Wayback Machine.\n"
                f"You may want to:\n"
                f"- Check if the URL is correct\n"
                f"- Try a different URL and date\n"
            )
        )

    except requests.exceptions.RequestException as e:
        logger.error(
            "Network error connecting to Wayback Machine",
            extra={"url": url, "error": str(e)},
        )
        return (
            protocol_addition_hint
            + wikipedia_hint_message
            + f"[ERROR]: Network Error: Failed to connect to Wayback Machine: {str(e)}"
        )

    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON response from Wayback Machine",
            extra={"url": url, "error": str(e)},
        )
        return (
            protocol_addition_hint
            + wikipedia_hint_message
            + f"[ERROR]: Data Error: Failed to parse response from Wayback Machine: {str(e)}"
        )

    except ValueError as e:
        # Only catch actual date validation errors
        error_str = str(e)
        if "date" in error_str.lower() or "month" in error_str.lower() or "year" in error_str.lower() or "day" in error_str.lower():
            logger.error(
                "Invalid date for Wayback Machine search",
                extra={"url": url, "error": str(e)},
            )
            return f"[ERROR]: Invalid date: {str(e)}"
        else:
            logger.error(
                "Data parsing error for Wayback Machine",
                extra={"url": url, "error": str(e)},
            )
            return (
                protocol_addition_hint
                + wikipedia_hint_message
                + f"[ERROR]: Data Error: Failed to parse response from Wayback Machine: {str(e)}"
            )

    except Exception as e:
        logger.exception(
            "Unexpected error while searching Wayback Machine",
            extra={"url": url},
        )
        return (
            protocol_addition_hint
            + wikipedia_hint_message
            + f"[ERROR]: Unexpected Error: An unexpected error occurred: {str(e)}"
        )


@mcp.tool()
async def scrape_website(url: str) -> str:
    """This tool is used to scrape a website for its content. Search engines are not supported by this tool. This tool can also be used to get YouTube video non-visual information (however, it may be incomplete), such as video subtitles, titles, descriptions, key moments, etc.

    Args:
        url: The URL of the website to scrape.
    Returns:
        The scraped website content.
    """
    # TODO: Long Content Handling
    try:
        return await smart_request(url)
    except Exception as e:
        logger.exception(
            "Error scraping website",
            extra={"url": url},
        )
        return f"[ERROR]: Failed to scrape website: {str(e)}"

if __name__ == "__main__":
    # mcp.run(transport="stdio")
    mcp.run(transport="sse", host="127.0.0.1", port=8937)