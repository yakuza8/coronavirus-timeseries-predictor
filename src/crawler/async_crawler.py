import asyncio
import logging

from aiohttp import ClientSession
from bs4 import BeautifulSoup
from src.crawler.crawler_utilities import get_countries_available_today_with_urls, find_charts_and_values, \
    get_request_header, SITE_URL, COUNTRY_DETAIL_URL, export_dataset

logging.basicConfig(level=logging.DEBUG)


async def get_today_available_county_dataset():
    """
    Asynchronous dataset obtainment procedure where at first site is fetched and then countries are parsed from site.
    For each available country, async task fired to get its data
    :return: Dataset up to the current day with different countries in form of dictionary where country values are
    also dictionary and inside that several types of daily data exist
    """
    logging.info('Crawler will fetch daily tables of available countries.')
    # Get session to further requests
    async with ClientSession() as session:
        content = await search_site(session)
        countries = get_countries_available_today_with_urls(content)

        dataset = {}
        tasks = []
        for index, (country_name, path) in enumerate(countries):
            logging.debug('Performing query on {0} at index {1}'.format(country_name, index))
            task = get_country_data_set(session, country_name, path, dataset)
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Export dataset
        export_dataset(dataset)

        return dataset


async def search_site(session: ClientSession) -> BeautifulSoup:
    logging.debug('Site access will be performed.')
    site_content = await perform_request_and_parse_url(session, SITE_URL)
    logging.debug('Site is retrieved.')
    return site_content


async def get_country_data_set(session: ClientSession, country_name: str, path: str, dataset: dict):
    data = await get_country_content(session, path)
    tables = find_charts_and_values(country_name, data)
    dataset[country_name] = tables


async def get_country_content(session: ClientSession, relative_path: str) -> BeautifulSoup:
    # Make country name lower and create query url
    url_for_country = COUNTRY_DETAIL_URL.format(relative_path)
    country_content = await perform_request_and_parse_url(session, url_for_country)
    logging.debug('{0} fetching finished.'.format(relative_path))
    return country_content


async def perform_request_and_parse_url(session: ClientSession, url: str) -> BeautifulSoup:
    headers = get_request_header()

    # Prepare request
    async with session.get(url, headers=headers) as response:
        data = await response.read()
        # Parse obtained data
        parsed_data = BeautifulSoup(data, 'html.parser')
        return parsed_data


loop = asyncio.get_event_loop()
future = asyncio.ensure_future(get_today_available_county_dataset())
loop.run_until_complete(future)
