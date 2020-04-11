import logging
import unittest
import urllib.request

from bs4 import BeautifulSoup
from src.crawler.crawler_utilities import get_countries_available_today_with_urls, find_charts_and_values, \
    get_request_header, SITE_URL, COUNTRY_DETAIL_URL

logging.basicConfig(level=logging.DEBUG)


class Crawler:

    def __init__(self):
        pass

    @staticmethod
    def get_today_available_county_dataset():
        """
        Procedure to get whole daataset where first available countries are fetched then for each country corresponding
        detail page is retrieved to parse data.
        :return: Dataset up to the current day with different countries in form of dictionary where country values are
        also dictionary and inside that several types of daily data exist
        """
        logging.info('Crawler will fetch daily tables of available countries.')
        content = Crawler._search_site()
        countries = get_countries_available_today_with_urls(content)

        dataset = {}
        for index, (country_name, path) in enumerate(countries):
            logging.debug('Performing query on {0} at index {1}'.format(country_name, index))
            data = Crawler._get_country_content(path)
            tables = find_charts_and_values(country_name, data)
            dataset[country_name] = tables

        return dataset

    @staticmethod
    def _search_site() -> BeautifulSoup:
        logging.debug('Site access will be performed.')
        site_content = Crawler._perform_request_and_parse_url(SITE_URL)
        logging.debug('Site is retrieved.')
        return site_content

    @staticmethod
    def _get_country_content(relative_path: str) -> BeautifulSoup:
        # Make country name lower and create query url
        url_for_country = COUNTRY_DETAIL_URL.format(relative_path)
        country_content = Crawler._perform_request_and_parse_url(url_for_country)
        logging.debug('{0} fetching finished.'.format(relative_path))
        return country_content

    @staticmethod
    def _perform_request_and_parse_url(url) -> BeautifulSoup:
        headers = get_request_header()

        # Prepare request
        request = urllib.request.Request(url, None, headers)
        response = urllib.request.urlopen(request)
        data = response.read()

        # Parse obtained data
        parsed_data = BeautifulSoup(data, 'html.parser')
        return parsed_data


class CrawlerUnittest(unittest.TestCase):

    def test_get_site_content(self):
        content = Crawler._search_site()
        self.assertIsNotNone(content)

    def test_get_countries_and_urls(self):
        content = Crawler._search_site()
        countries = get_countries_available_today_with_urls(content)
        self.assertTrue(len(countries) > 0)

    def test_get_data_for_valid_and_invalid_country(self):
        country, path = ('USA', 'country/us/')
        data = Crawler._get_country_content(path)
        self.assertTrue('404 Not Found' not in data.text)

        country, path = ('Black Mesa', 'country/zen/')
        data = Crawler._get_country_content(path)
        self.assertTrue('404 Not Found' in data.text)

    def test_chart_content(self):
        country, path = ('USA', 'country/us/')
        data = Crawler._get_country_content(path)
        tables = find_charts_and_values(country, data)
        self.assertTrue(len(tables) > 0)

    def test_get_dataset(self):
        dataset = Crawler.get_today_available_county_dataset()
        self.assertTrue(len(dataset) > 0)
