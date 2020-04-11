import logging
import yaml

from bs4 import BeautifulSoup
from typing import List, Tuple, Any

from src.crawler.data_exporter import DataExporter

SITE_URL = 'https://www.worldometers.info/coronavirus/#countries'
COUNTRY_DETAIL_URL = 'https://www.worldometers.info/coronavirus/{0}'


def get_countries_available_today_with_urls(content: BeautifulSoup) -> List[Tuple[Any, Any]]:
    """
    Procedure to read available counties to get their details
    :param content: Parsed HTML content of the main page of site
    :return: List of tuples where tuples are in form of (country_name, relative_path_to_detail_page)
    """

    logging.debug('Countries will be parsed.')
    countries = content.find_all('a', {'class': 'mt_a'})
    parsed_countries = list(set((country.text, country.attrs['href']) for country in countries))
    logging.debug('Countries are parsed {0} counties are fetched and they are {1}'.format(len(parsed_countries),
                                                                                          parsed_countries))
    return parsed_countries


def find_charts_and_values(country_name: str, content: BeautifulSoup) -> dict:
    """
    Procedure for getting data from the detail page of the target country
    :param country_name: Country name of the target country
    :param content: Parsed HTML content of the target country's detail page
    :return: Dictionary of data for that country where keys are table name and values are dictionary of x and y values
    of table
    """
    logging.debug('Obtaining charts for {0}.'.format(country_name))
    # Replace non-parsable characters and filter charts
    charts = [chart.string.replace('\t', '').replace('/*', '').replace('*/', '') for chart in
              content.find_all('script', {'type': 'text/javascript'}) if
              chart.string is not None and 'Highcharts.chart' in chart.string]

    try:
        # Do not take both charts where each table has two charts in page (the second one contains logarithmic values)
        chart_summaries = [yaml.load(chart[chart.index('{'): chart.index(');')]) for chart in charts]
        chart_summaries = [summary for summary in chart_summaries if
                           'title' in summary and 'xAxis' in summary and 'series' in summary]

        logging.debug('Finishing charts for {0}.'.format(country_name))
        # Create final dictionary where table name should have at least some characters
        return {
            summary['title']['text']: {
                'x': summary['xAxis']['categories'],
                'y': summary['series'][0]['data']
            } for summary in chart_summaries if len(summary['title']['text']) > 0
        }
    except Exception as e:
        logging.error('Error occurred for {0} as {1}', country_name, e)
        return {}


def get_request_header() -> dict:
    """
    Common functionality of forming header for further requests
    :return: Header as dictionary
    """
    # Sample user agent
    user_agent = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:75.0) Gecko/20100101 Firefox/75.0'
    headers = {'User-Agent': user_agent}

    return headers


def export_dataset(dataset: dict):
    """
    Simply write dataset
    :param dataset: Dataset obtained from crawler
    :return: Nothing
    """
    exporter = DataExporter()
    exporter.write_dataset(dataset)
