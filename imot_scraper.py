import numpy as np
import pandas as pd
import requests
import importlib
# from settings import *
import settings as s
importlib.reload(s)

from bs4 import BeautifulSoup
from urllib.parse import unquote, quote

class ImotScraper:

    """
    Scrapes the list of URLs
    """

    def __init__(self, main_url, imot_cities, imot_regions, imot_types, payload, headers, ads_url_2, encoding):
        """

        :param main_url:
        :param imot_cities:
        :param imot_regions:
        :param imot_types:
        :param payload:
        :param headers:
        :param imot_slinks:
        """

        self.main_url = main_url
        self.imot_cities = imot_cities
        self.imot_regions = imot_regions
        self.imot_types = imot_types
        self.payload = payload
        self.headers = headers
        self.encoding = encoding
        self.ads_url_2 = ads_url_2

        self.imot_slinks = {}
        self.all_ad_urls = []
        self.all_ads_details = pd.DataFrame()

    def get_slinks(self):
        """
        Gets the short URL string for all imot types
        """
        for city in self.imot_cities:
            imot_city = quote(city, encoding=self.encoding)

            for region in self.imot_regions:
                imot_region = quote(region, encoding=self.encoding)

                for imot_type in self.imot_types:
                    payload = self.payload.format(imot_type, imot_city, imot_region)
                    headers = self.headers

                    response = requests.request("POST",
                                                self.main_url,
                                                headers=headers,
                                                data=payload.encode('utf-8'))

                    s = response.text.find("slink")
                    e = response.text.find("';", s)
                    slink = response.text[s+6:e]
                    slink_key = ' '.join([str(imot_type), region, city])
                    self.imot_slinks[slink_key] = slink

    def get_all_ads(self):
        """
        Scrapes all URLs for the given types
        """

        def get_slink_pages(slink):
            """
            Gets the max page number (pagination)
            """

            main_url_req = requests.get(f'{self.ads_url_2}{slink}')
            print(f'{self.ads_url_2}{slink}')
            main_url_req.encoding = self.encoding
            soup = BeautifulSoup(main_url_req.text, 'html.parser')
            print(soup.status)
            page_numbers = soup.find('span', {'class': 'pageNumbersInfo'}).text.split(' ')[-1]
            print(page_numbers)
            return int(page_numbers)

        def get_slink_ads(slink):
            """
            """

            page_numbers = get_slink_pages(slink)
            ad_urls = []

            for page in range(1, int(page_numbers) + 1):
                print(page)
                p = f'https://www.imot.bg/pcgi/imot.cgi?act=3&slink={slink}&f1={page}'
                main_url_req = requests.get(p)
                main_url_req.encoding = self.encoding
                soup = BeautifulSoup(main_url_req.text, 'html.parser')

                hrefs = soup.find_all('a', href=True)
                fixed_str = r'imot.bg/pcgi/imot.cgi?act='
                ad_urls.extend(
                    list(set([_['href'] for _ in hrefs if fixed_str in _['href'] and
                              '&adv=' in _['href']])))

            # URL clean-up and deduplication
            ad_urls = ['https://' + _.replace('//', '') for _ in ad_urls]
            ad_urls = [_.replace('#player', '') for _ in ad_urls]
            ad_urls = [_.split('&slink')[0] for _ in ad_urls]
            ad_urls = list(set(ad_urls))

            return ad_urls

        for _ in self.imot_slinks.values():
            print(_)
            ads_urls = get_slink_ads(_)
            self.all_ad_urls.extend(ads_urls)

    def get_ad_info(self, ad_url):
        """

        :param ad_url:
        :return:
        """

        print(ad_url)
        ad_info = requests.get(ad_url)
        ad_info.encoding = self.encoding
        ad_soup = BeautifulSoup(ad_info.text, 'html.parser')

        try:
            ad_name = ad_soup.find('title').text
            f_split = ad_name.split(' в ')

            try:
                ad_type = f_split[0].replace('Продава ', '')
            except:
                ad_type = 'Missing'

            f_split2 = f_split[1].split(', ')

            try:
                ad_city = f_split2[0]
            except:
                ad_city = 'Missing'

            f_split3 = f_split2[1].split(' - ')
            try:
                ad_neighborhood = f_split3[0]
            except:
                ad_neighborhood = 'Missing'

            try:
                ad_kvm = float(f_split3[1].split(' / ')[0].replace(' кв.м', ''))
            except:
                ad_kvm = 0

            try:
                ad_price = f_split3[1].split(' / ')[1].split(' :: ')[0]
                ad_price, ad_currency = ad_price.split(' ')[0:2]
                ad_price = float(ad_price)
                ad_price = np.where(ad_currency == 'лв.', ad_price / 1.98, ad_price)
            except:
                ad_price = 0
                ad_currency = 'Missing'

            try:
                ad_price_per_kvm = round(ad_price / ad_kvm, 2)
            except:
                ad_price_per_kvm = 0

            try:
                ad_description = ad_soup.find('div', {'id': 'description_div'}).text
            except:
                ad_description = 'Missing'

            try:
                ad_street = ad_soup.find(
                    'h2', {'style': 'font-weight:normal; font-size:14px;'}).text.split(', ')[-1]
            except:
                ad_street = 'Missing'

        except Exception as e:
            print(e)

        try:
            ad_details = pd.DataFrame({
                # 'pic_url': pic_url,
                'ad_url': ad_url,
                'ad_city': ad_city,
                'ad_neighborhood': ad_neighborhood,
                'ad_type': ad_type,
                'ad_description': ad_description,
                'ad_price': ad_price,
                'ad_currency': ad_currency,
                'ad_kvm': ad_kvm,
                'ad_price_per_kvm': ad_price_per_kvm,
                # 'ad_price_change': ad_price_change,
                'ad_street': ad_street}, index=[0])

            return ad_details

        except Exception as e:
            print(e)
            pass

    def get_all_ads_info(self):
        """
        """

        for ad_url in self.all_ad_urls:
            self.all_ads_details = pd.concat([self.all_ads_details, self.get_ad_info(ad_url)])

        self.all_ads_details['avg_price_per_kvm_of_sample'] = self.all_ads_details.groupby(
            ['ad_city',
             'ad_neighborhood',
             'ad_type'])['ad_price_per_kvm'].transform('mean')

        self.all_ads_details['price_per_kvm_gain'] = \
            self.all_ads_details['ad_price_per_kvm'] / self.all_ads_details['avg_price_per_kvm_of_sample'] - 1
        self.all_ads_details['price_per_kvm_gain'] = round(self.all_ads_details['price_per_kvm_gain'] * 100, 2)


imot = ImotScraper(s.MAIN_URL, s.IMOT_CITIES, s.IMOT_REGIONS, s.IMOT_TYPES,
                   s.PAYLOAD, s.HEADERS, s.ADS_URL_2, s.ENCODING)
imot.get_slinks()
print(imot.imot_slinks)
imot.get_all_ads()
imot.get_all_ads_info()
r = imot.all_ads_details.reset_index(drop=True)