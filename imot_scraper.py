import io
import hashlib
import requests
import importlib
import numpy as np
import pandas as pd
import psycopg2 as pg
from datetime import datetime
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from urllib.parse import unquote, quote

import settings as s
importlib.reload(s)


class ImotScraper:

    """
    Scrapes the list of URLs and loads them in DB.
    """

    def __init__(self):
        """
        """

        self.main_url = s.MAIN_URL
        self.imot_cities = s.IMOT_CITIES
        self.imot_regions = s.IMOT_REGIONS
        self.imot_types = s.IMOT_TYPES
        self.payload = s.PAYLOAD
        self.headers = s.HEADERS
        self.encoding = s.ENCODING
        self.ads_url_2 = s.ADS_URL_2
        self.pg_connection_string = s.PG_CONNECTION_STRING

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

        def hash_description(x):

            hash_object = hashlib.sha256()
            byte_data = x.encode()
            hash_object.update(byte_data)
            hashed_data = hash_object.hexdigest()

            return hashed_data

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

        str_to_hash = ad_url+ad_city+ad_neighborhood+ad_type+ad_description+str(ad_price)+\
                      ad_currency+str(ad_kvm)+str(ad_price_per_kvm)+ad_street
        try:
            ad_details = pd.DataFrame({
                # 'pic_url': pic_url,
                'ad_url': ad_url,
                'ad_city': ad_city,
                'ad_neighborhood': ad_neighborhood,
                'ad_type': ad_type,
                'ad_description': ad_description,
                'ad_descr_hash': hash_description(str_to_hash),
                'ad_price': int(ad_price),
                'ad_currency': ad_currency,
                'ad_kvm': int(ad_kvm),
                'ad_price_per_kvm': ad_price_per_kvm,
                # 'ad_price_change': ad_price_change,
                'ad_street': ad_street,
                'updated_dt': datetime.today().strftime('%Y-%m-%d %H:%M:%S')}, index=[0])

            return ad_details

        except Exception as e:
            print(e)
            pass

    def get_all_ads_info(self):
        """
        """

        for ad_url in self.all_ad_urls:
            self.all_ads_details = pd.concat([self.all_ads_details, self.get_ad_info(ad_url)])

            dictionary = {'\t': ''}  # Can be extended with more problematic chars
            self.all_ads_details.replace(dictionary, regex=True, inplace=True)

        # self.all_ads_details['avg_price_per_kvm_of_sample'] = self.all_ads_details.groupby(
        #     ['ad_city',
        #      'ad_neighborhood',
        #      'ad_type'])['ad_price_per_kvm'].transform('mean')
        #
        # self.all_ads_details['price_per_kvm_gain'] = \
        #     self.all_ads_details['ad_price_per_kvm'] / self.all_ads_details['avg_price_per_kvm_of_sample'] - 1
        # self.all_ads_details['price_per_kvm_gain'] = round(self.all_ads_details['price_per_kvm_gain'] * 100, 2)

    def write_to_db(self, table_name_latest, table_name_history):

        engine = create_engine(self.pg_connection_string)
        conn = engine.raw_connection()
        # cur = conn.cursor()

        # table_name_latest = 'ads_latest'
        # table_name_history = 'ads_history'

        with conn.cursor() as cur:

            cur.execute(
                f"""
                CREATE TEMPORARY TABLE tmp (LIKE {table_name_history})
                """
            )

            output = io.StringIO()
            self.all_ads_details.to_csv(output, sep='\t', header=False, index=False)
            # imot.all_ads_details.to_csv(output, sep='\t', header=False, index=False)
            output.seek(0)


            cur.copy_from(output, 'tmp')

            cur.execute(
                f"""
                INSERT INTO {table_name_latest}
                SELECT * FROM tmp
                ON CONFLICT (ad_url) DO UPDATE
                    SET ad_city = CASE WHEN {table_name_latest}.ad_city <> excluded.ad_city THEN excluded.ad_city ELSE {table_name_latest}.ad_city END,
                        ad_neighborhood = CASE WHEN {table_name_latest}.ad_neighborhood <> excluded.ad_neighborhood THEN excluded.ad_neighborhood ELSE {table_name_latest}.ad_neighborhood END,
                        ad_type = CASE WHEN {table_name_latest}.ad_type <> excluded.ad_type THEN excluded.ad_type ELSE {table_name_latest}.ad_type END,
                        ad_description = CASE WHEN {table_name_latest}.ad_description <> excluded.ad_description THEN excluded.ad_description ELSE {table_name_latest}.ad_description END,
                        ad_hash = CASE WHEN {table_name_latest}.ad_hash <> excluded.ad_hash THEN excluded.ad_hash ELSE {table_name_latest}.ad_hash END,
                        ad_price = CASE WHEN {table_name_latest}.ad_price <> excluded.ad_price THEN excluded.ad_price ELSE {table_name_latest}.ad_price END,
                        ad_currency = CASE WHEN {table_name_latest}.ad_currency <> excluded.ad_currency THEN excluded.ad_currency ELSE {table_name_latest}.ad_currency END,
                        ad_kvm = CASE WHEN {table_name_latest}.ad_kvm <> excluded.ad_kvm THEN excluded.ad_kvm ELSE {table_name_latest}.ad_kvm END,
                        ad_price_per_kvm = CASE WHEN {table_name_latest}.ad_price_per_kvm <> excluded.ad_price_per_kvm THEN excluded.ad_price_per_kvm ELSE {table_name_latest}.ad_price_per_kvm END,
                        ad_street = CASE WHEN {table_name_latest}.ad_street <> excluded.ad_street THEN excluded.ad_street ELSE {table_name_latest}.ad_street END,
                        updated_ts = CASE WHEN {table_name_latest}.updated_ts <> excluded.updated_ts THEN excluded.updated_ts ELSE {table_name_latest}.updated_ts END
                    
                RETURNING *;
                """
            )

            cur.execute(
                f"""
                INSERT INTO {table_name_history}
                SELECT * FROM tmp
                ON CONFLICT (ad_hash) 
                DO NOTHING
                """
            )

            cur.execute(f"DROP TABLE tmp")
            conn.commit()
        conn.close()

if __name__ == '__main__':
    imot = ImotScraper()
    imot.get_slinks()
    print(imot.imot_slinks)
    imot.get_all_ads()
    imot.get_all_ads_info()
    imot.write_to_db('ads_latest', 'ads_history')














