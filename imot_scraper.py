import os
import io
import hashlib
import logging
import requests
import warnings
import importlib
import numpy as np
import pandas as pd
import settings as s
import psycopg2 as pg
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import unquote, quote

from bgner.imot_ner import ImotNer
from bgner import imot_config

load_dotenv()
importlib.reload(s)
warnings.filterwarnings("ignore")

class ImotScraper:

    """
    Scrapes the list of URLs and loads them in DB.
    """

    def __init__(self):
        """
        """

        db_user = os.getenv('POSTGRES_USER')
        db_pass = os.getenv('POSTGRES_PASSWORD')
        db_host = os.getenv('POSTGRES_HOST_IMOT')  # For prod use IMOT
        db_port = os.getenv('POSTGRES_PORT')
        db_database = os.getenv('POSTGRES_DB')

        self.main_url = s.MAIN_URL
        self.imot_cities = s.IMOT_CITIES
        self.imot_regions = s.IMOT_REGIONS
        self.imot_types = s.IMOT_TYPES
        self.payload = s.PAYLOAD
        self.headers = s.HEADERS
        self.encoding = s.ENCODING
        self.ads_url_2 = s.ADS_URL_2

        self.imot_slinks = {}
        self.all_ad_urls = []
        self.all_ads_details = pd.DataFrame()

        self.pg_connection_string = f'postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_database}'
        self.db_loads = 0

        self.ner = ImotNer(mode='predict')

        if not os.path.isdir('logs'):
            os.mkdir('logs')

        logging.basicConfig(filename=f"logs/run-{datetime.now().strftime('%Y%m%d%H%M%S')}.log", level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        logging.info(f"ETL started")

    def get_slinks(self):
        """
        Gets the short URL string for all imot types
        """

        logging.info("Get slinks started")
        self.imot_slinks = {}  # Re-initiate the dict, so every scheduled process gets a clean one

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
        logging.info("Get slinks completed")

    def get_all_ads(self):
        """
        Scrapes all URLs for the given types
        """

        self.all_ad_urls = []

        logging.info("Get URLs started")
        def get_slink_pages(slink):
            """
            Gets the max page number (pagination)
            """

            main_url_req = requests.get(f'{self.ads_url_2}{slink}')
            print(f'{self.ads_url_2}{slink}')
            main_url_req.encoding = self.encoding
            soup = BeautifulSoup(main_url_req.text, 'html.parser')
            # print(soup.status)
            page_numbers = soup.find('span', {'class': 'pageNumbersInfo'}).text.split(' ')[-1]
            print(page_numbers)
            return int(page_numbers)

        def get_slink_ads(slink) -> list:
            """
            """

            page_numbers = get_slink_pages(slink)
            ad_urls = []

            for page in range(1, int(page_numbers) + 1):
                # print(page)
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
            print(f"{len(ads_urls)} URLs should be added to the URL list")
            self.all_ad_urls.extend(ads_urls)
            print(f"URL list has {len(self.all_ad_urls)} URLs")
        logging.info("Get URLs completed")

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

        # print(ad_url)
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
        logging.info("Get ads content started")

        self.all_ads_details = pd.DataFrame()  # Re-initiate the df, so every scheduled run gets a fresh one

        for ad_url in self.all_ad_urls:
            self.all_ads_details = pd.concat([self.all_ads_details, self.get_ad_info(ad_url)])

            dictionary = {'\t': ''}  # Can be extended with more problematic chars
            self.all_ads_details.replace(dictionary, regex=True, inplace=True)
        logging.info("Get ads content completed")
        logging.info(f"There are {self.all_ads_details.shape[0]} ads")
        print('Scrapping all ads is done!')

        # self.all_ads_details['avg_price_per_kvm_of_sample'] = self.all_ads_details.groupby(
        #     ['ad_city',
        #      'ad_neighborhood',
        #      'ad_type'])['ad_price_per_kvm'].transform('mean')
        #
        # self.all_ads_details['price_per_kvm_gain'] = \
        #     self.all_ads_details['ad_price_per_kvm'] / self.all_ads_details['avg_price_per_kvm_of_sample'] - 1
        # self.all_ads_details['price_per_kvm_gain'] = round(self.all_ads_details['price_per_kvm_gain'] * 100, 2)

    def write_to_db(self, table_name_latest, table_name_history, drop=False):
        """
        The main idea is:
        - Load unique hashes of the ads to DB
        - Filter out only the hashes that are not existing in the history table
        - Extract back the 'new' ad_ids
        - Apply the NER model only to 'new' ads (We don't want to apply the NER more than once on the same ad)
        - Load the 'new' ads back
        - Truncate the latest table and recreate it as UNION of 'new' and 'old' ads from the history table

        :param table_name_latest:
        :param table_name_history:
        :param drop:
        :return:
        """
        logging.info("Load to DB started")
        engine = create_engine(self.pg_connection_string)
        conn = engine.raw_connection()

        start_timestamp = datetime.now()
        processed_records = self.all_ads_details.shape[0]
        imot_types = ', '.join(self.all_ads_details['ad_type'].unique())

        with conn.cursor() as cur:

            if drop and self.db_loads == 0:
                cur.execute(f"""
                DROP TABLE IF EXISTS {table_name_latest};
                DROP TABLE IF EXISTS {table_name_history};
                DROP TABLE IF EXISTS etl_tracker;
                """)
                print('Tables have been dropped!')
            conn.commit()

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS etl_tracker (
                    id SERIAL PRIMARY KEY,
                    start_timestamp TIMESTAMP,
                    end_timestamp TIMESTAMP,
                    processed_records INTEGER,
                    imot_types VARCHAR(500));
            
                CREATE TABLE IF NOT EXISTS {table_name_latest} (
                    ad_url VARCHAR(500),
                    ad_city VARCHAR(500),
                    ad_neighborhood VARCHAR(500),
                    ad_type VARCHAR(50),
                    ad_description VARCHAR(5000),
                    ad_hash VARCHAR(100),
                    ad_price INT,
                    ad_currency VARCHAR(20),
                    ad_kvm INT,
                    ad_price_per_kvm DECIMAL(12, 2),
                    ad_street VARCHAR(500),
                    updated_ts TIMESTAMP,
                    locations VARCHAR(500),
                    
                    UNIQUE (ad_url));
                    
                CREATE TABLE IF NOT EXISTS {table_name_history} (
                    ad_url VARCHAR(500),
                    ad_city VARCHAR(500),
                    ad_neighborhood VARCHAR(500),
                    ad_type VARCHAR(50),
                    ad_description VARCHAR(5000),
                    ad_hash VARCHAR(100),
                    ad_price INT,
                    ad_currency VARCHAR(20),
                    ad_kvm INT,
                    ad_price_per_kvm DECIMAL(12, 2),
                    ad_street VARCHAR(500),
                    updated_ts TIMESTAMP,
                    locations VARCHAR(500),
                    
                    UNIQUE (ad_hash));
                """)
            conn.commit()

            cur.execute(
                f"""
                 CREATE TEMPORARY TABLE ads_latest_hashes (
                    ad_url VARCHAR(500),
                    ad_hash VARCHAR(100)
                    );
                """)

            print(f"Ads to process: {self.all_ads_details.shape[0]}")
            out_hashes = io.StringIO()
            self.all_ads_details[['ad_url', 'ad_descr_hash']].to_csv(out_hashes, sep='\t', header=False, index=False)
            out_hashes.seek(0)
            cur.copy_from(out_hashes, 'ads_latest_hashes')
            conn.commit()

            keep_ads_sql = """
            SELECT DISTINCT ad_url
            FROM ads_latest_hashes
            WHERE ad_hash NOT IN (SELECT DISTINCT ad_hash FROM ads_history) 
            """

            self.keep_ads = list(pd.read_sql(keep_ads_sql, conn)['ad_url'])
            self.new_ads_details = self.all_ads_details.query("ad_url in @self.keep_ads")
            logging.info(f"There are {self.new_ads_details.shape[0]} new ads")

            # Put this in a ImotNer method
            self.new_ads_details['locations'] = ''
            res = []
            r = self.new_ads_details.shape[0]
            for i in range(r):
                print(i)
                s = self.new_ads_details.iloc[i]['ad_description']
                res.append(self.ner.predict(s))

            self.new_ads_details.loc[0:r, 'locations'] = res

            cur.execute(
                f"""
                CREATE TEMPORARY TABLE tmp (LIKE {table_name_history})
                """
            )

            output = io.StringIO()
            self.new_ads_details.to_csv(output, sep='\t', header=False, index=False)
            output.seek(0)

            cur.copy_from(output, 'tmp')

            cur.execute(f"""
            TRUNCATE {table_name_latest}
            """)

            cur.execute(
                f"""
                INSERT INTO {table_name_latest}
                SELECT * FROM tmp
                
                UNION
                
                SELECT * 
                FROM  ads_history 
                WHERE 
                    ad_hash IN (SELECT ad_hash FROM ads_latest_hashes)
                    AND
                    ad_hash NOT IN (SELECT ad_hash FROM tmp);
                """
            )

            cur.execute(
                f"""
                INSERT INTO {table_name_history}
                SELECT * FROM {table_name_latest}
                ON CONFLICT (ad_hash) 
                DO NOTHING
                """
            )

            cur.execute(f"DROP TABLE tmp")
            conn.commit()

            end_timestamp = datetime.now()
            etl_q = """INSERT INTO etl_tracker (start_timestamp, end_timestamp, processed_records, imot_types) 
                       VALUES (%s, %s, %s, %s)"""
            cur.execute(etl_q, (start_timestamp, end_timestamp, processed_records, imot_types))
            conn.commit()

        conn.close()
        self.db_loads += 1
        logging.info("Load in DB completed")

if __name__ == '__main__':
    imot = ImotScraper()
    imot.get_slinks()
    # print(imot.imot_slinks)
    imot.get_all_ads()
    imot.get_all_ads_info()
    imot.write_to_db('ads_latest', 'ads_history', drop=False)