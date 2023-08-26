import requests
from settings import *
from bs4 import BeautifulSoup

class ImotScraper:

    """
    Scrapes the list of URLs
    """

    def __init__(self, main_url, imot_types, payload, headers, ads_url_2, encoding):
        """

        :param main_url:
        :param imot_types:
        :param payload:
        :param headers:
        :param imot_slinks:
        """

        self.main_url = main_url
        self.imot_types = imot_types
        self.payload = payload
        self.headers = headers
        self.encoding = encoding
        self.ads_url_2 = ads_url_2

        self.imot_slinks = {}
        self.all_ad_urls = []

    def get_slinks(self):
        """
        Gets the short URL string for all imot types
        """

        for imot_type in self.imot_types:
            payload = self.payload.format(imot_type)
            headers = self.headers

            response = requests.request("POST",
                                        self.main_url,
                                        headers=headers,
                                        data=payload.encode('utf-8'))

            s = response.text.find("slink")
            e = response.text.find("';", s)
            slink = response.text[s + 6:e]
            self.imot_slinks[imot_type] = slink

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
                print(page_numbers)
                p = f'https://www.imot.bg/pcgi/imot.cgi?act=3&slink={slink}&f1={page}'
                main_url_req = requests.get(p)
                main_url_req.encoding = self.encoding
                soup = BeautifulSoup(main_url_req.text, 'html.parser')

                hrefs = soup.find_all('a', href=True)
                fixed_str = r'imot.bg/pcgi/imot.cgi?act='
                # ad_urls.extend(
                #     list(set([_['href'] for _ in hrefs if fixed_str in _['href'] and
                #               '&adv=' in _['href']])))

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


imot = ImotScraper(MAIN_URL, IMOT_TYPES, PAYLOAD, HEADERS, ADS_URL_2, ENCODING)
imot.get_slinks()
print(imot.imot_slinks)
imot.get_all_ads()