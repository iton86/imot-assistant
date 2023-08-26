from settings import *
from imot_scraper import ImotScraper

if __name__ == '__main__':
    imot = ImotScraper(MAIN_URL, IMOT_TYPES, PAYLOAD, HEADERS, ADS_URL_2, ENCODING)
    imot.get_slinks()
    print(imot.imot_slinks)
    imot.get_all_ads()