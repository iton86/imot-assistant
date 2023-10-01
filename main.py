import importlib
# from settings import *
import settings as s
importlib.reload(s)

import settings


from imot_scraper import ImotScraper

if __name__ == '__main__':
    imot = ImotScraper(s.MAIN_URL, s.IMOT_CITIES, s.IMOT_REGIONS, s.IMOT_TYPES, s.PAYLOAD, s.HEADERS, s.ADS_URL_2, s.ENCODING)
    imot.get_slinks()
    print(imot.imot_slinks)
    imot.get_all_ads()




