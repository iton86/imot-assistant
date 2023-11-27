MAIN_URL = "https://www.imot.bg/pcgi/imot.cgi"
ADS_URL_0 = 'https://www.imot.bg/pcgi/imot.cgi?act=3&slink={}&f1={}'
ADS_URL_1 = 'https://imoti-sofia.imot.bg/pcgi/imot.cgi?'
ADS_URL_2 = 'https://imot.bg/'
ENCODING = 'windows-1251'
# IMOT_TYPES = [1, 2, 3, 8, 9]
IMOT_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
IMOT_CITIES = ['град София']
# IMOT_REGIONS = ['Център', 'Манастирски ливади']
IMOT_REGIONS = ['Център']
PAYLOAD = "act=3&rub=1&rub_pub_save=1&topmenu=2&actions=1&f0=130.204.13.200&f1=1&f2=&f3=&f4=1&f7={}~&f28=&f29=&f43=&f44=&f30=EUR&f26=&f27=&f41=1&f31=&f32=&f54=&f38={}&f42=&f39=&f40={}&fe3=&fe4=&f45=&f46=&f51=&f52=&f33=&f34=&f35=&f36=&f37=&fe2=1&fe7=1"
HEADERS = {
                'Connection': 'keep-alive',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
                'sec-ch-ua': '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
                'sec-ch-ua-mobile': '?0',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
                'sec-ch-ua-platform': '"Windows"',
                'Content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'Accept': '*/*',
                'Origin': 'https://www.imot.bg',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Dest': 'empty',
                'Referer': 'https://www.imot.bg/pcgi/imot.cgi?act=5&adv=1e162452389677559',
                'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,bg;q=0.7',
                'Cookie': '_ga=GA1.2.250659246.1618746999; fbm_576877119001657=base_domain=.www.imot.bg; c_LikeBox=1; __gfp_64b=5BzA7crfuZduy2684b8jcRPltR2Kxb66R720qL_EHJr.R7|1618475128; _gid=GA1.2.1823822361.1637394514; fe2c=1; fbsr_576877119001657=A5NVFmS6Un6etLiWnMdiX3fjKHovrA3iBIiHOQljNmw.eyJ1c2VyX2lkIjoiMTAwMDAwNTAyNDUxMDEyIiwiY29kZSI6IkFRQ0VCdno4VHNQeGtuVHRtS3JUSWRsMHp6eHZ0TEhoX19YQzFJSDNyQ01CaHUzcVlOSFF5R0hpazVoVExDNzVOM0RRbENhRms1dmxEanhJbThsSy1oZUwwelk3WFVGcTFlYXlHX0cxXzNEek1Ja0hXc0lPN0dFWVNwOWoyR2pQeXo1cGpBdDdNdWZYbE9FVHJGb3dobEY3ZkxhY01oRkpSZjBLRXBwSVBfbEFSY3VKRDZZd0pVNXVlbV9JbHU5U2NNS3V2REtrSm9vMHFCbWpoX0Jqb3FmbDA1dzAzWDVuM3RNYndYR2t1dnl1UkRoTjlySHFUWHg3Q2NrbV81MUJsWjJhZUtUcFdheWNvRlB4NzlHSEdtcjVfWlhtaU44QkVrSXRZYTlYMGVzdE5rMTl2WjJDRW5VWFJRa1ZLbk5SN3VGWWN3aHpxWlMzYnNsSTlqUVMtekVlTXN4SVgzYzNlaFRscGxmMUJBaHk5UUh6VG1iM2c4YWNiWVVqcklnNGtPTSIsImFsZ29yaXRobSI6IkhNQUMtU0hBMjU2IiwiaXNzdWVkX2F0IjoxNjM3NDM5NjAwfQ; imot_session=e55e4f187299caa12aaf2ff5e5be047f; imot_session_redirect=; FCCDCF=[null,null,["[[],[],[],[],null,null,true]",1637442218507],["CPM0yZuPM0yZuEsABBBGBsCoAP_AAG_AAAIwHaBTxDrFIGlCSG5IUisQAAADQUCsAwQBBACBAEABAAIAIAQCkkAYBACgAAACIAAAIBZBAAIAAEgAAQABQAAAABEIAAAAAAAIIAAAAAAAAAAIAAACAAAAAAAIAAAAAAAAgQBAAIIAAhAAhAAQAAAIAAAAAAAAAAAAQO0AniFWKQNKEkNiQIFYgAAAGgoEYBgACCAECAIAAAAQAQAgFIIAwAAFAAAAEQAAAQCwCAAQAAJAAAgACgAAAACAQAAAAAAAQQAAAAAAAAAAQAAAEAAAAAAAQAAAAAAABAgCAAQQABCAAAAAgAAAQAAAAAAAAAAAAgAA","1~867.1716.46.1375.1188.574.66.70.1732.1143.93.864.2526.874.2109.108.122.1878.440.1963.89.2072.241.149.338.253.259.2186.1027.322.1449.1205.371.1699.394.1415.415.1721.1870.1419.2088.1911.2729.486.1558.495.494.1697.2481.2677.1800.1889.584.323","66E3FD32-9544-442D-ABD0-3578F0432067"],null,null,[]]; _gat_gtag_UA_1160575_1=1; FCNEC=[["AKsRol-XxS75bvRhXO-PuFmxF1Sjukcepz8kCKrQsAWEva3S03inbUVN3U9qbdqmC9hf1tAXRb83aKbkB7BYuFIZBT-kwmgMc2ADMeQnJm48G5WwxqvnrlI5K2jNNUOUZZxSmQJYHqWB7m8uEX93qYuGvAKGUvyFug=="],null,[]]'
            }
TABLE_NAME_LATEST = 'ads_latest'
TABLE_NAME_HISTORY = 'ads_history'
DROP = True  # Will drop TABLE_NAME_LATEST and TABLE_NAME_HISTORY
OPTIMIZED_SCRAPE = True
CLEAR_FOLDERS = True  # Deletes logs and jpgs folders
MAX_ADS_TO_SCRAPE = 150  # set to -1 to scrape everything
