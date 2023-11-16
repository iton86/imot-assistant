from imot_scraper import ImotScraper
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime
import os

def etl_pipeline():
    print(f"ETL has started on {datetime.now()}\n")
    # imot = ImotScraper()
    imot.get_slinks()
    # print(imot.imot_slinks)
    imot.get_all_ads()
    imot.get_all_ads_info()
    imot.write_to_db(table_name_latest='ads_latest', table_name_history='ads_history', drop=True)
    print(f"""\nETL has finished on {datetime.now()} |
                DB load: {imot.db_loads} | Rows processed: {imot.all_ads_details.shape[0]}""")

imot = ImotScraper()

scheduler = BackgroundScheduler()
job_id = 'imot_etl'

scheduler.add_job(etl_pipeline, 'interval', minutes=10, id=job_id, next_run_time=datetime.now())
scheduler.start()


job = scheduler.get_job(job_id)
if job:
    if job.trigger is not None:
        print(f"The job with ID '{job_id}' is currently running.")
        print(f"Next run time: {job.next_run_time}")
    else:
        print(f"The job with ID '{job_id}' is scheduled but not running.")
else:
    print(f"No job with ID '{job_id}' found.")


try:
    while True:
        time.sleep(2)  # Keep the script running
except (KeyboardInterrupt, SystemExit):
    # Shut down the scheduler gracefully when interrupted
    scheduler.shutdown()



