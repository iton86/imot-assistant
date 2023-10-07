from imot_scraper import ImotScraper
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime


def etl_pipeline():
    print("ETL has started")
    imot = ImotScraper()
    imot.get_slinks()
    print(imot.imot_slinks)
    imot.get_all_ads()
    imot.get_all_ads_info()
    imot.write_to_db('ads_latest', 'ads_history')
    print("ETL has finished")


scheduler = BackgroundScheduler()
job_id = 'imot_etl'

scheduler.add_job(etl_pipeline, 'interval', minutes=5, id=job_id, next_run_time=datetime.now())
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



