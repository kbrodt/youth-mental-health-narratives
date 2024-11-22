import argparse
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://www.drivendata.org/competitions/295/submissions/code/"
user = "<USER>"
password = """<PASSWORD>"""
chromedriver_path = ""


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--note",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default=URL,
    )

    args = parser.parse_args(args)

    return args


def main():
    args = parse_args()
    submission = Path(args.submission).absolute()
    assert submission.is_file(), f"No file {submission}"

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    delay = 5
    upload_delay = 5 * 60 * 60
    with webdriver.Chrome(options=options) as driver:
        driver.get(args.url)
        elem = driver.find_element(By.NAME, "login")
        elem.send_keys(user)
        elem = driver.find_element(By.NAME, "password")
        elem.send_keys(password)
        elem.send_keys(Keys.RETURN)

        elem = WebDriverWait(driver, delay).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="inner-header"]/div/div/button')))
        elem.send_keys(Keys.RETURN)

        elem = WebDriverWait(driver, delay).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="id_form"]/div[2]/fieldset/div/label/input')))
        elem.send_keys(str(submission))

        elem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.XPATH, '//*[@id="id_note"]')))
        elem.send_keys(f"{args.note}")

        elem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.XPATH, '//*[@id="id_form"]/div[3]/div/button[1]')))
        elem.send_keys(Keys.RETURN)
        elem = WebDriverWait(driver, upload_delay).until(EC.invisibility_of_element_located((By.XPATH, '//*[@id="id_form"]/div[3]/div/div/div[2]/progress')))


if __name__ == "__main__":
    main()
