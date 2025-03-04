# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:52:44 2025

@author: USER
"""
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import numpy as np
import time
from datetime import datetime, timedelta
from selenium.common.exceptions import TimeoutException
import pandas as pd
import re


#%%
def get_cngovMD(page_num = 8):
    arr = []
    i = 0
    for i in range(page_num):
        if i == 0 :
            url = 'http://www.npc.gov.cn/npc/c1773/c16074/c8494/c8495/c9317/index.html'
        else :
            url = f'http://www.npc.gov.cn/npc/c1773/c16074/c8494/c8495/c9317/index_{i}.html'
        
        response = requests.get(url)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        
        for item in soup.find_all('li'):
            arr.append(item.text.split()[-1])
        
        i = i+1
        
    return np.array(arr)

#%%
def close_ad(driver):
    try:
        # Wait for a moment for the ad to appear
        time.sleep(2)  # Adjust time as necessary
        close_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="hk-cs-openaccount"]/div/div[1]/div[2]')))

        close_button.click()
    except:
        pass
#%%    
def convert_to_date(input_str):
    # Extract the date/time part (after "·")
    input_str = str(input_str)
    tmp = input_str.find("·")
    input_str = input_str[tmp:]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for month in months:
        start_index = input_str.find(month)
        if start_index == -1:
            if month in ["Dec"]:
                date_time_str = input_str
            continue
        date_time_str = input_str[start_index:]
        break
    end_index = date_time_str.find("\n")
    date_time_str = date_time_str[:end_index]
    # Handle "XX minutes ago" or "HH:MM" (today's dates)
    minutes_ago_match = re.match(r"(\d+)\s+minutes? ago", date_time_str, re.IGNORECASE)
    if minutes_ago_match:
        minutes_ago = int(minutes_ago_match.group(1))
        parsed_time = datetime.now() - timedelta(minutes=minutes_ago)
        parsed_date = parsed_time.date()
        time_hour = parsed_time.hour
    else:
        try:
            # Try parsing "HH:MM" format
            parsed_time = datetime.strptime(date_time_str, "%H:%M")
            time_hour = parsed_time.hour
            parsed_date = datetime.now().date()
        except:
            parsed_date = None
            time_hour = None
    
    # Handle today's dates (time >= 16:00 → next day)
    if parsed_date:
        if time_hour >= 16:
            parsed_date += timedelta(days=1)
        return parsed_date.strftime("%Y-%m-%d")
    
    # Handle older dates (with or without year)
    try:
        # Attempt to parse with year (e.g., "Feb 27, 2024 21:43")
        parsed_datetime = datetime.strptime(date_time_str, "%b %d, %Y %H:%M")
    except ValueError:
        try:
            # Attempt to parse without year (e.g., "Feb 27 21:43")
            parsed_datetime = datetime.strptime(date_time_str, "%b %d %H:%M")
            parsed_datetime = parsed_datetime.replace(year=datetime.now().year)
            # Adjust year if future
            if parsed_datetime > datetime.now():
                parsed_datetime = parsed_datetime.replace(year=parsed_datetime.year - 1)
        except ValueError:
            # Both parsing attempts failed
            return datetime.now().strftime("%Y-%m-%d")
    
    # Apply 16:00 rule to all cases
    if parsed_datetime.hour >= 16:
        parsed_datetime += timedelta(days=1)
    
    return parsed_datetime.strftime("%Y-%m-%d")
#%%
def scroll(n = 100):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get('https://news.futunn.com/main?lang=en-us')
    
    button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="__layout"]/div/div[2]/div[3]/button')))
    button.click()
    #driver.execute_script("window.scrollTo(0, 2000);")
    time.sleep(5) 
    driver.execute_script("window.scrollTo(0, 3500);")
    
    #count = 0
    for i in range(n):
        # Scroll to the bottom
        try:
            last_height = driver.execute_script("return document.body.scrollHeight")
            
            print(last_height) 
            if last_height >= 2000061: break
            h = last_height-2000
            #j = (i+1)*1000+3000
            driver.execute_script(f'window.scrollTo(0, {last_height})')
            time.sleep(5)
            driver.execute_script(f"window.scrollTo(0, {h});")
            time.sleep(1)  # Wait for content to load
            close_ad(driver)
        except TimeoutException:
            #if count == 2: 
                #driver.close
                #break
            #count = count+1
            close_ad(driver)
            continue
    
    return driver
        
#%%
def get_news(driver, news_file = None):
    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')
    
    a = soup.find_all('a', {"class": "market-item list-item"})
    
    title = []
    footer = []
    for item in a:
        t = item.find('h2',{"class":"market-item__title"}).get_text()
        f = item.find('div',{"class":"market-item__footer"}).get_text()
        title.append(t)
        footer.append(convert_to_date(f))# 
    
    
    if news_file is None:
        df = pd.DataFrame({
            'Date': footer,
            'Header': title
        })
        
    
    else:
        df = news_file
        i = 0
        for item in title:
            if item not in df['Header'].values:
                new_row = pd.DataFrame({'Date': [footer[i]], 'Header': [item]})
                df = pd.concat([new_row,df], ignore_index=True)
        i = i+1
    
    return df
    

    