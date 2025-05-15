import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

# Load and filter the dataset for tax, land and prop and financial case type as theyre apt for commercial courts
df = pd.read_parquet("indiankanoon1.parquet")
df_tax = df[df["Case_Type"].str.contains("Tax|Land&Property|Financial", case=False, na=False)].copy()

# Setin up Selenium headless browser
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(), options=chrome_options)
wait = WebDriverWait(driver, 10)

#field names and their corresponding titles
SECTION_MAP = {
    "Facts":        {"checkbox": "structchk0", "title": "Fact"},
    "Issues":       {"checkbox": "structchk1", "title": "Issue"},
    "PetArg":       {"checkbox": "structchk2", "title": "Petitioner's Argument"},
    "RespArg":      {"checkbox": "structchk3", "title": "Respondent's Argument"},
    "Section":      {"checkbox": "structchk4", "title": "Analysis of the law"},
    "Precedent":    {"checkbox": "structchk5", "title": "Precedent Analysis"},
    "CDiscource":   {"checkbox": "structchk6", "title": "Court's Reasoning"},
    "Conclusion":   {"checkbox": "structchk7", "title": "Conclusion"},
}

def extract_structured_sections(url):
    try:
        driver.get(url)
        
        # Wait until page is loaded and checkboxes are present
        wait.until(EC.presence_of_element_located((By.ID, "structchk0")))

        all_checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
        print("Found checkboxes:")
        for cb in all_checkboxes:
            print(f" - id: {cb.get_attribute('id')}, displayed: {cb.is_displayed()}, enabled: {cb.is_enabled()}")

        for section_name, section in SECTION_MAP.items():
            try:
                print(f"\n🔍 Attempting: {section_name} → Checkbox ID: {section['checkbox']}")
                print(f"🌐 Current URL: {driver.current_url}")
                
                checkboxes = driver.find_elements(By.ID, section["checkbox"])
                if not checkboxes:
                    print(f"⚠️ Checkbox {section['checkbox']} not found on page, skipping.")
                    continue

                checkbox = checkboxes[0]

                is_displayed = checkbox.is_displayed()
                is_enabled = checkbox.is_enabled()
                print(f"   → is_displayed: {is_displayed}, is_enabled: {is_enabled}")

                if is_displayed and is_enabled:
                    driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)
                    time.sleep(0.3)
                    driver.execute_script("arguments[0].click();", checkbox)
                    print(f" Clicked checkbox: {section['checkbox']}")
                else:
                    print(f"   ⚠️ Checkbox {section['checkbox']} is not interactable")

            except Exception as e:
                print(f"Error clicking {section['checkbox']}: {e}")

        # Wait for content to load
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        result = {}

        for paragraph in soup.find_all("p"):
            print(paragraph.get("title"))

        for key, section in SECTION_MAP.items():
            try:
                paragraph = soup.find("p", title=section["title"])

                if paragraph:
                    result[key] = paragraph.get_text(strip=True)
                    print(f"   - {key}: {'Extracted' if result[key] else 'No content found'}")
                else:
                    result[key] = None
                    print(f"   - {key}:No matching <p> tag found with title '{section['title']}'")

            except Exception as e:
                result[key] = None
                print(f"⚠️ Error extracting section {key}: {e}")

        return result

    except Exception as e:
        print(f"Error extracting from {url}: {e}")
        return {key: None for key in SECTION_MAP}

for col in SECTION_MAP:
    df_tax[col] = None

for idx, row in tqdm(df_tax.iterrows(), total=len(df_tax)):
    url = row["Doc_url"]
    scraped_data = extract_structured_sections(url)
    for key in SECTION_MAP:
        df_tax.at[idx, key] = scraped_data[key]

df_tax.drop(columns=["Text"], inplace=True, errors='ignore')

df_tax.to_csv("formatted_tax_cases.csv", index=False)

print("formatted data saved to 'formatted_tax_cases.csv'")