# One must first navigate to https://www.bbc.co.uk/food/sitemap.xml to download
# the sitemap from bbc food (and put it in the data folder).
# Then this script can be run to extract the ingredients and images.
from bs4 import BeautifulSoup as bs
from typing import List
from tqdm import tqdm
import requests
from PIL import Image, UnidentifiedImageError
import io
import json
import csv

from src.file_locations import (
    BBC_INGREDIENTS_FPATH,
    DATA_DIR,
    BBC_INGREDIENTS_FPATH,
    BBC_IMAGES_DIR
)


def main():
    valid_recipes_urls = extract_all_valid_recipes_URLS()
    pbar = tqdm(valid_recipes_urls)
    for url in pbar:
        recipe_id = url.split("/")[-1]
        pbar.set_description(recipe_id)
        save_single_recipe(url, recipe_id)


def save_single_recipe(url, recipe_id):
    page = requests.get(url)
    soup = bs(page.content, "html.parser")
    if get_and_save_image_data(recipe_id, soup):
        save_ingredients_list(soup, recipe_id)


def save_ingredients_list(soup, recipe_id):
    ingredients_list = json.loads(
        soup.find("script", type="application/ld+json").text)["recipeIngredient"]
    ingredients_list = [recipe_id] + ingredients_list
    with open(BBC_INGREDIENTS_FPATH, "a") as f:
        wr = csv.writer(f)
        wr.writerow(ingredients_list)


def get_and_save_image_data(recipe_id, soup):
    possible_urls = [x for x in soup.find_all("img") if x.get("src", None)
                        and recipe_id in x["src"]]
    if not possible_urls:
        return
    img_data = None
    for img_url in possible_urls[::-1]: # Start at the back for the largest
        img_data = requests.get(img_url["src"]).content
        try:
            image = Image.open(io.BytesIO(img_data))
        except UnidentifiedImageError:
            continue
        if image.mode not in ["P", "CMYK"]:
            break
        # Image doesn't exist for requested recipe
        img_data = None
        continue
    if img_data:
        with open(BBC_IMAGES_DIR / f"{recipe_id}.jpg", "wb") as f:
            f.write(img_data)
    return img_data


def extract_all_valid_recipes_URLS() -> List[str]:
    fname = "BBC_good_food_sitemap.xml"
    fpath = DATA_DIR / fname
    with open(fpath, "r") as f:
        content = f.readlines()
        context = "".join(content)
        bs_content = bs(context, "lxml")
    all_locs = bs_content.find_all("loc")

    valid_recipes_urls = []
    for loc in all_locs:
        url = loc.text
        if "/food/recipes/" not in url:
            continue
        valid_recipes_urls.append(url)
    return valid_recipes_urls


if __name__ == "__main__":
    main()