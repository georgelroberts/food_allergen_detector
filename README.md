<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to, given either (or both) title of a dish or an image of said dish, detect whether or not it contains certain allergens.
This starts with just gluten, but will hopefully expand to other allergens.

## Prerequisites

Please find the packages in requirements.txt.
Please add the root of the directory to the python path. If using conda:
- conda install conda-build
- conda develop <root_dir_path>

## Datasets

I have built a dataset myself using BBC Good Food recipes (all with high quality images), by scraping
the image and the ingredients. Download https://www.bbc.co.uk/food/sitemap.xml and run src/bbc_good_food_scraper.py
to get the data.


### OLD
The initial dataset can be downloaded from https://www.kaggle.com/pes12017000148/food-ingredients-and-recipe-dataset-with-images. 
This contains 13,582 food images, along with an ingredient dish and a description of the dish. 
There is also a dataset AIFood that contains 370,000 food images, however this appears to not be publically available.
This dataset was found to be too noisy, with incorrect labels and low quality images.

<!-- ROADMAP -->
## Roadmap

- [] Add the title of the dish
- [] Detect lactose

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

