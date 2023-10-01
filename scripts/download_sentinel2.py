import ee
import time

# Region 1-4 coverage
BOUNDS_r14 = [
    -53.00,
    64.00,
    -44.00,
    70.00,
]

# Region 5-6 coverage
BOUNDS_r56 = [
    -34.00,
    77.00,
    -20.00,
    80.50,
]

COUNTRY = "GL"  # See https://en.wikipedia.org/wiki/List_of_FIPS_country_codes
BORDER_DATASET = "USDOS/LSIB_SIMPLE/2017"
SATELLITE_DATASET = "COPERNICUS/S2_SR_HARMONIZED"

BANDS = {
    # "B1": "Aerosols",
    # "B2": "Blue",
    "B3": "Green",
    # "B4": "Red",
    # "B5": "Red Edge 1",
    # "B6": "Red Edge 2",
    # "B7": "Red Edge 3",
    "B8": "NIR",
    # "B8A": "Red Edge 4",
    # "B9": "Water vapor",
    # "B11": "SWIR 1",
    # "B12": "SWIR 2",
}

# Trigger the authentication flow.
# ee.Authenticate()

# Initialize Google Earth Engine library
ee.Initialize()


def get_country_geometry(country_name, countries_borders_dataset=BORDER_DATASET):
    """
    Get a country geometry
    :param country_name: the country name
    :param countries_borders_dataset: the countries borders dataset to use - See https://developers.google.com/earth-engine/datasets/catalog/USDOS_LSIB_SIMPLE_2017
    :return: the country geometry
    """

    world_countries = ee.FeatureCollection(countries_borders_dataset)
    filter_country = ee.Filter.eq("country_co", country_name)
    country = world_countries.filter(filter_country)

    return country.geometry()


def get_clear_imagery(image):
    """
    mask clouds in a Sentinel-2 image.
    See: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    and here for the collection 2 QA60 bits: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    :param image: the image
    :return: image without clouds
    """

    # Bits 9 and 10 are clouds and cirrus, respectively.
    cloud_shadows_bit_mask = 1 << 3
    clouds_low_probability_bit_mask = 1 << 7
    clouds_medium_probability_bit_mask = 1 << 8
    clouds_high_probability_bit_mask = 1 << 9
    cirrus_bit_mask = 1 << 10
    # Get the pixel QA band.
    qa = image.select("QA60")
    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(clouds_high_probability_bit_mask).eq(0)
        and qa.bitwiseAnd(cloud_shadows_bit_mask).eq(0)
        # and qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask)


def create_image_collection(
    start_date,
    end_date,
    satellite_dataset,
    bounds,
    bands,
):
    """
    Create an image collection containing satellite image in REGION_RECTANGLE and within the time range.
    Warning: the end_date is excluded
    Min start date is '2013-04-11'
    Max end date is '2021-01-22'
    :param start_date: the start date
    :param end_date: the end date
    :param satellite_dataset: the dataset to use for the imagery
    :param bounds: the bounds to use to get the imagery
    :param bands: sensor bands
    :return: the image collection
    """
    return (
        ee.ImageCollection(satellite_dataset)
        .filterDate(start_date, end_date)
        .filterBounds(ee.Geometry.Rectangle(bounds))
        .map(get_clear_imagery)
        .select(list(bands.keys()), list(bands.values()))
    )


def image_collection_to_median_image(
    image_collection,
    country_name,
    bounds,
):
    """
    Reduce an image collection to a median image and
    keep only parts of the image within REGION_RECTANGLE and the country shape
    :param image_collection: the image collection
    :param country_name: the name of the country
    :param bounds: the bounds to use to get the imagery
    :return: the median image
    """
    # Reduce collection by median
    # and clip image to keep only the region of interest that is inside the country
    image_collection = image_collection.median().clip(ee.Geometry.Rectangle(bounds))
    if country_name != "":
        image_collection = image_collection.clip(get_country_geometry(country_name))
    return image_collection


# If there is multiple image collections to reduce into one image, do it as follow :
# merged_image_collections = image_collection_2014 \
#     .merge(image_collection_2015) \
#     .merge(image_collection_2016) \
#     .merge(image_collection_2017) \
#     .merge(image_collection_2018) \
#     .merge(image_collection_2019) \
#     .merge(image_collection_2020)


def create_export_task(
    image,
    task_name,
    bounds,
):
    """
    Create an export to drive task in Google Earth Engine
    :param image: the image to export
    :param task_name: the task name, it will be used as the name of the folder to export to in drive
    :param bounds: the bounds to use to get the imagery
    :return: the task
    """
    return ee.batch.Export.image.toDrive(
        image,
        task_name,
        **{
            "folder": task_name,
            "scale": 30,
            "maxPixels": 1_000_000_000,
            "fileFormat": "GeoTIFF",
            "region": ee.Geometry.Rectangle(bounds),
        },
    )


def start_task(task):
    """
    Start a google earth engine task while keeping tracks of its state
    :param task: the task
    """
    task.start()
    while True:
        status = task.status()
        if status["state"] == "COMPLETED":
            print("COMPLETED")
            break
        elif status["state"] == "FAILED":
            print("FAILED")
            print(status["error_message"])
            break
        elif status["state"] == "CANCEL_REQUESTED":
            print("CANCEL_REQUESTED")
            break
        else:
            print("ON_GOING")
        time.sleep(30)


if __name__ == "__main__":
    """
    Download dataset tiles from Sentinel-2.
    """

    name = "sentinel2-B3-B8"
    region = "r14"
    bounds = BOUNDS_r14

    # date
    # Sentinel-2 dataset availability, see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

    # date we are interesting in
    image_date = ["2019-06-05", "2019-06-19", "2019-07-31", "2019-08-25"]

    # set range to ensure we get the proper image
    start_date = ["2019-06-04", "2019-06-18", "2019-07-24", "2019-08-23"]
    end_date = ["2019-06-06", "2019-06-22", "2019-08-01", "2019-08-27"]

    for i in range(0, 4):
        task_name = f"{name}_{image_date[i]}_{region}"

        print("Processing...")
        image_collection = create_image_collection(
            start_date=start_date[i],
            end_date=end_date[i],
            satellite_dataset=SATELLITE_DATASET,
            bounds=bounds,
            bands=BANDS,
        )
        median_image = image_collection_to_median_image(
            image_collection=image_collection, country_name=COUNTRY, bounds=bounds
        )
        export_task = create_export_task(
            image=median_image, task_name=task_name, bounds=bounds
        )
        start_task(export_task)
