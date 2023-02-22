"""
Create map for a figure of the federated learning network

Script sources:
- https://github.com/xmba15/gadm
- https://geopandas.org/en/stable/gallery/create_geopandas_from_pandas.html
- https://www.latlong.net/
"""

import os
import argparse
import pandas as pd
import geopandas as gpd
from gadm import GADMDownloader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, help='Absolute path to output directory to write figure to')
    args = parser.parse_args()

    fig, ax = plt.subplots()
    color = "#DCDCDC"
    countries = ["Belgium", "Czechia", "Germany"]

    # Plot countries
    # Source: https://github.com/xmba15/gadm
    downloader = GADMDownloader(version="4.0")
    for country in countries:
        ad_level = 0
        gdf = downloader.get_shape_data_by_country_name(country_name=country, ad_level=ad_level)
        assert isinstance(gdf, gpd.GeoDataFrame)
        gdf.plot(ax = ax, color = color, edgecolor='white', linewidth = 0.3)

    # Plot cities
    # Source: https://geopandas.org/en/stable/gallery/create_geopandas_from_pandas.html
    df = pd.DataFrame(
        {'City': ['Brussels', 'Prague', 'Greifswald'],
         'Country': ['Belgium', 'Czechia', 'Germany'],
         'Latitude': [50.850346, 50.075539, 54.097198],
         'Longitude': [4.351721, 14.437800, 13.387940]})

    cities_xy = gpd.points_from_xy(df.Longitude, df.Latitude)
    for city_xy in cities_xy:
        ax.scatter(city_xy.x, city_xy.y, color = '#646464')

    # Save figure
    plt.axis('off')
    fig.savefig(os.path.join(args.output_path, 'fl_map.png'), dpi = 300, transparent=True)
