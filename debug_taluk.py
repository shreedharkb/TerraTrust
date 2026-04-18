import geopandas as gpd

gdf = gpd.read_file('data/Taluk.shp').to_crs(epsg=4326)

# Davangere district code is 14 (from the district-level row)
davangere_taluks = gdf[(gdf['KGISDistri'] == '14') & (gdf['KGISTalukN'].notna())]
print(f"Davangere taluks found: {len(davangere_taluks)}")
print(davangere_taluks[['KGISDistri','KGISTalukC','KGISTalukN','LGD_TalukC']].to_string())

# Print centroids
for _, row in davangere_taluks.iterrows():
    c = row.geometry.centroid
    print(f"  {row['KGISTalukN']}: ({c.y:.4f}N, {c.x:.4f}E)")
