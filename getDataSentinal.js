var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');

var shapefile = ee.FeatureCollection('projects/ee-dhingrabilli/assets/kws');

Map.centerObject(shapefile, 11);

var spatialFiltered = sentinel2.filterBounds(shapefile);

var maxCloudCover = 5;

var images_2022 = spatialFiltered
  .filterDate('2022-01-01', '2022-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxCloudCover));

function maskClouds(image) {
      var scl = image.select('SCL');
      var clear_sky_pixels = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(11));
      return image.updateMask(clear_sky_pixels).divide(10000);
}

var images_2022_masked = images_2022.map(maskClouds);

var mangroves_coll = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS');
var mangroves = mangroves_coll.first().select(['1'], ['Mangroves']).clip(shapefile); // grab the image and rename the band
mangroves = mangroves.unmask(0);

// Rename the band to 'Mangroves'
mangroves = mangroves.select([0], ['Mangroves']).clip(shapefile);
var mangrovesVis = {min: 0, max: 1.0, palette: ['000000','d40115']};
Map.addLayer(mangroves, mangrovesVis, 'Mangroves in Rectangle', false);
print(images_2022_masked)

var i=0
images_2022_masked.getInfo(function(images) {
  images.features.forEach(function(feature) {
    var image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/'+feature.properties['system:index']);
    image=maskClouds(image)
    var visParams = {bands: ['B11','B12','B4'], min: 0, max: 0.3};
    Map.addLayer(image.clip(shapefile), visParams, 'image'+i, false);
    var image_with_mangroves = image.addBands(mangroves, ['Mangroves']).clip(shapefile);
    print('Full 2022 composite'+i, image_with_mangroves);
        // Export 2022 image, with all bands, casted to float
    Export.image.toDrive({
      image: image_with_mangroves.select('B11', 'B12', 'B4','B5','B6','B7','B8A', 'Mangroves').toFloat(),
      description: 'krishna'+i,
      scale: 30,
      region: shapefile, // .geometry().bounds() needed for multipolygon
      folder: 'MangroveClassification',
      maxPixels: 1e13
    });
    i=i+1
  });
});