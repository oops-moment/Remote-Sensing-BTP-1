var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');

// Load your shapefile
var shapefile = ee.FeatureCollection('projects/ee-dhingrabilli/assets/kws');

// Set view center and zoom (12)
Map.centerObject(shapefile, 11);

var spatialFiltered = l8.filterBounds(shapefile);
var maxCloudCover = 20; 
// Grab images from the year 2022
var images_2022 = spatialFiltered.filterDate('2022-01-01', '2022-12-31').filter(ee.Filter.lt('CLOUD_COVER', maxCloudCover));
// define function for masking clouds out of images (from SR images)
var maskClouds = function(image) {
  // Bit 0 - Fill
  // Bit 1 - Dilated Cloud
  // Bit 2 - Cirrus
  // Bit 3 - Cloud
  // Bit 4 - Cloud Shadow
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);

  // Apply the scaling factors to the appropriate bands.
  var opticalBands = image.select('SR_B.')
  var thermalBands = image.select('ST_B.*')
  
  // Replace the original bands with the scaled ones and apply the masks.
  return image.addBands(opticalBands, null, true)
      .addBands(thermalBands, null, true)
      .updateMask(qaMask)
      .updateMask(saturationMask);
}

// define collections of masked images
var images_2022_masked = images_2022.map(maskClouds);

// load in Mangroves collection (consists of 1 image)
var mangroves_coll = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS');
var mangroves = mangroves_coll.first().select(['1'], ['Mangroves']).clip(shapefile); // grab the image and rename the band
mangroves = mangroves.unmask(0);

// Rename the band to 'Mangroves'
mangroves = mangroves.select([0], ['Mangroves']).clip(shapefile);
var mangrovesVis = {min: 0, max: 1.0, palette: ['000000','d40115']};
Map.addLayer(mangroves, mangrovesVis, 'Mangroves in Rectangle', false);
// Visualize each image in the collection
images_2022_masked.getInfo(function(images) {
  images.features.forEach(function(feature) {
    var image = ee.Image(feature.id);
    var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo();

    var visParams = {bands: ['SR_B6','SR_B4','SR_B7'], min: 0, max: 20000};
    Map.addLayer(image.clip(shapefile), visParams, 'image'+date, false);
    var image_with_mangroves = image.addBands(mangroves, ['Mangroves']).clip(shapefile);
    print('Full 2022 composite'+date, image_with_mangroves);
        // Export 2022 image, with all bands, casted to float
    Export.image.toDrive({
      image: image_with_mangroves.select('SR_B4', 'SR_B6', 'SR_B7', 'Mangroves').toFloat(),
      description: 'krishna'+date,
      scale: 30,
      region: shapefile, // .geometry().bounds() needed for multipolygon
      folder: 'MangroveClassification',
      maxPixels: 1e13
    });
  });
});

