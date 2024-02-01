from faker import Faker
import csv
import random

fake = Faker()

crops = ['Wheat', 'Cotton', 'Mustard', 'Bajra', 'Rice', 'Sugarcane']

num_data_points = 20000

file_path = 'generated_data.csv'

header = ['Farm ID', 'Soil Type', 'Soil pH', 'Organic Matter (%)', 
          'Nitrogen Level (ppm)', 'Phosphorus Level (ppm)', 'Potassium Level (ppm)', 
          'Temperature (°C)', 'Rainfall (mm)', 'Crop Recommended']

with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    
    for i in range(num_data_points):
        farm_id = i + 1
        soil_type = fake.random_element(elements=('Loamy', 'Sandy', 'Clayey'))
        soil_ph = round(random.uniform(6, 7.5), 1)
        organic_matter = round(random.uniform(1.5, 3.5), 1)
        nitrogen_level = random.randint(18, 30)
        phosphorus_level = random.randint(10, 20)
        potassium_level = random.randint(25, 35)
        temperature = random.randint(25, 30)
        rainfall = random.randint(700, 900)
        crop_recommended = fake.random_element(elements=crops)
        
        writer.writerow([farm_id, soil_type, soil_ph, organic_matter, nitrogen_level, 
                         phosphorus_level, potassium_level, temperature, rainfall, crop_recommended])

print(f"Sample data has been generated and saved to '{file_path}'")
