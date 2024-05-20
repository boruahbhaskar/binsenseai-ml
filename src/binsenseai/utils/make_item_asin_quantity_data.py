import json


instance_file = "./artifacts/data_transformation/instances.json"

# Load the JSON data
with open(instance_file) as f:
    data = json.load(f)

# Convert the JSON data to the desired format
output_data = []
for asin, item in data.items():
    output_item = {
        "asin": asin,
        "name": item["name"],
        "quantity": item["quantity"],
        "bin_image_list": item["bin_image_list"]
    }
    output_data.append(output_item)

# Save the converted data to a new JSON file
with open('./artifacts/data_transformation/item_asin_quantity.json', 'w') as f:
    json.dump(output_data, f, indent=4)