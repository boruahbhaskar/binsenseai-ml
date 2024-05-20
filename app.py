import gradio as gr
import json 
from src.binsenseai.utils.get_relevant_images import get_images
from src.binsenseai.pipeline.predictions import PredictionPipeline
import ast
from tabulate import tabulate


# Load the items list from a JSON file
with open('./artifacts/data_transformation/item_asin_quantity.json') as f:
    data = json.load(f)


# Extract the item names from the JSON data
items = [item["name"] for item in data]


cart = []  # initialize an empty cart


# Function to add items to the cart
def add_to_cart(item_name, quantity):
    asin = next((item["asin"] for item in data if item["name"] == item_name), None)
    #item = [asin,quantity]
    item = [item_name,asin,quantity]
    if item not in cart:
        cart.append(item)
    
    return str(cart),"# Added Items to Bin \n\n| Product Name | ASIN | Quantity | \n| --- | --- | --- |\n" + "\n".join([f"| {row[0]} | {row[1]} | {row[2]} |" for row in cart])
    
    #return str(cart)


def clear_bin():
    cart.clear()
    empty_md = "### Cart Items\nYour cart is empty."
    return empty_md#str(cart)


# Function to get relevant images
def get_relevant_images(input_item_list):

    # Convert the string to a list of lists
    input_list_str = str(input_item_list)
    input_item_list_ast = ast.literal_eval(input_list_str)
    
    asin_quantity_list = [ [item[1], item[2]] for item in input_item_list_ast] 

    
    relevant_image = get_images(asin_quantity_list)
    image_path = relevant_image['image_path']
    image_name = relevant_image['image_name']
    return image_path, image_name


# Function to validate_item_bin_image using CV model
def validate_item_bin_image(image_name_output,cart_text):
    #relevant_image = get_images(input_item_list)
    obj = PredictionPipeline()
    
    # Convert the string to a list of lists
    cart_text = ast.literal_eval(cart_text)
    
    cart_text = [ [item[1], item[2]] for item in cart_text] #list(map(lambda x: x[1:x], cart_text))

    image_asin_list = [image_name_output,cart_text]
    #print("image_asin_list ",image_asin_list)
  
    predict = obj.predict(image_asin_list)
    
    return "# Inference Result\n\n| Product Name | ASIN | Quantity | In Stock |\n| --- | --- | --- | --- |\n" + "\n".join([f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |" for row in predict])

    #return predict

# Gradio interface for adding items to the cart
with gr.Blocks() as app:
    gr.Markdown("### Add Item to Cart")
    with gr.Row():
        item_input = gr.Dropdown(label="Item Name", choices=items)
        quantity_input = gr.Number(label="Quantity")
        add_button = gr.Button("Add to Bin")
        
    cart_text = gr.Textbox(label="Added Items", value="", visible=False)    
    added_item_display =  gr.Markdown(label="Added Items to Bin - ")
    add_button.click(add_to_cart, inputs=[item_input, quantity_input], outputs=[cart_text,added_item_display])
    
    # cart_markdown = gr.Markdown()
    # clear_button = gr.Button("Clear Bin")
    # clear_button.click(clear_bin, outputs=[cart_markdown])
    


    gr.Markdown("### Display the most Relevant Item Image")
    with gr.Row():
        #display_input = gr.Textbox(label="Item Name to Display")
        display_button = gr.Button("Display Image")
  
    display_output = gr.Image(label="Display Relevant Image", width=400, height=400)
    image_name_output = gr.Textbox(label="Image Name", visible=False)
    display_button.click(get_relevant_images, inputs=[cart_text], outputs=[display_output,image_name_output])
    
    gr.Markdown("### Validate if items and its respective quantity in order exists in the bin image")
    with gr.Row():
        display_button = gr.Button("Validate Item Order and Bin Image")

    ineference_result = gr.Markdown(label="Ineference Result")

    display_button.click(validate_item_bin_image, inputs=[image_name_output, cart_text], outputs=[ineference_result])



app.launch()
