from rembg import remove
from PIL import Image

"""
Extremely light-weight, AI powered, background remover by rembg
"""

input_path=""
output_path = ""

input=Image.open(input_path)
output = remove(input)
output.save(output_path)

