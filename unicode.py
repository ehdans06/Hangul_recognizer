from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

font = ImageFont.truetype('NanumBarunGothic.ttf', 14)

for i in range(0xac00, 0xd7a4):
	filename = chr(i) + '.png'
	img = Image.new('L', (28, 28))
	draw = ImageDraw.Draw(img)
	draw.text((7,7), chr(i), (255,255,255), font = font)
	img = img.rotate(45)
	img.save(filename)

