from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype('NanumBarunGothic.ttf', 28)

for i in range(0x30+128, 0x49+128):
    for j in range(0x21+128, 0x7F+128):
        char = bytes([i, j]).decode('euc-kr')
        img = Image.new('L', (28, 28))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, 255, font=font)
        img.save(char + '.bmp')




'''
img = Image.new('L', (28, 28))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('NanumBarunGothic.ttf', 28)
draw.text((0, 0), '뷁', (255,255,255), font=font)
img.save('sample-out.png')
'''
