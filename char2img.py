from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle as pk

# EUC-KR
def euc_kr(image_size, font, save_img=False, save_dataset=False):
    if save_dataset:
        num_cho, num_jung, num_jong = 19, 21, 28
        data = {
            'images': [],
            'labels': []
        }

    for i in range(0x30+128, 0x49+128):
        for j in range(0x21+128, 0x7F+128):
            char = bytes([i, j]).decode('euc-kr')
            img = Image.new('L', (image_size, image_size))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), char, 255, font=font)

            if save_img:
                img.save(char + '.bmp')

            if save_dataset:
                img_data = np.asarray(img, dtype=np.uint8) / 256  # img to ndarray
                cho, jung, jong = jamo_decomposition(char)
                label_onehot = np.zeros(num_cho + num_jung + num_jong)
                label_onehot[cho] = 1
                label_onehot[num_cho + jung] = 1
                label_onehot[num_cho + num_jung + jong] = 1
                data['images'].append(img_data)
                data['labels'].append(label_onehot)

    if save_dataset:
        data['images'] = np.array(data['images']).reshape((-1, image_size, image_size, 1))
        data['labels'] = np.array(data['labels'])
        file_path = 'dataset/euc-kr.pkl'
        with open(file_path, 'wb') as f:
            pk.dump(data, f, pk.HIGHEST_PROTOCOL)


# Unicode
def unicode(image_size, font):
    for i in range(0xAC00, 0xD7A4):
        char = chr(i)
        img = Image.new('L', (image_size, image_size))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, 255, font=font)
        # img = img.rotate(45)
        img.save(char + '.bmp')

def unicode_one(image_size, font):
    img = Image.new('RGB', (image_size, image_size))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), '뷁', (255,255,255), font=font)
    img.save('unicode-one.png')

def jamo_decomposition(char):
    # Jamo decomposition
    # num_cho, num_jung, num_jong = 19, 21, 28
    # 한글유니코드값 = 0xAC00 + ( (초성순서 * 21) + 중성순서 ) * 28 + 종성순서
    hangul_code = ord(char) - int('0xAC00',16)
    cho_jung, jong = hangul_code // 28, hangul_code % 28
    cho, jung = cho_jung // 21, cho_jung % 21
    return cho, jung, jong

def ndarray_to_img(img_data):
    # ndarray to img
    #img_data.reshape((image_size, image_size))
    img_data = img_data * 256
    img = Image.fromarray(np.uint8(img_data))

def main():
    image_size = 28
    font = ImageFont.truetype('fonts/NanumBarunGothic.ttf', image_size)
    euc_kr(image_size=image_size, font=font, save_dataset=True)

if __name__ == '__main__':
    main()