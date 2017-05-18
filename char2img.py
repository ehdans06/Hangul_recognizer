from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle as pk

# EUC-KR
def euc_kr(save_img=False, save_dataset=False):
    if save_dataset:
        num_cho, num_jung, num_jong = 19, 21, 28
        data = {
            'images': [],
            'labels_cho': [],
            'labels_jung': [],
            'labels_jong': []
        }

    font = ImageFont.truetype('fonts/NanumBarunGothic.ttf', 28)
    for i in range(0x30+128, 0x49+128):
        for j in range(0x21+128, 0x7F+128):
            char = bytes([i, j]).decode('euc-kr')
            img = Image.new('L', (28, 28))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), char, 255, font=font)

            if save_img:
                img.save(char + '.bmp')

            if save_dataset:
                img_data = np.asarray(img, dtype=np.uint8) / 256  # img to ndarray
                cho, jung, jong = jamo_decomposition(char)
                cho_onehot, jung_onehot, jong_onehot = np.zeros(num_cho), np.zeros(num_jung), np.zeros(num_jong)
                cho_onehot[cho] = 1
                jung_onehot[jung] = 1
                jong_onehot[jong] = 1
                data['images'].append(img_data)
                data['labels_cho'].append(cho_onehot)
                data['labels_jung'].append(jung_onehot)
                data['labels_jong'].append(jong_onehot)

    if save_dataset:
        data['images'] = np.array(data['images']).reshape((-1, 28, 28, 1))
        data['labels_cho'] = np.array(data['labels_cho'])
        data['labels_jung'] = np.array(data['labels_jung'])
        data['labels_jong'] = np.array(data['labels_jong'])
        file_path = 'dataset/euc-kr.pkl'
        with open(file_path, 'wb') as f:
            pk.dump(data, f, pk.HIGHEST_PROTOCOL)


# Unicode
def unicode():
    font = ImageFont.truetype('fonts/NanumBarunGothic.ttf', 28)
    for i in range(0xAC00, 0xD7A4):
        char = chr(i)
        img = Image.new('L', (28, 28))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, 255, font=font)
        # img = img.rotate(45)
        img.save(char + '.bmp')

def unicode_one():
    font = ImageFont.truetype('fonts/NanumBarunGothic.ttf', 28)
    img = Image.new('RGB', (28, 28))
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

def ndarray_to_img():
    # ndarray to img
    img_data = np.array([[1,2],[3,4]])
    img_data = img_data * 256
    img = Image.fromarray(np.uint8(img_data.reshape((28, 28))))

def main():
    euc_kr(save_dataset=True)

if __name__ == '__main__':
    main()