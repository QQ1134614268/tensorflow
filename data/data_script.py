import os.path

import requests


# python<3.9 _ssl v1.1.0: 1. 升级到python3.9; 2.降低urllib3版本1.x; 3.重新编译替换openssl
def download_minist_data(url, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 图片链接
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
    }
    r = requests.get(url, headers=headers)

    with open(os.path.join(save_dir, url.split("/")[-1]), mode="wb") as f:
        f.write(r.content)  # 图片内容写入文件


if __name__ == '__main__':
    arr = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ]
    for v in arr:
        download_minist_data(v, "mnist")
