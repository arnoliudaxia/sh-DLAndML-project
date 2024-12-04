import re


def whiten_data(data, channel_mean, channel_std):
    """白化数据（标准化）"""
    return (data - channel_mean[:, None]) / channel_std[:, None]



def remove_symbols(text):
    # 使用正则表达式匹配中文字符、字母和数字，去除其他符号
    cleaned_text = re.sub(r'[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]', '', text)
    return cleaned_text