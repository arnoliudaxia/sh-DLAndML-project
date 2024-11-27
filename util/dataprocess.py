

def whiten_data(data, channel_mean, channel_std):
    """白化数据（标准化）"""
    return (data - channel_mean[:, None]) / channel_std[:, None]



