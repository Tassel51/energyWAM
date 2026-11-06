import av
import numpy as np
from energywam.data.episode_dataset import _decode_video


def test_real_h264_decode_frame_order_and_rgb(tmp_path):
    path=tmp_path/'clip.mp4'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=5)
        stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        stream.codec_context.thread_count=1
        for index in range(8):
            frame=av.VideoFrame.from_ndarray(np.full((24,32,3),32+index*20,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    frames=_decode_video(path)
    assert frames.shape==(8,24,32,3) and frames.dtype==np.uint8
    assert np.all(np.diff(frames.mean(axis=(1,2,3)))>15)
