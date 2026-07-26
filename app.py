import streamlit as st
import numpy as np, cv2, requests, time, io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

API_URL = 'http://127.0.0.1:8000'

st.set_page_config(page_title='실시간 AI 탐지', page_icon='🎥', layout='wide')
st.title('🎥 실시간 객체 탐지 (라이브 스트림)')
st.caption('셔터 없이 카메라 영상이 흐르고, 프레임마다 자동으로 예측합니다.')

conf = st.sidebar.slider('신뢰도(conf)', 0.0, 1.0, 0.25, 0.05)
every = st.sidebar.slider('몇 프레임마다 예측', 1, 10, 3,
                          help='숫자가 크면 가볍고 덜 부드럽습니다')
try:
    ok = requests.get(f'{API_URL}/health', timeout=5).json()
    st.sidebar.success(f"API 정상 · 클래스 {ok['classes']}종")
except Exception:
    st.sidebar.error('예측 API에 연결 불가 (Part A 셀 실행 확인)')

# 구글 STUN 서버 (브라우저-서버 연결에 필요)
RTC_CONF = RTCConfiguration({'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]})

class Detector(VideoProcessorBase):
    def __init__(self):
        self.conf = 0.25
        self.every = 3
        self.i = 0
        self.last = None          # 마지막 박스 결과(프레임 건너뛸 때 재사용)

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')   # 카메라 프레임(BGR)
        self.i += 1
        # every 프레임마다 한 번만 API 호출(속도 조절)
        if self.i % self.every == 0:
            ok, buf = cv2.imencode('.jpg', img)
            try:
                r = requests.post(f'{API_URL}/predict-image',
                                  files={'file': ('f.jpg', buf.tobytes(), 'image/jpeg')},
                                  params={'conf': self.conf}, timeout=10)
                arr = np.frombuffer(r.content, np.uint8)
                self.last = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # 박스 그린 이미지
            except Exception:
                pass
        out = self.last if self.last is not None else img
        import av
        return av.VideoFrame.from_ndarray(out, format='bgr24')

ctx = webrtc_streamer(
    key='live',
    video_processor_factory=Detector,
    rtc_configuration=RTC_CONF,
    media_stream_constraints={'video': True, 'audio': False},
    async_processing=True,
)

# 슬라이더 값을 실시간으로 프로세서에 전달
if ctx.video_processor:
    ctx.video_processor.conf = conf
    ctx.video_processor.every = every
