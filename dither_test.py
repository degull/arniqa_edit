import ctypes
import numpy as np
from pathlib import Path

# 프로젝트 경로 설정 (dither.dll 파일이 있는 경로)
PROJECT_ROOT = Path(__file__).resolve().parent  # 절대 경로로 설정

# dither.dll 파일 로드
dither_lib = ctypes.CDLL(str(PROJECT_ROOT / "utils" / "dither_extension" / "dither.dll"))

# dither 함수의 인자를 설정 (함수 시그니처 맞추기)
dither_lib.dither.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags="C_CONTIGUOUS"),  # input_t (이미지 데이터)
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),  # p_t (팔레트 데이터)
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int   # nc (채널 개수)
]

# dither 함수 반환 타입 설정
dither_lib.dither.restype = None

# 테스트를 위한 입력 이미지 및 팔레트 데이터 생성
width, height, channels = 256, 256, 3  # 예시로 256x256 RGB 이미지
input_image = np.random.rand(height, width, channels).astype(np.float32)  # 랜덤 이미지 생성
palette = np.random.rand(256, 3).astype(np.float32)  # 예시로 256색 팔레트 생성 (2D 배열)

# dither 함수 호출
dither_lib.dither(input_image, palette, width, height, channels)

print("Dithering 작업 완료!")
