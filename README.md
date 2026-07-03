# ComputerVisionCPP

컴퓨터비전 강의 실습을 위해 작성된 C++/OpenCV 예제 모음입니다. `ComputerVision.sln` 하나의 Visual Studio 솔루션 안에 디렉토리별로 독립된 콘솔 프로젝트가 들어 있으며, 각 디렉토리는 서로 다른 컴퓨터비전 주제(색상 처리, 필터링, 엣지 검출, 형태학적 연산, 특징점 매칭, 영역 분할, 딥러닝 객체 탐지 등)를 다룹니다.

## 개발 환경

- Visual Studio 2022 (Platform Toolset v143), x64
- OpenCV 4.6.0 (`opencv_world460.lib` / `opencv_world460d.lib`)
  - 환경 변수 `OPENCV_DIR` 이 OpenCV `build` 경로를 가리키도록 설정 필요
- `LibTorchYoloV5`, `LibTorchYoloV5_Video` 두 프로젝트는 추가로 LibTorch(PyTorch C++ API)가 필요
  - 환경 변수 `LIBTORCH_DIR` 설정 필요 (`torch_cpu.lib`, `torch.lib`, `c10.lib` 사용)
- 샘플 이미지는 `images/` 폴더에 있습니다 (`lena.bmp`, `coins.jpg`, `house.bmp`, `desert.bmp`, `mountain1/2.bmp`, `notredame1/2.jpg`, `pepper.bmp`, `sailboat.bmp`, `stonewall1/2.bmp` 등).

## 실행 방법

대부분의 프로젝트는 이미지(또는 비디오) 경로를 **커맨드라인 인자**로 받는 콘솔 프로그램입니다. Visual Studio에서 실행하려면 각 프로젝트 속성의 `디버깅 > 명령 인수`에 이미지 경로를 지정한 뒤 실행(F5)하면 됩니다. 프로젝트별 필요 인자는 아래 표의 "실행 인자" 항목을 참고하세요.

---

## 1. 기초 & OpenCV 사용법

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **First** | 컬러 이미지를 그레이스케일로 변환해 원본과 나란히 출력하는 기초 실습 | `imread`, `cvtColor(COLOR_BGR2GRAY)`, `imshow` | `<이미지경로>` |
| **DrawShapes** | 빈 캔버스에 사각형/직선/타원/채워진 원을 그리는 실습 | `rectangle`, `line`, `ellipse`, `circle` (커스텀 래퍼 함수) | 없음 (콘솔 메뉴로 도형 선택: 1~4) |
| **MouseEvent** | 마우스 콜백으로 좌표 이동/클릭 및 클릭 지점 픽셀의 BGR 값을 확인 | `setMouseCallback`, `EVENT_MOUSEMOVE/LBUTTONDOWN/RBUTTONDOWN` | `<이미지경로>` |
| **Trackbar** | 트랙바로 두 이미지를 알파 블렌딩 | `createTrackbar`, `addWeighted` | `<이미지경로1> <이미지경로2>` |
| **CVPlot** | sin(x) 값을 실시간으로 갱신하며 2D 그래프로 시각화 | `plot::Plot2d`(OpenCV `plot` 모듈) | 없음 (이미지 입력 없이 데이터만 시각화) |

## 2. 픽셀 연산 & 색상/명암 처리

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **Inverse** | ⚠️ Visual Studio 기본 템플릿("Hello World!")만 남아 있는 미구현 프로젝트. 이름상 역행렬 등의 실습이 계획되었던 것으로 추정 | 없음 | - |
| **ColorChange** | 선형 변환(y = ax + b)을 이용한 밝기/대비 조절을 픽셀 단위로 직접 구현 | 픽셀 순회 + `saturate_cast` (콘솔에서 alpha, beta 입력) | `<이미지경로>` |
| **ColorConversion** | 컬러 영상을 채널별로 분리해 R/G/B 단색 강조 영상으로 재구성 | `split`, `merge`, `Mat::zeros` | `<이미지경로>` |
| **GammaCorrection** | 감마 보정(거듭제곱 변환)을 그레이스케일 이미지에 픽셀 단위로 직접 적용 | `cvtColor`, `pow` 기반 수동 연산 (콘솔에서 감마값 입력) | `<이미지경로>` |
| **Posterization** | 색상 양자화를 통한 포스터화 효과 (V2/V3 버전은 미구현) | 픽셀 순회 기반 `ColorReduceV1`, `getTickCount`로 처리시간 측정 | `<이미지경로>` |
| **Binarization** | 사용자 입력 threshold로 컬러 채널별 이진화를 직접 구현 | 픽셀 순회 기반 수동 이진화 (콘솔에서 threshold 입력) | `<이미지경로>` |
| **Histogram** | 그레이/컬러 히스토그램 계산·시각화 및 히스토그램 평활화 비교 | `Histogram1D` 헬퍼 클래스, `calcHist`, `equalizeHist`, `threshold` | `<이미지경로>` |

## 3. 필터링 & 잡음 제거

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **Blurring** | 평균 필터와 가우시안 필터의 블러링 효과 비교 | `blur`, `GaussianBlur`, `getGaussianKernel` | `<이미지경로>` |
| **Sharpening** | 언샤프 마스크(Unsharp Mask) 기법으로 이미지 선명화 | `GaussianBlur` + 행렬 연산(`image*(1+amount) - blurred*amount`) | `<이미지경로>` |
| **SharpeningFilter2D** | 3x3 커스텀 샤프닝 커널을 `filter2D`로 직접 적용 | `filter2D`, 수동 커널 구성 | `<이미지경로>` |
| **SaltPepper** | 이미지에 소금-후추(salt & pepper) 잡음을 추가 | 커스텀 `SaltPepper()` 함수 | `<이미지경로>` |
| **NoiseRemoval** | 잡음을 추가한 뒤 평균 필터와 중간값 필터로 제거·비교 | `SaltPepper()`, `blur`, `medianBlur` | `<이미지경로>` |

## 4. 엣지 검출

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **SobelEdge** | Sobel 연산자로 X/Y 방향 그래디언트를 구해 엣지 검출 및 이진화 | `Sobel`, `convertScaleAbs`, `threshold` | `<이미지경로>` (그레이스케일) |
| **LaplacianEdge** | 가우시안 블러로 노이즈를 제거한 뒤 라플라시안 필터로 엣지 검출 | `GaussianBlur`, `Laplacian`, `convertScaleAbs` | `<이미지경로>` |
| **CannyEdgeDetector** | 트랙바로 임계값을 조절하며 Canny 엣지 검출 결과를 실시간 확인 | `Canny`, `createTrackbar` | `<이미지경로>` |
| **HoughTransform** | Canny 엣지 검출 후 표준 허프 변환으로 직선 검출 | `Canny`, `HoughLines` | `<이미지경로>` (그레이스케일) |

## 5. 형태학적 연산

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **Dilation** | 이진화 영상에 대한 팽창(dilation) 연산 비교 (1회 vs 5회 반복) | `threshold`, `dilate` | `<이미지경로>` |
| **Erosion** | 이진화 영상에 대한 침식(erosion) 연산 비교 (1회 vs 5회 반복) | `threshold`, `erode` | `<이미지경로>` |
| **Thinning** | 형태학적 연산을 반복 적용해 이진 이미지의 골격(skeleton) 추출 | `morphologyEx(MORPH_OPEN)`, `erode`, `bitwise_not/and/or` | `<이미지경로>` (그레이스케일) |

## 6. 기하 변환 & 이미지 피라미드

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **PyramidUpDown** | 이미지 피라미드를 이용한 확대/축소, 키보드로 반복 적용 | `pyrUp`, `pyrDown` (키보드 'u'/'d'로 조작) | `<이미지경로>` |
| **Cropping** | 마우스 드래그로 ROI를 지정해 이미지를 잘라내고 저장 | `setMouseCallback`, `Rect`, `imwrite` (키보드 's' 저장, 'r' 초기화) | `<이미지경로>` |
| **TrackbarThresholding** | 트랙바로 이진화 방식/임계값을 조절하며 적응형 임계처리와 비교 | `adaptiveThreshold`, `threshold`, `createTrackbar` (2개) | `<이미지경로>` |

## 7. 특징점 검출 & 매칭

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **CornerExtractor** | Shi-Tomasi 방식으로 코너점 검출 | `goodFeaturesToTrack` | `<이미지경로>` (그레이스케일) |
| **CornerExtractor2** | FAST 알고리즘으로 키포인트(코너) 검출 | `FastFeatureDetector` | `<이미지경로>` |
| **KeypointMatching** | SURF 특징점 검출 및 두 이미지 간 디스크립터 매칭 | `xfeatures2d::SURF`, `BFMatcher`, `drawMatches` | `<이미지경로1> <이미지경로2>` |
| **KeypointMatching2** | SIFT 특징점 검출 및 FLANN 기반 매칭, 거리 기준 필터링 | `SIFT`, `FlannBasedMatcher`, `drawMatches` | `<이미지경로1> <이미지경로2>` |

## 8. 영역 분할 & 응용 알고리즘

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **Watershed** | Watershed 알고리즘을 이용한 전경/배경 분할 및 객체 라벨링 | `threshold(OTSU)`, `distanceTransform`, `connectedComponents`, `watershed` | `<이미지경로>` |
| **LaneDetection** | 주행 영상에서 그레이→블러→Canny→ROI 마스킹→허프 변환으로 좌/우 차선을 실시간 검출 | `GaussianBlur`, `Canny`, `fillPoly`+`bitwise_and` (ROI), `HoughLinesP`, 기울기 기반 좌/우 차선 분리 | `<비디오경로>` |

## 9. 딥러닝 기반 객체 탐지 (LibTorch + YOLOv5)

| 디렉토리 | 실습 내용 | 주요 함수/기능 | 실행 인자 |
|---|---|---|---|
| **LibTorchYoloV5** | LibTorch로 TorchScript YOLOv5 모델을 로드해 비디오에서 객체(카드) 탐지 (이미지 처리용 구버전 코드가 주석으로 남아 있음) | `torch::jit::load`, letterbox 리사이즈, `xywh2xyxy`, `cv::dnn::NMSBoxes` | `<모델(.pt)경로> <비디오경로>` |
| **LibTorchYoloV5_Video** | LibTorchYoloV5와 동일한 파이프라인의 정리된 비디오 객체 탐지 버전 | `torch::jit::load`, letterbox 리사이즈, `xywh2xyxy`, `cv::dnn::NMSBoxes` | `<모델(.pt)경로> <비디오경로>` |
