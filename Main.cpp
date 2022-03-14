#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/imgproc/imgproc_c.h>

#include <Windows.h>
#include <iostream>
#include <stdio.h>

#define CV_HAAR_FIND_BIGGEST_OBJECT   4
#define CV_HAAR_SCALE_IMAGE   2
#define WIN_NAME "침입자 탐지 프로그램"
#define DIFF_THRESHOLD 0.1
#define FACE_CLASSIFIER_PATH "C:\\opencv-4.1.1\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"
#define IMAGES_SAVE_PATH "C:\\Users\\Administrator\\Desktop\\detect\\"

using namespace std;
using namespace cv;

// string to Lstr 형식변환
wstring s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}


// 현재 시간정보를 문자열로 가져오는 메소드
const std::string getCurrentTS2Str() {
	time_t now = time(0);
	struct tm tstruct;
	char buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);
	return buf;
}

int main(int argc, char *argv[]) {

	// 디렉토리가 없으면 만들어줌
		// char 형식 변환
	wstring stemp = s2ws(IMAGES_SAVE_PATH);
	LPCWSTR result = stemp.c_str();
	CreateDirectory(result, NULL);

	// OpenCL을 사용할 수 있는지 테스트
	if (!ocl::haveOpenCL()) {
		cout << "에러 : OpenCL을 사용할 수 없는 시스템입니다." << endl;
		return  -1;
	}
	// 컨텍스트 생성
	ocl::Context context;
	if (!context.create(ocl::Device::TYPE_GPU)) {
		cout << " 에러 : 컨텍스트를 생성할 수 없습니다." << endl;
		return  -1;
	}
	// 장치 0 번 사용 
	ocl::Device(context.device(0));
	// OpenCL Enable 상태에서 
	ocl::setUseOpenCL(true);

	// 웹 캠 실행
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cerr << "웹 캠 디바이스를 찾을 수 없습니다." << std::endl;
		return 0;
	}

	// 윈도우 이름 설정
	namedWindow(WIN_NAME, 1);

	// 얼굴인식 템플릿 설정
	CascadeClassifier face_cascade;
	face_cascade.load(FACE_CLASSIFIER_PATH);

	cout << "'esc' 키를 이용하여 프로그램 종료 " << std::endl;

	// 이전 그레이 스케일 프레임을 저장하는 변수
	UMat frameBeforeGrayScale;

	while (true) {
		bool isFrameValid = true;
		UMat frameOriginalMat;
		UMat frame;

		try {
			capture >> frameOriginalMat; // 웹 캠 프레임 원본 크기로 설정
		}
		catch (Exception& e) {
			cerr << "프레임을 받아올 수 없습니다." << e.err << std::endl;
			isFrameValid = false;
		}

		capture.read(frame);

		// frame 기본 설정 & face 설정
		UMat frameCurrentGrayScale;
		vector<Rect> faces;
		cvtColor(frame, frameCurrentGrayScale, COLOR_BGR2GRAY);
		equalizeHist(frameCurrentGrayScale, frameCurrentGrayScale);
		face_cascade.detectMultiScale(frameCurrentGrayScale, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));

		if (frameCurrentGrayScale.size == frameBeforeGrayScale.size) {
			
			// 정규화를 통해 이전 프레임과 현재 프레임의 차이의 정도를 구함
			double errorL2 = norm(frameCurrentGrayScale, frameBeforeGrayScale, NORM_L2);
			double diff = errorL2 / (double)(frameCurrentGrayScale.rows * frameBeforeGrayScale.rows);

			// 차이가 클 경우
			if (diff >= DIFF_THRESHOLD) {

				// 얼굴을 감지한 경우
				if (faces.size() != 0) {

					// 빨간색 원으로 표시
					for (size_t i = 0; i < faces.size(); i++)
					{
						Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
						ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2),
							0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
					}


					for (int i = 0; i < faces.size(); i++) {
						// 얼굴이 프레임을 벗어나지 않았을 경우에만 진행
						if (
							0 <= faces[i].x
							&& 0 <= faces[i].width
							&& faces[i].x + faces[i].width <= frame.cols
							&& 0 <= faces[i].y
							&& 0 <= faces[i].height
							&& faces[i].y + faces[i].height <= frame.rows
							) {

							// 얼굴 부분만 잘라냄
							UMat faceFrame = frame(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));

							// 침입자의 얼굴 프레임을 이미지로 저장
							imwrite(IMAGES_SAVE_PATH + string("face_") + getCurrentTS2Str() + std::string("_") + std::to_string(i) + std::string(".jpg"), faceFrame);

							// 침입자 탐지 메세지를 화면에 출력
							putText(
								frame,
								"Intruder Detected!",
								Point(50, 230),
								CV_FONT_HERSHEY_COMPLEX_SMALL,
								1.0,
								Scalar(0, 0, 255)
							);
						}
					}
				}
				else { // 움직임만 있었을 경우
					// 움직임 감지 메시지
					putText(
						frame,
						"Moving Object Detected!",
						Point(20, 230),
						CV_FONT_HERSHEY_COMPLEX_SMALL,
						1.0,
						Scalar(255, 0, 0)
					);

					int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
					int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
					UMat faceFrame = frame(Rect(0, 0, frame_width, frame_height));

					// 움직인 프레임을 이미지로 저장
					imwrite(IMAGES_SAVE_PATH + string("moving_") +getCurrentTS2Str() + std::string("_") + std::string(".jpg"), faceFrame);
				}
			}
		}

		// 윈도우에 결과 출력
		imshow(WIN_NAME, frame);

		// 현재 프레임을 이전 프레임 저장 변수에 옮김
		frameBeforeGrayScale = frameCurrentGrayScale;

		int keyCode = waitKey(30);

		// esc 키가 눌리면 프레임 캡쳐 종료
		if (keyCode == 27) {
			break;
		}
	}

	return 0;
}