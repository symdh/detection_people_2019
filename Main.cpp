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
#define WIN_NAME "ħ���� Ž�� ���α׷�"
#define DIFF_THRESHOLD 0.1
#define FACE_CLASSIFIER_PATH "C:\\opencv-4.1.1\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"
#define IMAGES_SAVE_PATH "C:\\Users\\Administrator\\Desktop\\detect\\"

using namespace std;
using namespace cv;

// string to Lstr ���ĺ�ȯ
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


// ���� �ð������� ���ڿ��� �������� �޼ҵ�
const std::string getCurrentTS2Str() {
	time_t now = time(0);
	struct tm tstruct;
	char buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);
	return buf;
}

int main(int argc, char *argv[]) {

	// ���丮�� ������ �������
		// char ���� ��ȯ
	wstring stemp = s2ws(IMAGES_SAVE_PATH);
	LPCWSTR result = stemp.c_str();
	CreateDirectory(result, NULL);

	// OpenCL�� ����� �� �ִ��� �׽�Ʈ
	if (!ocl::haveOpenCL()) {
		cout << "���� : OpenCL�� ����� �� ���� �ý����Դϴ�." << endl;
		return  -1;
	}
	// ���ؽ�Ʈ ����
	ocl::Context context;
	if (!context.create(ocl::Device::TYPE_GPU)) {
		cout << " ���� : ���ؽ�Ʈ�� ������ �� �����ϴ�." << endl;
		return  -1;
	}
	// ��ġ 0 �� ��� 
	ocl::Device(context.device(0));
	// OpenCL Enable ���¿��� 
	ocl::setUseOpenCL(true);

	// �� ķ ����
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cerr << "�� ķ ����̽��� ã�� �� �����ϴ�." << std::endl;
		return 0;
	}

	// ������ �̸� ����
	namedWindow(WIN_NAME, 1);

	// ���ν� ���ø� ����
	CascadeClassifier face_cascade;
	face_cascade.load(FACE_CLASSIFIER_PATH);

	cout << "'esc' Ű�� �̿��Ͽ� ���α׷� ���� " << std::endl;

	// ���� �׷��� ������ �������� �����ϴ� ����
	UMat frameBeforeGrayScale;

	while (true) {
		bool isFrameValid = true;
		UMat frameOriginalMat;
		UMat frame;

		try {
			capture >> frameOriginalMat; // �� ķ ������ ���� ũ��� ����
		}
		catch (Exception& e) {
			cerr << "�������� �޾ƿ� �� �����ϴ�." << e.err << std::endl;
			isFrameValid = false;
		}

		capture.read(frame);

		// frame �⺻ ���� & face ����
		UMat frameCurrentGrayScale;
		vector<Rect> faces;
		cvtColor(frame, frameCurrentGrayScale, COLOR_BGR2GRAY);
		equalizeHist(frameCurrentGrayScale, frameCurrentGrayScale);
		face_cascade.detectMultiScale(frameCurrentGrayScale, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));

		if (frameCurrentGrayScale.size == frameBeforeGrayScale.size) {
			
			// ����ȭ�� ���� ���� �����Ӱ� ���� �������� ������ ������ ����
			double errorL2 = norm(frameCurrentGrayScale, frameBeforeGrayScale, NORM_L2);
			double diff = errorL2 / (double)(frameCurrentGrayScale.rows * frameBeforeGrayScale.rows);

			// ���̰� Ŭ ���
			if (diff >= DIFF_THRESHOLD) {

				// ���� ������ ���
				if (faces.size() != 0) {

					// ������ ������ ǥ��
					for (size_t i = 0; i < faces.size(); i++)
					{
						Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
						ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2),
							0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
					}


					for (int i = 0; i < faces.size(); i++) {
						// ���� �������� ����� �ʾ��� ��쿡�� ����
						if (
							0 <= faces[i].x
							&& 0 <= faces[i].width
							&& faces[i].x + faces[i].width <= frame.cols
							&& 0 <= faces[i].y
							&& 0 <= faces[i].height
							&& faces[i].y + faces[i].height <= frame.rows
							) {

							// �� �κи� �߶�
							UMat faceFrame = frame(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));

							// ħ������ �� �������� �̹����� ����
							imwrite(IMAGES_SAVE_PATH + string("face_") + getCurrentTS2Str() + std::string("_") + std::to_string(i) + std::string(".jpg"), faceFrame);

							// ħ���� Ž�� �޼����� ȭ�鿡 ���
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
				else { // �����Ӹ� �־��� ���
					// ������ ���� �޽���
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

					// ������ �������� �̹����� ����
					imwrite(IMAGES_SAVE_PATH + string("moving_") +getCurrentTS2Str() + std::string("_") + std::string(".jpg"), faceFrame);
				}
			}
		}

		// �����쿡 ��� ���
		imshow(WIN_NAME, frame);

		// ���� �������� ���� ������ ���� ������ �ű�
		frameBeforeGrayScale = frameCurrentGrayScale;

		int keyCode = waitKey(30);

		// esc Ű�� ������ ������ ĸ�� ����
		if (keyCode == 27) {
			break;
		}
	}

	return 0;
}