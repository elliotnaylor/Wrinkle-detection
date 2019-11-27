#define _CRT_SECURE_NO_WARNINGS

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <chrono>

#include "frangi.h"
#include "houghLineP.h"
#include "FaceLandmarks.h"
#include "SpecularHighlightRemoval.h"
#include "gaborWrinkle.h"

#include <iostream>
#include <ctype.h>


using namespace cv;
using namespace std;
using namespace std::chrono;

Mat image;

bool backProj = true;
bool selectObject = false;
int tracking = 0;
Point origin;
Rect selection;

class CV_EXPORTS Range2d
{
public:
	Range2d();
	Range2d(int _start, int _end, int _start2, int _end2);
	int start, end, start2, end2;
};

Range2d::Range2d()
	: start(0), end(0), start2(0), end2(0) {}

inline
Range2d::Range2d(int _start, int _end, int _start2, int _end2)
	: start(_start), end(_end), start2(_start2), end2(_end2) {}



string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}



int main() {
	//Declarations
	VideoCapture cap;
	Rect trackWindow;
	const int MAX_ORIENTATION = 3;

	vector<Mat> gaborBank;
	double xmin[4], xmax[4];

	//2. Create and apply a gabor filter bank
	float orientations[MAX_ORIENTATION] = {
		-CV_PI / 2,
		CV_PI / 5,
		CV_PI / 2
	};

	FaceLandmarks fl;
	SpecularHighlightRemoval spec;
	WrinkleDetection wrinkle;

	int slider = 20, q = 1, w = 2, e = 15, t = 250;
	namedWindow("slider", 0);
	createTrackbar("trackbar", "slider", &q, 20);
	createTrackbar("trackbar2", "slider", &w, 20);
	createTrackbar("trackbar3", "slider", &e, 50);
	createTrackbar("Threshold", "slider", &t, 500);
	createTrackbar("density", "slider", &slider, 200);

	Mat gray, gray32, kernel, frame, gabor, gaborFinal, gaborWeights, thresh, contour, wrinkles, roughness, score, mask;

	//fl.InitializeFacemark("lbfmodel.yaml");
	fl.InitializeHaar("haarcascade_frontalface.xml");

	char* location = new char[100];
	char* roughnessLoc = new char[100];
	char* wrinkleLoc = new char[100];
	char* threshLoc = new char[100];
	char* gaborLoc = new char[100];
	char* specLoc = new char[100];
	char* diffuseLoc = new char[100];
	char* colourSeg = new char[100];

	for (int imageTick = 5612; imageTick < 7612; imageTick++) {
		sprintf(location, "D:\\Downloads\\ApparentAgev2\\00%i.jpg", imageTick);
		sprintf(roughnessLoc, "D:\\Downloads\\Complete\\00%i-roughness.jpg", imageTick);
		sprintf(wrinkleLoc , "D:\\Downloads\\Complete\\00%i-wrinkle.jpg", imageTick);
		sprintf(threshLoc, "D:\\Downloads\\Complete\\00%i-thresh.jpg", imageTick);
		sprintf(gaborLoc, "D:\\Downloads\\Complete\\00%i-gabor.jpg", imageTick);
		sprintf(specLoc, "D:\\Downloads\\Complete\\00%i-specular.jpg", imageTick);
		sprintf(diffuseLoc, "D:\\Downloads\\Complete\\00%i-diffuse.jpg", imageTick);
		sprintf(colourSeg, "D:\\Downloads\\Complete\\00%i-segmented.jpg", imageTick);
	
		frame = imread(location);
		frame.copyTo(mask);
		mask.setTo(Scalar(0, 0, 0));
		fl.initialize(frame, mask);

		spec.initialize(frame.rows, frame.cols);
		wrinkle.initialize(frame.rows, frame.cols, fl.face);

		//If there is a face, carry on, if not skip image
		if (fl.haarCascade()) {
			fl.colourSegmentation();
			fl.faceMask();
			wrinkle.run(frame, fl.getMask());
			spec.run(fl.getImage());

			//Save the images
			imwrite(roughnessLoc, wrinkle.getRoughness());
			imwrite(wrinkleLoc, wrinkle.getWrinkles());
			imwrite(threshLoc, wrinkle.getThresh());
			imwrite(gaborLoc, wrinkle.getGabor());
			imwrite(specLoc, spec.specularImage);
			imwrite(diffuseLoc, spec.diffuseImage);

			Mat im;
			frame.copyTo(im, fl.getImage());

			imwrite(colourSeg, im);

			cout << "Finished image: " << imageTick << endl;
	}
	
	//Hit esc to quit
	char c = (char)waitKey(10);
	if (c == 27)
		break;
	}

	return 0;
}