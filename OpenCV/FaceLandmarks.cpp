//#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "FaceLandmarks.h"
#include "frangi.h"

FaceLandmarks::FaceLandmarks() { }

FaceLandmarks::~FaceLandmarks() {
	facemark.release();
	image.release();
	mask.release();
}

//-------------------------- Initializing --------------------//

//Setting up Facemark
void FaceLandmarks::InitializeFacemark(string _modelAddress) {
	facemark = createFacemarkLBF();
	facemark->loadModel(_modelAddress);
}

//Setting up Haar Cascade (Should be called in the Init)
bool FaceLandmarks::InitializeHaar(string _haarAddress) {
	cascadeFace.load(_haarAddress);
	return 0;
}

//Declare 3d head position with face location, used for getting 3d head rotation
void FaceLandmarks::headPositionInitialization() {
	
	modelPoints.push_back(Point3f(0.0f, 0.0f, 0.0f)); //Nose
	modelPoints.push_back(Point3f(0.0f, -330.0f, -65.0f)); //Chin
	modelPoints.push_back(Point3f(-225.0f, 170.0f, -135.0f)); //Left eye
	modelPoints.push_back(Point3f(225.0f, 170.0f, -135.0f));  //Right eye
	modelPoints.push_back(Point3f(-150.0f, -150.0f, -125.0f)); //Left side of mouth
	modelPoints.push_back(Point3f(150.0f, -150.0f, -125.0f));  //Right side of mouth

}

//If you have your own face location
void FaceLandmarks::initialize(Mat _image, Mat _mask, Rect _face) {
	
	image = Mat(_image.rows, _image.cols, CV_8UC3);
	mask = Mat(_image.rows, _image.cols, CV_8UC3);
	gray = Mat(_image.rows, _image.cols, CV_8UC1);
	YCbCr = Mat(_image.rows, _image.cols, CV_8UC3);
	hsv = Mat(_image.rows, _image.cols, CV_8UC3);

	image = _image.clone();
	face = _face;
	
	cvtColor(_image, gray, COLOR_BGR2GRAY);
	cvtColor(_image, YCbCr, COLOR_BGR2YCrCb);
	cvtColor(_image, hsv, COLOR_BGR2HSV);
}

//Use this if haar cascade is being used, call after this
void FaceLandmarks::initialize(Mat _image, Mat _mask) {
	mask = Mat(_image.rows, _image.cols, CV_8UC3); //Should be 8UC1, consider changing soon
	image = Mat(_image.rows, _image.cols, CV_8UC3);
	gray = Mat(_image.rows, _image.cols, CV_8UC1);
	YCbCr = Mat(_image.rows, _image.cols, CV_8UC3);
	hsv = Mat(_image.rows, _image.cols, CV_8UC3);

	image = _image.clone();

	cvtColor(_image, gray, COLOR_BGR2GRAY);
	cvtColor(_image, YCbCr, COLOR_BGR2YCrCb);
	cvtColor(_image, hsv, COLOR_BGR2HSV);
}

//---------------------------- Skin Segmentation techniques -------------------//

float distance(Point2f coord1, Point2f coord2) {
	float dx = coord1.x - coord2.x;
	float dy = coord1.y - coord2.y;
	float dist = sqrt(dx*dx + dy * dy);
	return dist;
}

void FaceLandmarks::colourSegmentation() {

	typedef Point3_<uint8_t> Pixel;
	int low_Y = 80, low_Cb = 85, low_Cr = 135;
	float low_H = 50, low_S = 0.23, low_V = 0.68;
	int low_R = 95, low_G = 40, low_B = 20;

	int Y, Cb, Cr, H, S, V, R, G, B;

	int startPixel = 0;
	int endPixel = mask.cols * mask.rows;

	Mat skin = image.clone();
	contour = Mat(image.cols, image.rows, CV_8UC1);

	//parallel_for_(Range(startPixel, endPixel), [&](const Range& r) {
	   //Change loop to only loop over face area
	for (int i = startPixel; i < endPixel; i++) {

		Y = YCbCr.ptr<unsigned char>()[i * 3 + 0];
		Cb = YCbCr.ptr<unsigned char>()[i * 3 + 1];
		Cr = YCbCr.ptr<unsigned char>()[i * 3 + 2];

		B = image.ptr<unsigned char>()[i * 3 + 0];
		G = image.ptr<unsigned char>()[i * 3 + 1];
		R = image.ptr<unsigned char>()[i * 3 + 2];

		if (R > low_R && G > low_G && B > low_B && R > G && R > B && R - G > 15
			||
			Y > low_Y && Cb > low_Cb && Cr > low_Cr
			&& Cr <= (1.5862 * Cb) + 20
			&& Cr >= (0.3448 * Cb) + 6.2069
			&& Cr >= (-4.5652 * Cb) + 234.5652
			&& Cr <= (-1.15 * Cb) + 301.75
			&& Cr <= (2.2857 * Cb) + 432.85) {

			contour.ptr<unsigned char>()[i] = 255;

			mask.ptr<unsigned char>()[i * 3 + 0] = 255;
			mask.ptr<unsigned char>()[i * 3 + 1] = 255;
			mask.ptr<unsigned char>()[i * 3 + 2] = 255;
		}
		else {
			mask.ptr<unsigned char>()[i * 3 + 0] = 0;
			mask.ptr<unsigned char>()[i * 3 + 1] = 0;
			mask.ptr<unsigned char>()[i * 3 + 2] = 0;
			contour.ptr<unsigned char>()[i] = 0;
		}
	}

	//Use a kernel to fill in any small gapes in the face (expensive),
	if (face.width > 100) {
		morphologyEx(mask, mask, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(5, 5)));
	}
	else {
		//morphologyEx(mask, mask, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(3, 3)));
	}

	Mat thresh;

	cvtColor(mask, thresh, COLOR_BGR2GRAY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(thresh, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

	mask.setTo(Scalar(0, 0, 0));

	//Cycles through contouring areas and compares the distance from the face location
	//check if contouring area is within the face area instead
	for (int i = 0; i < contours.size(); i++) {

		if (contourArea(contours[i]) > face.width) {
			Point2f location(face.x + (face.x + face.width) / 2, (face.y + face.y + face.height) / 2);
			RotatedRect temp = minAreaRect(Mat(contours[i]));

			if (distance(location, temp.center) < face.height * 0.7) {
				drawContours(mask, contours, i, Scalar(255, 255, 255), FILLED, LINE_8, hierarchy);
				//ellipse(mask, temp, Scalar(0, 255, 255), 2, 8);
			}
		}
	}

}

//Method relies on the user being in good light, but works
//Relies on colour segmentation to be working properly
Mat FaceLandmarks::getSkinColour() {

	vector<Mat> hsvPlanes;
	split(hsv, hsvPlanes);

	Mat hHist, sHist, vHist;
	int histH = 400, histW = 512;

	int histSize = 256;

	float range[] = { 0, 256 };

	const float* histRange = { range };

	calcHist(&hsvPlanes[0], 1, 0, Mat(), hHist, 1, &histSize, &histRange, true, true);
	calcHist(&hsvPlanes[1], 1, 0, Mat(), sHist, 1, &histSize, &histRange, true, true);
	calcHist(&hsvPlanes[2], 1, 0, Mat(), vHist, 1, &histSize, &histRange, true, true);

	int binW = cvRound((double)histW / histSize);

	Mat hist(histH, histW, CV_8UC3, Scalar(0, 0, 0));

	normalize(hHist, hHist, 0, hist.rows, NORM_MINMAX, -1, Mat());
	normalize(sHist, sHist, 0, hist.rows, NORM_MINMAX, -1, Mat());
	normalize(vHist, vHist, 0, hist.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(hist, Point(binW*(i - 1), histH - cvRound(hHist.at<float>(i - 1))),
			Point(binW*(i), histH - cvRound(hHist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(hist, Point(binW*(i - 1), histH - cvRound(sHist.at<float>(i - 1))),
			Point(binW*(i), histH - cvRound(sHist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(hist, Point(binW*(i - 1), histH - cvRound(vHist.at<float>(i - 1))),
			Point(binW*(i), histH - cvRound(vHist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	return hist;

	/*
	float v = 0;
	int count = 0;

	for (int pixel = 0; pixel < mask.cols * mask.rows; pixel++) {
		if (mask.ptr<unsigned char>()[pixel * 3] >= 255) {
			v += _hsv.ptr<unsigned char>()[pixel * 3 + 3];
			count++;
		}
	}
	

	v /= count;
	

	//150 being white 0 being black
	//Will need tweaking in the future to check h and s
	if (v > 140) {
		//LOG("Very light skin");
	}
	else if (v > 132) {
		//LOG("Light skin");
	}
	else if (v > 116) {
		//LOG("Medium skin")
	}
	else if (v > 103) {
		//LOG("Olive skin")
	}
	else if (v > 65) {
		//LOG("Brown skin")
	}
	else if (v < 66) {
		//LOG("Black skin")
	}
	*/
}

//Test techniques below are not being used
void FaceLandmarks::scaleEllipse(Mat _gray) {

	threshold(_gray, _gray, 127, 255, THRESH_BINARY);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(_gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<RotatedRect> ellipses;

	for (int i = 0; i < contours.size(); i++) {
		if (contours[i].size() >= 100) {
			RotatedRect temp = fitEllipse(Mat(contours[i]));
			if (temp.size.area() >= 1.1 * contourArea(contours[i])) {
				ellipses.push_back(temp);
				drawContours(mask, contours, i, Scalar(255, 255, 255), -1, 8);
				//ellipse(_mask, temp, Scalar(0, 255, 255), 2, 8);
			}
		}
	}
}

void FaceLandmarks::histogramProjection(Mat _hsv) {

	float hranges[] = { 0, 180 };
	const float *phranges = hranges;
	int ch[] = { 0, 0 };

	//Swap out range depending on found journals, find most appropriate
	//inRange(_hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
	//        Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), grayMask);

	//int vmin = 10, vmax = 256, smin = 30;
	//inRange(_hsv, Scalar(0, smin, MIN(vmin, vmax)), Scalar(180, 256, MAX(vmin, vmax)), mask);
	hue.create(_hsv.size(), _hsv.depth());

	//colourSegmentation();

	mixChannels(&_hsv, 1, &hue, 1, ch, 1);
	if (tracking < 0) {
		Mat roi(hue, face), maskroi(mask, face);
		calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
		normalize(hist, hist, 0, 255, NORM_MINMAX);
		tracking = 1;
	}

	calcBackProject(&hue, 1, 0, hist, backProj, &phranges);
	backProj &= mask;
	RotatedRect trackBox = CamShift(backProj, face, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 1, 1));

	//Stops it crashing if the Camshift area goes too small
	if (face.area() <= 1) {
		int cols = backProj.cols;
		int rows = backProj.rows;
		int r = (MIN(cols, rows) + 5) / 6;
		face = Rect(face.x - r, face.y - r, face.x + r, face.y + r) &
			Rect(0, 0, cols, rows);
	}

	cvtColor(backProj, mask, COLOR_GRAY2BGR);
	ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);

	//Use a kernel to fill in any small gapes in the face
	morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
}

//----------------------------- Face and Feature detection ----------------------//

//Haar cascade method, this locates and finds the biggest face on the screen and passes it out
bool FaceLandmarks::haarCascade() {

	vector<Rect> selection;
	cascadeFace.detectMultiScale(image, selection, 1.1, 2, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_SCALE_IMAGE, Size(30, 30));

	if (selection.size() > 0) {
		//LOG("Haar found a face")

		int b = getBiggest(selection);
		face = selection[b];

		tracking = -1;
		return 1;
	}

	return 0;
}

//Cycles through all faces found and finds the biggest
int FaceLandmarks::getBiggest(vector<Rect> _selection) {
	int face = 0;
	if (_selection.size() > 1) {

		int biggest = 0;
		for (int i = 0; i < _selection.size(); i++) {
			if (_selection[i].height > biggest) {
				face = i;
				biggest = _selection[i].height;
			}
		}
	}
	return face;
}

//Called to find landmarks on the suers face (doesn't display anything)
bool FaceLandmarks::findLandmarks() {

	vector<Rect> f;
	f.push_back(face);

	if (facemark->fit(gray, f, landmarks)) {
		//LOG("Landmarks found");
		f.clear();
		return 1;
	}
	else {
		//LOG("ERROR: facemark not working");
		f.clear();
		return 0;
	}
}

//locates all found landmarks and displays them with circles
//This will mainly be used for debugging
void FaceLandmarks::placeLandmarks() {
	for (int i = 0; i < landmarks[0].size(); i++) {
		//LOG("Landmarks being placed");
		Point center(landmarks[0][i]);
		ellipse(image, center, Size(5, 5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
}

//Used for segmenting features on the face, such as eyes, nose, lips, etc
void FaceLandmarks::fillLandmarks(landmarkPositions l, Scalar _colour) {
	vector<Point> markers;

	switch (l) {
	case JAWLINE:
		for (int i = 2; i <= 14; i++) {
			markers.push_back(landmarks[0][i]);
		}
		fillConvexPoly(mask, markers, _colour, CV_32S, 0);

		break;
	case MOUTH:

		for (int i = 48; i <= 59; i++) {
			if (i == 51) i++;
			markers.push_back(landmarks[0][i]);
		}

		fillConvexPoly(mask, markers, _colour, CV_32S, 0);

		break;
	case LEFT_EYE:
		for (int i = 42; i <= 47; i++) {
			markers.push_back(landmarks[0][i]);
		}

		fillConvexPoly(mask, markers, _colour, CV_32S, 0);

		break;
	case RIGHT_EYE:
		for (int i = 36; i <= 41; i++) {
			markers.push_back(landmarks[0][i]);
		}

		fillConvexPoly(mask, markers, _colour, CV_32S, 0);

		break;
	case LEFT_BROW:
		for (int i = 22; i <= 26; i++) {
			markers.push_back(landmarks[0][i]);
		}

		fillConvexPoly(mask, markers, _colour, CV_32S, 0);

		break;
	case RIGHT_BROW:

		for (int i = 17; i <= 21; i++) {
			markers.push_back(landmarks[0][i]);
		}

		fillConvexPoly(mask, markers, _colour, CV_32S, 0);

		break;
	case NOSE:
		markers.push_back(landmarks[0][27]);
		for (int i = 31; i <= 35; i++) {
			markers.push_back(landmarks[0][i]);
		}

		fillConvexPoly(mask, markers, _colour, CV_32S, 0);

		break;

	}
	markers.clear();
}
//--------------------------------- Image Quality --------------------------------//

//Estimates the head rotation to decide whether the picture should be taken
void FaceLandmarks::headPositionEstimation() {

	Point2f temp = landmarks[0][33];

	float centerX = face.x + face.width * 0.5;
	float centerY = face.y + face.height * 0.5;

	//Declarations on feature positions
	landmarkSmallSet.push_back(landmarks[0][33]); //Nose
	landmarkSmallSet.push_back(landmarks[0][8]); //Chin
	landmarkSmallSet.push_back(landmarks[0][45]); //Left eye
	landmarkSmallSet.push_back(landmarks[0][36]); //Right eye
	landmarkSmallSet.push_back(landmarks[0][54]); //Left side of mouth
	landmarkSmallSet.push_back(landmarks[0][48]); //Right side of mouth

	//Camera matrix definition
	Point2f center = Point2f(image.cols / 2, image.rows / 2);
	double focalLength = image.cols;

	Mat cameraMatrix = (Mat_<double>(3, 3) <<
		focalLength, 0, center.x,
		0, focalLength, center.y,
		0, 0, 1);

	Mat distCoeffs = Mat::zeros(4, 1, DataType<double>::type);

	Mat rotationVector;
	Mat translationVector;

	solvePnP(modelPoints, landmarkSmallSet, cameraMatrix, distCoeffs, rotationVector, translationVector);

	//Creating directional line
	vector<Point3f> noseEnd3D;
	vector<Point2f> noseEnd2D;
	noseEnd3D.push_back(Point3f(0, 0, 300.0));

	projectPoints(noseEnd3D, rotationVector, translationVector, cameraMatrix, distCoeffs, noseEnd2D);

	line(image, landmarkSmallSet[0], noseEnd2D[0], Scalar(255, 0, 0), 5);

	landmarkSmallSet.clear();
}

//Adds up existing white pixels in face area and estimates if the lighting is good enough
bool FaceLandmarks::lightingQuality() {
	int white = 0, black = 0, count = 0;

	Mat_<Vec3b> M = mask;
	int pixels = face.width * face.height;

	for (int i = face.y; i < face.y + face.height; i++) {
		for (int j = face.x; j < face.x + face.width; j++) {

			if (M.at<Vec3b>(i, j)[0] == 255) {
				white++;
			}
		}
	}

	//Adjustable, just less than half of the face is not segmented,
	//then lighting is bad
	if (white >= pixels * 0.45) {
		//LOG("Good lighting!");
		return 1;
	}
	else {
		//LOG("Bad lighting!");
		return 0;
	}
}

//--------------------------------- Output --------------------------------//

//Gets the image and applies the mask
Mat FaceLandmarks::getImage() {
	Mat temp;

	image.copyTo(temp, mask);
	return temp;
}

Mat FaceLandmarks::getMask() {
	return mask;
}

void FaceLandmarks::faceMask() {
	
	colourSegmentation();

	//mask.setTo(Scalar(255, 255, 255));
	//if (findLandmarks()) {
	//	fillLandmarks(MOUTH, Scalar(0, 0, 0));
	//	fillLandmarks(RIGHT_EYE, Scalar(0, 0, 0));
	//	fillLandmarks(LEFT_EYE, Scalar(0, 0, 0));
	//	fillLandmarks(LEFT_BROW, Scalar(0, 0, 0));
	//	fillLandmarks(RIGHT_BROW, Scalar(0, 0, 0));
	//	fillLandmarks(NOSE, Scalar(0, 0, 0));
	//}

	//for (int i = 0; i < landmarks[0].size(); i++) {
	//	
	//	circle(image, landmarks[0][i], 6, Scalar(0, 255, 0), 4);
	//}

	//morphologyEx(mask, mask, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(5, 5)));
}